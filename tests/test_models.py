import pytest
import torch

from stressnet.checkpoint import load_checkpoint, save_checkpoint
from stressnet.constants import FULL_MODALITIES, WRIST_MODALITIES
from stressnet.models import MultiModalNet, count_parameters
from stressnet.training import sample_modality_mask


def _wrist_batch(B=4):
    return {"chest": {}, "wrist": {"ACC": torch.randn(B, 1920, 3), "BVP": torch.randn(B, 3840, 1),
                                   "EDA": torch.randn(B, 240, 1), "TEMP": torch.randn(B, 240, 1)}}


# Parameter counts printed by the original Colab notebook; a mismatch means the architecture drifted and
# checkpoints trained there would no longer load.
@pytest.mark.parametrize("body,mods,expected", [
    ("cnn", FULL_MODALITIES, 208_548),
    ("cnn", WRIST_MODALITIES, 51_524),
    ("tcn", FULL_MODALITIES, 389_668),
    ("tcn", WRIST_MODALITIES, 123_972),
])
def test_param_counts_match_notebook(body, mods, expected):
    assert count_parameters(MultiModalNet(mods, body=body)) == expected


@pytest.mark.parametrize("body", ["cnn", "tcn"])
@pytest.mark.parametrize("num_classes", [2, 4])
def test_forward_shape(body, num_classes):
    model = MultiModalNet(WRIST_MODALITIES, body=body, num_classes=num_classes).eval()
    assert model(_wrist_batch()).shape == (4, num_classes)


def test_invalid_body_rejected():
    with pytest.raises(ValueError):
        MultiModalNet(WRIST_MODALITIES, body="lstm")


def test_absent_modality_equals_masked_modality():
    """Hiding a modality by omitting it and by masking it must give the same output: both use the token."""
    torch.manual_seed(0)
    model = MultiModalNet(WRIST_MODALITIES, body="cnn").eval()
    batch = _wrist_batch()
    absent = {"chest": {}, "wrist": {k: v for k, v in batch["wrist"].items() if k != "TEMP"}}
    mask = {"wrist_temp": torch.zeros(4, dtype=torch.bool)}
    with torch.no_grad():
        torch.testing.assert_close(model(absent), model(batch, modality_mask=mask))


def test_mask_changes_output_but_all_present_mask_does_not():
    model = MultiModalNet(WRIST_MODALITIES, body="cnn").eval()
    batch = _wrist_batch()
    with torch.no_grad():
        base = model(batch)
        same = model(batch, modality_mask={m: torch.ones(4, dtype=torch.bool) for m in WRIST_MODALITIES})
        hidden = model(batch, modality_mask={"wrist_bvp": torch.zeros(4, dtype=torch.bool)})
    torch.testing.assert_close(base, same)
    assert not torch.allclose(base, hidden)


def test_empty_batch_raises():
    with pytest.raises(ValueError, match="no modality data"):
        MultiModalNet(WRIST_MODALITIES).eval()({"chest": {}, "wrist": {}})


def test_modality_mask_keeps_at_least_one():
    torch.manual_seed(0)
    mask = sample_modality_mask(WRIST_MODALITIES, 2000, 0.99, torch.device("cpu"))
    present = torch.stack([mask[m] for m in WRIST_MODALITIES], dim=1)
    assert present.any(dim=1).all()


def test_modality_mask_rates_per_modality():
    torch.manual_seed(0)
    mask = sample_modality_mask(WRIST_MODALITIES, 20000, {"wrist_acc": 0.4, "wrist_bvp": 0.0}, torch.device("cpu"))
    assert mask["wrist_bvp"].all()
    assert 0.55 < mask["wrist_acc"].float().mean() < 0.65


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    mods = ["wrist_bvp", "wrist_eda"]
    model = MultiModalNet(mods, body="cnn", num_classes=2).eval()
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, mods, "cnn", 2, config={"dropout": 0.3}, trained_on_subjects=["S2"])
    loaded, ckpt = load_checkpoint(path)
    batch = {"chest": {}, "wrist": {"BVP": torch.randn(3, 3840, 1), "EDA": torch.randn(3, 240, 1)}}
    with torch.no_grad():
        torch.testing.assert_close(model(batch), loaded(batch))
    assert ckpt["trained_on_subjects"] == ["S2"] and ckpt["modalities"] == mods
