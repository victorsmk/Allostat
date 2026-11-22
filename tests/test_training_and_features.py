import numpy as np
import pytest

from stressnet import classical
from stressnet.constants import WRIST_MODALITIES
from stressnet.data.splits import to_binary_windows
from stressnet.evaluation import extract_metrics, paired_comparison, summarize_subsets
from stressnet.features import build_feature_df, feature_columns, hr_features_from_peaks
from stressnet.training import class_weights_from_windows, loso_deep_cv, train_deep_model


def test_class_weights_inverse_frequency():
    wins = [{"model_label": 0}] * 8 + [{"model_label": 1}] * 2
    w = class_weights_from_windows(wins, num_classes=2).numpy()
    assert w[1] / w[0] == pytest.approx(4.0) and w.mean() == pytest.approx(1.0)


def test_hr_features_recover_known_rate():
    fs = 64
    t = np.arange(60 * fs) / fs
    f = hr_features_from_peaks(np.sin(2 * np.pi * 1.2 * t), "x", fs)  # 72 bpm
    assert f["x_hr_mean"] == pytest.approx(72, abs=2)
    assert f["x_sdnn"] < 30


def test_hr_features_nan_when_no_peaks():
    f = hr_features_from_peaks(np.zeros(640), "x", 64)
    assert np.isnan(f["x_hr_mean"]) and f["x_n_peaks"] == 0


def test_feature_df_shapes(built_windows):
    windows, _ = built_windows
    df = build_feature_df(windows, mode="wrist")
    assert len(df) == len(windows) and {"subject", "label"} <= set(df.columns)
    assert len(feature_columns(df)) == 33  # matches the notebook's wrist feature count
    with pytest.raises(KeyError):  # wrist-only windows carry no chest signals
        build_feature_df(windows, mode="full")


def test_classical_loso_beats_chance_on_synthetic(built_windows):
    windows, _ = built_windows
    df = build_feature_df(to_binary_windows(windows), mode="wrist")
    folds = classical.loso_cv(df, feature_columns(df), "logreg", labels=[0, 1])
    assert [f["held_out_subject"] for f in folds] == ["S2", "S3", "S4"]
    assert np.mean([f["macro_f1"] for f in folds]) > 0.8
    summary = classical.summarize_folds({("wrist", "logreg"): folds}, ["non_stress", "stress"])
    assert summary.loc[0, "macro_f1_mean"] > 0.8


def test_deep_training_runs_and_learns(built_windows):
    import torch
    windows, _ = built_windows
    binary = to_binary_windows(windows)
    train = [w for w in binary if w["subject"] != "S4"]
    val = [w for w in binary if w["subject"] == "S4"]
    model, hist = train_deep_model(["wrist_bvp", "wrist_eda"], "cnn", train, val, num_classes=2, epochs=6,
                                   batch_size=16, verbose=False, select_best=False,
                                   device=torch.device("cpu"), seed=0)
    assert len(hist) == 6
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert hist[-1]["val_macro_f1"] > 0.8  # synthetic stress is very separable


def test_loso_deep_cv_with_subsets_and_cache(built_windows, tmp_path, capsys):
    import torch
    windows, _ = built_windows
    binary = to_binary_windows(windows)
    kw = dict(num_classes=2, epochs=2, batch_size=16, modality_dropout_p={"wrist_acc": 0.4, "wrist_temp": 0.4},
              eval_subsets={"bvp_eda_only": {"wrist_acc": False, "wrist_temp": False}},
              cache_dir=tmp_path / "cache", select_best=False, device=torch.device("cpu"))
    res = loso_deep_cv(WRIST_MODALITIES, "cnn", "t", binary, **kw)
    assert [f["held_out_subject"] for f in res] == ["S2", "S3", "S4"]
    assert set(res[0]["subsets"]) == {"bvp_eda_only", "full"}
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 3
    capsys.readouterr()
    res2 = loso_deep_cv(WRIST_MODALITIES, "cnn", "t", binary, **kw)  # all folds come from cache
    assert capsys.readouterr().out.count("[cache]") == 3
    assert extract_metrics(res)["macro_f1"].tolist() == extract_metrics(res2)["macro_f1"].tolist()
    assert summarize_subsets(res).shape[0] == 2


def test_paired_comparison():
    import pandas as pd
    a = pd.DataFrame({"held_out": list("abcdefgh"), "macro_f1": [0.5, 0.6, 0.7, 0.55, 0.65, 0.6, 0.7, 0.5]})
    b = a.assign(macro_f1=a["macro_f1"] + [0.1, 0.08, 0.12, 0.09, 0.11, 0.07, 0.13, 0.1])
    table, p = paired_comparison(a, b, "a", "b", verbose=False)
    assert table["diff"].mean() == pytest.approx(0.1, abs=0.02) and p < 0.05
