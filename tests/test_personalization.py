import numpy as np
import torch
from torch import nn

from stressnet import personalization as P


def _windows(n_per_class=40, step=15.0):
    ws = []
    for c in range(3):
        for i in range(n_per_class):
            ws.append({"model_label": c, "t_start_sec": 1000.0 * c + i * step, "dataset": f"d{c}"})
    return ws


def test_sample_calibration_no_overlap_and_clean_test_set():
    ws = _windows()
    rng = np.random.default_rng(0)
    cal, test = P.sample_calibration(ws, 3, rng)
    assert len(cal) == 9
    assert len(set(test) & set(cal)) == 0
    for c in cal:                                   # nothing in test overlaps a calibration window
        for i in test:
            if ws[i]["dataset"] == ws[c]["dataset"]:
                assert abs(ws[i]["t_start_sec"] - ws[c]["t_start_sec"]) >= 60.0
    for a in cal:                                   # calibration windows are >= 60 s apart within a session
        for b in cal:
            if a != b and ws[a]["dataset"] == ws[b]["dataset"]:
                assert abs(ws[a]["t_start_sec"] - ws[b]["t_start_sec"]) >= 60.0
    assert sorted({ws[i]["model_label"] for i in cal}) == [0, 1, 2]


def test_sample_calibration_takes_what_exists():
    ws = _windows(n_per_class=2)                    # 2 windows 15 s apart: only one can be picked per class
    cal, test = P.sample_calibration(ws, 5, np.random.default_rng(1))
    assert len(cal) == 3


def test_fit_bias_moves_predictions_towards_calibration_labels():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 20)
    logits = torch.tensor(rng.normal(size=(60, 3)), dtype=torch.float32)
    logits[np.arange(60), y] += 1.5                 # the model does carry class information ...
    logits[:, 0] += 2.5                             # ... but systematically over-predicts class 0
    b = P.fit_bias(logits, y)
    assert b[0] < 0 and (b[1] > 0 or b[2] > 0)
    acc = lambda lg: float((lg.argmax(1).numpy() == y).mean())
    assert acc(logits + b) > acc(logits) + 0.2


def test_fit_head_adapts_but_stays_close_to_global():
    torch.manual_seed(0)
    head = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    emb = torch.randn(45, 8)
    y = np.tile([0, 1, 2], 15)
    emb[np.arange(45), y] += 3.0                    # separable by construction
    before = float((head(emb).argmax(1).numpy() == y).mean())
    adapted = P.fit_head(head, emb, y, steps=80, lr=1e-2, l2sp=1e-3)
    after = float((adapted(emb).argmax(1).numpy() == y).mean())
    assert after >= before and after > 0.9
    drift = sum(((a - b) ** 2).sum() for a, b in zip(adapted.parameters(), head.parameters()))
    assert float(drift.detach()) < 50.0
    assert all(p.requires_grad for p in head.parameters())   # the original head is untouched


def test_mask_absent_protects_unseen_class():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 30)
    logits = torch.tensor(rng.normal(size=(60, 3)), dtype=torch.float32)
    logits[np.arange(60), y] += 2.0
    plain = P.fit_bias(logits, y)
    masked = P.fit_bias(logits, y, mask_absent=True)
    assert plain[2] < -0.05                      # unmasked: the never-seen class 2 is pushed down
    assert masked[2] == 0.0                      # masked: left alone


def test_head_replay_and_mask_keep_unseen_class():
    torch.manual_seed(0)
    head = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    centers = torch.randn(3, 8) * 3
    make = lambda c, n: centers[c] + 0.3 * torch.randn(n, 8)
    emb = torch.cat([make(0, 10), make(1, 10)])
    y = np.repeat([0, 1], 10)
    test2 = make(2, 50)

    def frac2(h):
        with torch.no_grad():
            return float((h(test2).argmax(1) == 2).float().mean())

    base = P.fit_head(head, emb, y, steps=100, lr=1e-2, l2sp=1e-3)
    masked = P.fit_head(head, emb, y, steps=100, lr=1e-2, l2sp=1e-3, mask_absent=True)
    replay = P.fit_head(head, emb, y, steps=100, lr=1e-2, l2sp=1e-3,
                        extra_emb=make(2, 10), extra_y=np.full(10, 2))
    assert frac2(base) <= frac2(head) + 1e-9     # plain fine-tuning never helps the unseen class
    assert frac2(replay) >= frac2(base)          # replaying examples of it does not hurt, usually helps
    assert frac2(masked) >= frac2(base) - 1e-9
