"""Cheap per-user adaptation of a trained global ``MultiModalNet`` from a few labelled calibration windows.

All methods keep the encoders frozen and work on the concatenated per-modality embeddings, so adapting one
user takes milliseconds and the personal state is tiny (what a federated client would keep locally):

* ``fit_bias``  - one additive offset per class on the logits (a personal decision threshold)
* ``fit_head``  - fine-tune the classifier head, with an L2 penalty towards the global weights (L2-SP)

Both can protect classes the user never labelled (``mask_absent``) and ``fit_head`` can mix in replay /
pseudo-labelled examples (``extra_emb``), see scripts/cross_session_fix.py.

Calibration windows are sampled so that neither they nor anything overlapping them can be in the test set.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.torch_dataset import WindowDataset
from .training import move_batch


@torch.no_grad()
def embed(model, windows, device, batch_size=128):
    """Concatenated per-modality embeddings (N, 64 * n_modalities), model in eval mode."""
    model.eval()
    out = []
    for batch in DataLoader(WindowDataset(windows), batch_size=batch_size, shuffle=False):
        inputs, _ = move_batch(batch, device)
        feats = []
        for name in model.modalities:
            location, ch, _, _ = model.specs[name]
            feats.append(model.encoders[name](inputs[location][ch]))
        out.append(torch.cat(feats, dim=1).cpu())
    return torch.cat(out)


def sample_calibration(windows, k_per_class, rng, min_gap_sec=60.0):
    """Pick up to ``k_per_class`` windows of every class present, no two within ``min_gap_sec`` of each other
    in the same recording session (a window is 60 s long and the step is 15 s, so neighbours overlap).
    Returns (calibration_idx, test_idx); test excludes every window overlapping a calibration window.

    A session is identified by ``dataset`` (one recording per dataset tag and subject)."""
    ids = np.arange(len(windows))
    t = np.array([w["t_start_sec"] for w in windows])
    sess = np.array([w["dataset"] for w in windows])
    y = np.array([w["model_label"] for w in windows])
    chosen = []
    for c in sorted(set(y.tolist())):
        picked = []
        for i in rng.permutation(ids[y == c]):
            if all(sess[i] != sess[j] or abs(t[i] - t[j]) >= min_gap_sec for j in picked):
                picked.append(i)
                if len(picked) == k_per_class:
                    break
        chosen.extend(picked)
    chosen = np.array(chosen, dtype=int)
    blocked = np.zeros(len(windows), dtype=bool)
    for j in chosen:
        blocked |= (sess == sess[j]) & (np.abs(t - t[j]) < min_gap_sec)
    return chosen, ids[~blocked]


def _balanced_weights(y, num_classes, sample_weight=None):
    """Class weights total / (C * count_c) computed from (optionally sample-weighted) counts; 0 for absent classes."""
    y = np.asarray(y)
    sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    counts = np.bincount(y, weights=sw, minlength=num_classes).astype(float)
    w = np.where(counts > 0, counts.sum() / (num_classes * np.maximum(counts, 1e-12)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def _present(y, num_classes):
    return torch.tensor(np.bincount(np.asarray(y), minlength=num_classes) > 0)


def _weighted_ce(z, y_t, y, num_classes, sample_weight=None):
    """Class-balanced (and optionally per-sample weighted) cross-entropy, normalised by the total weight."""
    cw = _balanced_weights(y, num_classes, sample_weight)[y_t]
    sw = torch.ones(len(y)) if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float32)
    w = cw * sw
    return (F.cross_entropy(z, y_t, reduction="none") * w).sum() / w.sum()


def fit_bias(logits, y, num_classes=3, steps=100, lr=0.05, l2=0.01, mask_absent=False):
    """Per-class logit offset minimising class-balanced CE on the calibration set (+ small L2 to stay near 0).
    ``mask_absent``: classes missing from the calibration labels are left out of the softmax and keep offset 0,
    so the adapter cannot push them down just because it never saw them."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    present = _present(y, num_classes)
    b = torch.zeros(num_classes, requires_grad=True)
    opt = torch.optim.Adam([b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        z = logits + b
        if mask_absent:
            z = z.masked_fill(~present, -1e9)
        loss = _weighted_ce(z, y_t, y, num_classes) + l2 * (b ** 2).sum()
        loss.backward()
        opt.step()
    out = b.detach()
    return out * present if mask_absent else out


def fit_head(head, emb, y, num_classes=3, steps=60, lr=3e-3, l2sp=1e-2, mask_absent=False,
             extra_emb=None, extra_y=None, extra_weight=0.3):
    """Fine-tune a copy of ``head`` on calibration embeddings with class-balanced CE and an L2-SP penalty that
    keeps the weights close to the global ones. Returns the adapted head (eval mode).

    mask_absent  - classes absent from ``y`` are excluded from the softmax in the loss
    extra_emb/y  - additional (replay or pseudo-labelled) examples, each weighted by ``extra_weight``
                   relative to a calibration example; they count towards the class balance."""
    adapted = copy.deepcopy(head).eval()          # eval: no dropout on a handful of examples
    ref = [p.detach().clone() for p in adapted.parameters()]
    emb = torch.as_tensor(emb, dtype=torch.float32)
    y_all = np.asarray(y)
    sample_w = None
    present = _present(y, num_classes)
    if extra_emb is not None and len(extra_emb):
        emb = torch.cat([emb, torch.as_tensor(extra_emb, dtype=torch.float32)])
        sample_w = np.concatenate([np.ones(len(y_all)), np.full(len(extra_y), extra_weight)])
        y_all = np.concatenate([y_all, np.asarray(extra_y)])
    y_t = torch.as_tensor(y_all, dtype=torch.long)
    opt = torch.optim.Adam(adapted.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        z = adapted(emb)
        if mask_absent and extra_emb is None:
            z = z.masked_fill(~present, -1e9)
        loss = _weighted_ce(z, y_t, y_all, num_classes, sample_w)
        loss = loss + l2sp * sum(((p - r) ** 2).sum() for p, r in zip(adapted.parameters(), ref))
        loss.backward()
        opt.step()
    return adapted.eval()
