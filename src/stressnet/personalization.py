"""Cheap per-user adaptation of a trained global ``MultiModalNet`` from a few labelled calibration windows.

All methods keep the encoders frozen and work on the concatenated per-modality embeddings, so adapting one
user takes milliseconds and the personal state is tiny (what a federated client would keep locally):

* ``fit_bias``  - one additive offset per class on the logits (a personal decision threshold)
* ``fit_head``  - fine-tune the classifier head, with an L2 penalty towards the global weights (L2-SP)

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


def _balanced_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(float)
    w = np.where(counts > 0, counts.sum() / (num_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def fit_bias(logits, y, num_classes=3, steps=100, lr=0.05, l2=0.01):
    """Per-class logit offset minimising class-balanced CE on the calibration set (+ small L2 to stay near 0)."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    b = torch.zeros(num_classes, requires_grad=True)
    opt = torch.optim.Adam([b], lr=lr)
    w = _balanced_weights(np.asarray(y), num_classes)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(logits + b, y_t, weight=w) + l2 * (b ** 2).sum()
        loss.backward()
        opt.step()
    return b.detach()


def fit_head(head, emb, y, num_classes=3, steps=60, lr=3e-3, l2sp=1e-2):
    """Fine-tune a copy of ``head`` on calibration embeddings with class-balanced CE and an L2-SP penalty that
    keeps the weights close to the global ones. Returns the adapted head (eval mode)."""
    adapted = copy.deepcopy(head).eval()          # eval: no dropout on a handful of examples
    ref = [p.detach().clone() for p in adapted.parameters()]
    y_t = torch.as_tensor(y, dtype=torch.long)
    emb = torch.as_tensor(emb, dtype=torch.float32)
    w = _balanced_weights(np.asarray(y), num_classes)
    opt = torch.optim.Adam(adapted.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(adapted(emb), y_t, weight=w)
        loss = loss + l2sp * sum(((p - r) ** 2).sum() for p, r in zip(adapted.parameters(), ref))
        loss.backward()
        opt.step()
    return adapted.eval()
