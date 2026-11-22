"""Training / evaluation loops for the multimodal deep models (single split, LOSO, final model)."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .constants import WINDOW_SEC
from .data.torch_dataset import WindowDataset
from .models import MultiModalNet


def get_device(verbose=True) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (no GPU visible)"
        print(f"Using device: {device} ({name})")
    return device


def sample_modality_mask(modalities, batch_size, drop_prob, device):
    """Per-sample, per-modality boolean mask (True = present). ``drop_prob`` is one float or a dict
    {modality: prob}. Guarantees at least one modality survives per sample."""
    if isinstance(drop_prob, dict):
        probs = torch.tensor([drop_prob.get(m, 0.0) for m in modalities], device=device)
    else:
        probs = torch.full((len(modalities),), float(drop_prob), device=device)

    present = torch.rand(batch_size, len(modalities), device=device) >= probs.unsqueeze(0)
    none_present = ~present.any(dim=1)
    if none_present.any():
        forced_idx = torch.randint(0, len(modalities), (int(none_present.sum()),), device=device)
        present[none_present.nonzero(as_tuple=True)[0], forced_idx] = True
    return {name: present[:, i] for i, name in enumerate(modalities)}


def class_weights_from_windows(windows, label_field="model_label", num_classes=4):
    """Inverse-frequency class weights (normalized to mean 1) from the training split only."""
    counts = np.zeros(num_classes)
    for w in windows:
        counts[w[label_field]] += 1
    inv = 1.0 / np.maximum(counts, 1)
    return torch.tensor(inv * (num_classes / inv.sum()), dtype=torch.float32)


def move_batch(batch, device):
    chest = {ch: t.to(device) for ch, t in batch["chest"].items()}
    wrist = {ch: t.to(device) for ch, t in batch["wrist"].items()}
    return {"chest": chest, "wrist": wrist}, batch["label"].to(device)


def run_epoch(model, loader, criterion, device, optimizer=None, modality_dropout_p=0.0,
              fixed_modality_mask=None):
    """One pass over ``loader``. ``modality_dropout_p`` randomly masks modalities (training only);
    ``fixed_modality_mask`` {name: False} forces modalities absent for every sample (evaluation)."""
    is_train = optimizer is not None
    model.train(is_train)
    all_true, all_pred = [], []
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            inputs, labels = move_batch(batch, device)
            if is_train and modality_dropout_p:
                mask = sample_modality_mask(model.modalities, labels.size(0), modality_dropout_p, device)
            elif fixed_modality_mask is not None:
                mask = {name: torch.full((labels.size(0),), present, dtype=torch.bool, device=device)
                        for name, present in fixed_modality_mask.items()}
            else:
                mask = None
            logits = model(inputs, modality_mask=mask)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            all_true.append(labels.detach().cpu().numpy())
            all_pred.append(logits.detach().argmax(1).cpu().numpy())
    y_true, y_pred = np.concatenate(all_true), np.concatenate(all_pred)
    return {
        "loss": total_loss / n,
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "y_true": y_true, "y_pred": y_pred,
    }


def train_deep_model(modalities, body, train_windows, val_windows, num_classes=4, epochs=30, batch_size=32,
                     lr=1e-3, weight_decay=1e-2, dropout=0.3, modality_dropout_p=0.0, verbose=True,
                     select_best=True, device=None, window_sec=WINDOW_SEC, specs=None, seed=None):
    """Train on ``train_windows``, tracking ``val_windows`` every epoch.

    select_best=True restores the epoch with the best VAL macro-F1 (only valid when val is a tuning split).
    In LOSO val IS the test subject, so use select_best=False and evaluate the final epoch.
    """
    device = device or get_device(verbose=False)
    if seed is not None:
        torch.manual_seed(seed)
    train_loader = DataLoader(WindowDataset(train_windows), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(WindowDataset(val_windows), batch_size=batch_size, shuffle=False, num_workers=0)

    model = MultiModalNet(modalities, body=body, num_classes=num_classes, dropout=dropout,
                          window_sec=window_sec, specs=specs).to(device)
    weights = class_weights_from_windows(train_windows, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history, best_val_f1, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, optimizer=optimizer,
                       modality_dropout_p=modality_dropout_p)
        va = run_epoch(model, val_loader, criterion, device, optimizer=None)
        history.append({"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"],
                        "train_macro_f1": tr["macro_f1"], "val_macro_f1": va["macro_f1"],
                        "train_balanced_acc": tr["balanced_accuracy"], "val_balanced_acc": va["balanced_accuracy"]})
        if va["macro_f1"] > best_val_f1:
            best_val_f1, best_state = va["macro_f1"], copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  epoch {epoch:2d}/{epochs} | train f1 {tr['macro_f1']:.3f} | val f1 {va['macro_f1']:.3f}")

    if select_best:
        model.load_state_dict(best_state)
    return model, history


def loso_deep_cv(modalities, body, feat_name, windows, num_classes=4, epochs=20, batch_size=32, lr=1e-3,
                 weight_decay=1e-2, dropout=0.3, modality_dropout_p=0.0, eval_subsets=None, cache_dir=None,
                 force_rebuild=False, verbose=False, select_best=True, device=None, **train_kwargs):
    """Leave-one-subject-out CV. ``eval_subsets``: {subset_name: {modality: False, ...}}; each held-out fold is
    evaluated once with all modalities ('full', always added) and once per entry using a fixed mask, so the
    hidden modality genuinely uses its learned token. Per-fold results are cached in ``cache_dir`` so an
    interrupted run (Colab disconnects) resumes where it stopped."""
    device = device or get_device(verbose=False)
    subjects = sorted({w["subject"] for w in windows})
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    class_labels = list(range(num_classes))
    eval_subsets = dict(eval_subsets or {})
    eval_subsets.setdefault("full", None)

    fold_metrics = []
    for held_out in subjects:
        fold_cache = cache_dir / f"{feat_name}_{body}_{held_out}.pkl" if cache_dir else None
        if fold_cache and fold_cache.exists() and not force_rebuild:
            with open(fold_cache, "rb") as f:
                fold_metrics.append(pickle.load(f))
            print(f"  [cache] {feat_name}/{body} held out {held_out}")
            continue

        train_w = [w for w in windows if w["subject"] != held_out]
        val_w = [w for w in windows if w["subject"] == held_out]
        model, _ = train_deep_model(
            modalities, body, train_w, val_w, num_classes=num_classes, epochs=epochs, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay, dropout=dropout, modality_dropout_p=modality_dropout_p,
            verbose=verbose, select_best=select_best, device=device, **train_kwargs)
        val_loader = DataLoader(WindowDataset(val_w), batch_size=batch_size, shuffle=False, num_workers=0)

        subset_results = {}
        for subset_name, mask in eval_subsets.items():
            r = run_epoch(model, val_loader, nn.CrossEntropyLoss(), device, optimizer=None,
                          fixed_modality_mask=mask)
            subset_results[subset_name] = {
                "balanced_accuracy": r["balanced_accuracy"],
                "macro_f1": r["macro_f1"],
                "per_class_f1": f1_score(r["y_true"], r["y_pred"], average=None, labels=class_labels),
                "confusion_matrix": confusion_matrix(r["y_true"], r["y_pred"], labels=class_labels),
                "y_true": r["y_true"], "y_pred": r["y_pred"],
            }

        fold = {"held_out_subject": held_out, "subsets": subset_results}
        if fold_cache:
            with open(fold_cache, "wb") as f:
                pickle.dump(fold, f)
        print(f"  {feat_name}/{body} held out {held_out}: "
              + "  ".join(f"{k}: f1={m['macro_f1']:.3f}" for k, m in subset_results.items()))
        fold_metrics.append(fold)
    return fold_metrics


def train_final_model(windows, modalities, body, num_classes, epochs, batch_size, lr, weight_decay, dropout,
                      seed=42, device=None, verbose=True, window_sec=WINDOW_SEC, specs=None,
                      modality_dropout_p=0.0):
    """Train on every window of every subject in ``windows``. No validation split, so the reported training
    metrics are NOT an estimate of generalization."""
    device = device or get_device(verbose=False)
    torch.manual_seed(seed)
    loader = DataLoader(WindowDataset(windows), batch_size=batch_size, shuffle=True, num_workers=0)
    model = MultiModalNet(modalities, body=body, num_classes=num_classes, dropout=dropout,
                          window_sec=window_sec, specs=specs).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_windows(windows, num_classes=num_classes).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, loader, criterion, device, optimizer=optimizer,
                       modality_dropout_p=modality_dropout_p)
        history.append({"epoch": epoch, "train_loss": tr["loss"], "train_macro_f1": tr["macro_f1"],
                        "train_balanced_accuracy": tr["balanced_accuracy"]})
        if verbose:
            print(f"  epoch {epoch:2d}/{epochs} | loss {tr['loss']:.3f} macro_f1 {tr['macro_f1']:.3f} "
                  f"bal_acc {tr['balanced_accuracy']:.3f}")
    return model, history
