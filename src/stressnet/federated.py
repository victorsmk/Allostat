"""Federated averaging (FedAvg) simulation: every person is a client that trains on its own windows and only
shares model weights, never signals.

One round: the server sends the global weights to the selected clients; each trains ``local_epochs`` epochs on its
own data (fresh AdamW, class-balanced loss from its LOCAL label counts); the server replaces the global weights by
the average of the returned weights, weighted by each client's number of windows. Batch-norm running statistics are
averaged like any other floating-point tensor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .constants import WINDOW_SEC
from .data.torch_dataset import WindowDataset
from .models import MultiModalNet, count_parameters
from .training import class_weights_from_windows, get_device, move_batch


def average_states(states, weights):
    """Weighted mean of a list of state_dicts. Integer buffers (batch-norm step counters) are taken from the
    first state."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    out = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            out[key] = sum(float(wi) * s[key] for wi, s in zip(w, states))
        else:
            out[key] = first.clone()
    return out


def local_update(model, global_state, windows, num_classes, epochs, batch_size, lr, weight_decay, device, seed):
    """Train ``model`` (re-initialised from ``global_state``) on one client's windows; returns its new state."""
    model.load_state_dict(global_state)
    model.train()
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(WindowDataset(windows), batch_size=batch_size, shuffle=True, generator=gen,
                        drop_last=len(windows) > batch_size)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_windows(windows, num_classes=num_classes).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        for batch in loader:
            inputs, labels = move_batch(batch, device)
            if labels.size(0) < 2:               # batch norm needs more than one sample
                continue
            opt.zero_grad()
            criterion(model(inputs), labels).backward()
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fedavg(clients, modalities, body="cnn", num_classes=3, rounds=30, local_epochs=1, participation=1.0,
           batch_size=32, lr=1e-3, weight_decay=1e-2, dropout=0.3, device=None, seed=0, eval_fn=None,
           eval_every=5, min_windows=8, window_sec=WINDOW_SEC, specs=None, verbose=False):
    """Run FedAvg over ``clients`` ({client_id: [windows]}). Returns (global model in eval mode, history).
    ``eval_fn(model) -> dict`` is called every ``eval_every`` rounds and at the last one, for learning curves
    only (the final-round model is always the one returned; nothing is selected on it)."""
    device = device or get_device(verbose=False)
    torch.manual_seed(seed)
    work = MultiModalNet(modalities, body=body, num_classes=num_classes, dropout=dropout,
                         window_sec=window_sec, specs=specs).to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in work.state_dict().items()}
    eligible = sorted(c for c, w in clients.items() if len(w) >= min_windows)
    if not eligible:
        raise ValueError("no client has enough windows")
    rng = np.random.default_rng(seed)
    history = []
    for r in range(1, rounds + 1):
        n_sel = max(1, int(round(participation * len(eligible))))
        chosen = rng.choice(eligible, size=n_sel, replace=False)
        states, sizes = [], []
        for c in chosen:
            states.append(local_update(work, global_state, clients[c], num_classes, local_epochs, batch_size, lr,
                                       weight_decay, device, seed=int(rng.integers(2 ** 31))))
            sizes.append(len(clients[c]))
        global_state = average_states(states, sizes)
        if eval_fn is not None and (r % eval_every == 0 or r == rounds):
            work.load_state_dict(global_state)
            work.eval()
            history.append({"round": r, **eval_fn(work)})
            if verbose:
                print(f"  round {r}/{rounds}: {history[-1]}", flush=True)
    work.load_state_dict(global_state)
    return work.eval(), history


def communication_per_round(model, n_clients):
    """Bytes moved in one round (float32 weights, down + up per participating client)."""
    n_params = count_parameters(model)
    return {"parameters": n_params, "bytes_per_client": 2 * 4 * n_params, "bytes_per_round": 2 * 4 * n_params * n_clients}
