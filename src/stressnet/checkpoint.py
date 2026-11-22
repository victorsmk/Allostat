"""Save / load deployable model checkpoints (same dict format the notebook wrote)."""

from __future__ import annotations

from pathlib import Path

import torch

from .constants import MODALITY_SPECS, WINDOW_SEC
from .models import MultiModalNet


def save_checkpoint(path, model, modalities, body, num_classes, config=None, trained_on_subjects=None,
                    window_sec=WINDOW_SEC, specs=None):
    specs = specs or MODALITY_SPECS
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "modalities": list(modalities),
        "body": body,
        "num_classes": num_classes,
        "config": config or {},
        "window_sec": window_sec,
        "modality_specs": {m: specs[m] for m in modalities},
        "trained_on_subjects": trained_on_subjects or [],
    }, path)


def load_checkpoint(path, device="cpu"):
    """Rebuild a MultiModalNet from a checkpoint and put it in eval mode. Returns (model, checkpoint_dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = MultiModalNet(ckpt["modalities"], body=ckpt["body"], num_classes=ckpt["num_classes"],
                          dropout=cfg.get("dropout", 0.3), window_sec=ckpt.get("window_sec", WINDOW_SEC),
                          specs=ckpt.get("modality_specs") or None)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
