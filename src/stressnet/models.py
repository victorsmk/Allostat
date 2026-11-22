"""Multimodal 1D-CNN / TCN with one encoder tower per modality and learned "missing modality" tokens.

Module and parameter names are identical to the original notebook, so checkpoints saved from it
(``wrist_cnn_binary_full15.pt`` etc.) load into these classes unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import MODALITY_SPECS, WINDOW_SEC, modality_input_len

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # older torch
    from torch.nn.utils import weight_norm

EMBED_DIM = 64


class Chomp1d(nn.Module):
    """Cut the extra right-padding a dilated conv adds so the output length equals the input length."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class PatchifyStem(nn.Module):
    """Downsample an arbitrary-length (B, T_in, C_in) signal to exactly (B, 16, target_len) with one strided
    conv, so a 42,000-sample chest ECG window and a 240-sample wrist EDA window look identical downstream."""

    def __init__(self, in_channels, input_len, target_len=64, out_channels=16):
        super().__init__()
        patch = max(1, input_len // target_len)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=patch, stride=patch)
        self.bn = nn.BatchNorm1d(out_channels)
        self.resize = nn.AdaptiveAvgPool1d(target_len)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T_in, C_in) -> (B, C_in, T_in)
        x = F.relu(self.bn(self.conv(x)))
        return self.resize(x)


class CNNBody(nn.Module):
    """16 -> 32 -> 64 plain conv stack."""

    def __init__(self, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TemporalBlock(nn.Module):
    """Dilated-causal TCN residual block (Bai et al., 2018)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation))
        self.chomp1 = Chomp1d(pad)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, dilation=dilation))
        self.chomp2 = Chomp1d(pad)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.dropout(self.relu(self.chomp1(self.conv1(x))))
        out = self.dropout(self.relu(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBody(nn.Module):
    """16 -> 32 -> 64 with dilation (1, 2): same widths as CNNBody, so CNN vs TCN isolates the conv type."""

    def __init__(self, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            TemporalBlock(16, 32, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock(32, 64, kernel_size=3, dilation=2, dropout=dropout),
        )

    def forward(self, x):
        return self.net(x)


class ModalityEncoder(nn.Module):
    """patchify stem -> CNN or TCN body -> GAP: one 64-dim vector per modality, whatever its rate/length."""

    def __init__(self, in_channels, input_len, body="cnn", target_len=64, dropout=0.2):
        super().__init__()
        if body not in ("cnn", "tcn"):
            raise ValueError(f"body must be 'cnn' or 'tcn', got {body!r}")
        self.stem = PatchifyStem(in_channels, input_len, target_len=target_len)
        self.body = CNNBody(dropout=dropout) if body == "cnn" else TCNBody(dropout=dropout)

    def forward(self, x):
        x = self.body(self.stem(x))
        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, 64, 64) -> (B, 64)


class MultiModalNet(nn.Module):
    """One ModalityEncoder per modality, outputs concatenated into a small head. Each modality has a learnable
    'missing' token substituted when that modality is unavailable, either structurally (absent from the batch)
    or per sample through ``modality_mask`` (training-time modality dropout).

    ``specs`` / ``window_sec`` default to the WESAD registry; pass your own to use other devices/datasets.
    """

    def __init__(self, modalities, body="cnn", num_classes=4, head_hidden=64, dropout=0.3,
                 specs=None, window_sec=WINDOW_SEC):
        super().__init__()
        self.specs = specs or MODALITY_SPECS
        self.modalities = list(modalities)
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(self.specs[name][3], modality_input_len(name, window_sec, self.specs),
                                  body=body, dropout=dropout)
            for name in self.modalities
        })
        self.missing_tokens = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(EMBED_DIM)) for name in self.modalities
        })
        for p in self.missing_tokens.values():
            nn.init.normal_(p, mean=0.0, std=0.02)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(EMBED_DIM * len(self.modalities), head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, batch, modality_mask=None):
        """``modality_mask``: optional {name: BoolTensor(B,)}, True = present. A modality with no key in
        ``batch`` always uses its missing token; the mask is for modalities that ARE in ``batch`` but should be
        hidden for some samples."""
        ref = None
        for name in self.modalities:
            location, ch, _, _ = self.specs[name]
            if location in batch and ch in batch[location]:
                ref = batch[location][ch]
                break
        if ref is None:
            raise ValueError("no modality data found in batch")
        B, device = ref.shape[0], ref.device

        feats = []
        for name in self.modalities:
            location, ch, _, _ = self.specs[name]
            token = self.missing_tokens[name]
            if location in batch and ch in batch[location]:
                emb = self.encoders[name](batch[location][ch])
                if modality_mask is not None and name in modality_mask:
                    present = modality_mask[name].to(device).view(-1, 1).float()
                    emb = present * emb + (1 - present) * token.unsqueeze(0)
            else:
                emb = token.unsqueeze(0).expand(B, -1)
            feats.append(emb)
        return self.head(torch.cat(feats, dim=1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
