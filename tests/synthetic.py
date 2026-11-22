"""Synthetic WESAD-shaped data so the pipeline can be tested without the 13 GB dataset."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from stressnet.constants import CHEST_FS, WRIST_FS

# (raw label, seconds). Baseline is long enough to survive a 30 s trim and still hold windows.
DEFAULT_PROTOCOL = [(0, 10), (1, 150), (0, 10), (2, 130), (0, 10), (3, 100), (4, 70)]


def make_subject_arrays(seed=0, protocol=DEFAULT_PROTOCOL, stress_hr_boost=0.4):
    """Return the dict stored inside a WESAD SX.pkl. Stress segments get a faster pulse and more EDA so the
    classes are learnable."""
    rng = np.random.default_rng(seed)
    label_chunks = [np.full(int(sec * CHEST_FS), lab, dtype=np.int64) for lab, sec in protocol]
    label = np.concatenate(label_chunks)
    total_sec = len(label) / CHEST_FS

    def per_sample_stress(fs):
        lab_rs = np.repeat(label, 1)[:: CHEST_FS // fs] if CHEST_FS % fs == 0 else None
        if lab_rs is None:
            idx = (np.arange(int(total_sec * fs)) * CHEST_FS / fs).astype(int)
            lab_rs = label[np.minimum(idx, len(label) - 1)]
        return (lab_rs == 2).astype(float)

    def pulse(fs, base_hz):
        n = int(total_sec * fs)
        stress = per_sample_stress(fs)[:n]
        t = np.arange(n) / fs
        hz = base_hz + stress_hr_boost * stress
        phase = 2 * np.pi * np.cumsum(hz) / fs
        return np.sin(phase) + 0.05 * rng.standard_normal(n)

    chest = {
        "ACC": 0.05 * rng.standard_normal((len(label), 3)),
        "ECG": pulse(CHEST_FS, 1.2)[:, None],
        "EDA": (2.0 + 0.5 * per_sample_stress(CHEST_FS) + 0.02 * rng.standard_normal(len(label)))[:, None],
        "EMG": 0.01 * rng.standard_normal((len(label), 1)),
        "Resp": np.sin(2 * np.pi * 0.25 * np.arange(len(label)) / CHEST_FS)[:, None],
        "Temp": (33.0 + 0.01 * rng.standard_normal(len(label)))[:, None],
    }
    n = {k: int(total_sec * fs) for k, fs in WRIST_FS.items()}
    wrist = {
        "ACC": 8.0 * rng.standard_normal((n["ACC"], 3)),
        "BVP": pulse(WRIST_FS["BVP"], 1.2)[:, None],
        "EDA": (2.0 + 0.5 * per_sample_stress(WRIST_FS["EDA"])[: n["EDA"]]
                + 0.02 * rng.standard_normal(n["EDA"]))[:, None],
        "TEMP": (33.0 + 0.01 * rng.standard_normal(n["TEMP"]))[:, None],
    }
    return {"signal": {"chest": chest, "wrist": wrist}, "label": label}


def write_subject_pkl(root: Path, sid: str, seed: int = 0, **kwargs) -> Path:
    data = make_subject_arrays(seed=seed, **kwargs)
    data["subject"] = sid
    path = Path(root) / sid / f"{sid}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path
