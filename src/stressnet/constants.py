"""Dataset constants and the modality registry shared by data code and models."""

from __future__ import annotations

CHEST_FS = 700
WRIST_FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}

CHEST_CHANNELS = ["ACC", "ECG", "EDA", "EMG", "Resp", "Temp"]
WRIST_CHANNELS = ["ACC", "BVP", "EDA", "TEMP"]

# WESAD raw label ids: 0 transient, 1 baseline, 2 stress, 3 amusement, 4 meditation, 5/6/7 ignore.
# Raw label -> model class index (4-class setup). Binary stress/non-stress is derived with
# data.splits.to_binary_windows.
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}
BASELINE_LABEL_RAW = 1
CLASS_NAMES = ["baseline", "stress", "amusement", "meditation"]
BIN_CLASS_NAMES = ["non_stress", "stress"]

# S1 and S12 were discarded by the dataset authors (sensor malfunction).
ALL_SUBJECTS = [f"S{i}" for i in list(range(2, 12)) + list(range(13, 18))]

WINDOW_SEC = 60
STEP_SEC = 15

# name -> (location, channel key in window["chest"/"wrist"], sample rate Hz, n signal channels)
MODALITY_SPECS = {
    "chest_acc": ("chest", "ACC", CHEST_FS, 3),
    "chest_ecg": ("chest", "ECG", CHEST_FS, 1),
    "chest_eda": ("chest", "EDA", CHEST_FS, 1),
    "chest_emg": ("chest", "EMG", CHEST_FS, 1),
    "chest_resp": ("chest", "Resp", CHEST_FS, 1),
    "chest_temp": ("chest", "Temp", CHEST_FS, 1),
    "wrist_acc": ("wrist", "ACC", WRIST_FS["ACC"], 3),
    "wrist_bvp": ("wrist", "BVP", WRIST_FS["BVP"], 1),
    "wrist_eda": ("wrist", "EDA", WRIST_FS["EDA"], 1),
    "wrist_temp": ("wrist", "TEMP", WRIST_FS["TEMP"], 1),
}

FULL_MODALITIES = list(MODALITY_SPECS)
WRIST_MODALITIES = [m for m in MODALITY_SPECS if m.startswith("wrist_")]


def modality_input_len(name: str, window_sec: int = WINDOW_SEC, specs: dict | None = None) -> int:
    """Number of samples of one window of the given modality."""
    specs = specs or MODALITY_SPECS
    _, _, fs, _ = specs[name]
    return int(window_sec * fs)
