"""Hand-crafted features for the classical baselines (window dict -> flat feature row)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import iqr

from .constants import CHEST_FS, WRIST_FS


def _flatten(x):
    """Drop a trailing size-1 channel dim ((N,1) -> (N,)); leave multi-channel arrays untouched."""
    x = np.asarray(x)
    return x.reshape(-1) if x.ndim > 1 and x.shape[1] == 1 else x


def basic_stats(x, prefix, fs=None):
    """Summary stats; pass ``fs`` to also get a linear trend (units/sec) over the window."""
    x = _flatten(x).astype(np.float64)
    feats = {
        f"{prefix}_mean": np.mean(x),
        f"{prefix}_std": np.std(x),
        f"{prefix}_min": np.min(x),
        f"{prefix}_max": np.max(x),
        f"{prefix}_range": np.ptp(x),
        f"{prefix}_iqr": iqr(x),
    }
    if fs is not None and len(x) > 1:
        t = np.arange(len(x)) / fs
        feats[f"{prefix}_slope"] = np.polyfit(t, x, 1)[0]
    return feats


def acc_features(acc, prefix, fs):
    """Vector-magnitude stats (orientation-invariant) plus per-axis std from a raw (N, 3) window."""
    acc = np.asarray(acc, dtype=np.float64)
    mag = np.linalg.norm(acc, axis=1)
    feats = basic_stats(mag, f"{prefix}_mag", fs=fs)
    feats[f"{prefix}_energy"] = float(np.mean(mag ** 2))
    for i, axis in enumerate(["x", "y", "z"]):
        feats[f"{prefix}_{axis}_std"] = float(np.std(acc[:, i]))
    return feats


def eda_features(eda, prefix, fs):
    """Level/trend stats plus a crude SCR peak count (local maxima with minimum prominence, <=1 per s)."""
    eda = _flatten(eda).astype(np.float64)
    feats = basic_stats(eda, prefix, fs=fs)
    if len(eda) > 3:
        peaks, props = find_peaks(eda, prominence=0.02, distance=max(1, int(fs)))
        feats[f"{prefix}_n_scr"] = len(peaks)
        feats[f"{prefix}_scr_mean_prom"] = float(np.mean(props["prominences"])) if len(peaks) else 0.0
    else:
        feats[f"{prefix}_n_scr"] = 0
        feats[f"{prefix}_scr_mean_prom"] = 0.0
    return feats


def hr_features_from_peaks(sig, prefix, fs, min_hr=40, max_hr=180):
    """Peak-based HRV from a pulsatile signal (ECG or BVP): mean/std HR, SDNN, RMSSD, pNN50.
    Returns NaNs (not zeros) when fewer than 3 peaks are found, so an imputer can fill them."""
    sig = _flatten(sig).astype(np.float64)
    min_dist = max(1, int(fs * 60 / max_hr))
    peaks, _ = find_peaks(sig, distance=min_dist, prominence=np.std(sig) * 0.5)
    if len(peaks) < 3:
        return {
            f"{prefix}_hr_mean": np.nan, f"{prefix}_hr_std": np.nan,
            f"{prefix}_sdnn": np.nan, f"{prefix}_rmssd": np.nan,
            f"{prefix}_pnn50": np.nan, f"{prefix}_n_peaks": float(len(peaks)),
        }
    rr = np.diff(peaks) / fs
    hr = 60.0 / rr
    valid = (hr >= min_hr) & (hr <= max_hr)
    rr_v, hr_v = (rr[valid], hr[valid]) if valid.sum() >= 2 else (rr, hr)
    diffs = np.diff(rr_v) * 1000
    return {
        f"{prefix}_hr_mean": float(np.mean(hr_v)),
        f"{prefix}_hr_std": float(np.std(hr_v)),
        f"{prefix}_sdnn": float(np.std(rr_v) * 1000),
        f"{prefix}_rmssd": float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) else np.nan,
        f"{prefix}_pnn50": float(np.mean(np.abs(diffs) > 50)) if len(diffs) else np.nan,
        f"{prefix}_n_peaks": float(len(peaks)),
    }


def resp_features(resp, prefix, fs):
    """Chest-only: level/trend stats plus breathing rate (breaths/min) from peak spacing."""
    resp = _flatten(resp).astype(np.float64)
    feats = basic_stats(resp, prefix, fs=fs)
    peaks, _ = find_peaks(resp, distance=int(fs * 1.5), prominence=np.std(resp) * 0.3)
    duration_min = len(resp) / fs / 60
    feats[f"{prefix}_breath_rate"] = len(peaks) / duration_min if duration_min > 0 else np.nan
    return feats


def emg_features(emg, prefix):
    """Chest-only: muscle activation level (mean abs, std, RMS)."""
    emg = _flatten(emg).astype(np.float64)
    return {
        f"{prefix}_mean_abs": float(np.mean(np.abs(emg))),
        f"{prefix}_std": float(np.std(emg)),
        f"{prefix}_rms": float(np.sqrt(np.mean(emg ** 2))),
    }


def extract_wrist_features(window):
    w = window["wrist"]
    feats = {}
    feats.update(acc_features(w["ACC"], "wrist_acc", WRIST_FS["ACC"]))
    feats.update(hr_features_from_peaks(w["BVP"], "wrist_bvp", WRIST_FS["BVP"]))
    feats.update(eda_features(w["EDA"], "wrist_eda", WRIST_FS["EDA"]))
    feats.update(basic_stats(w["TEMP"], "wrist_temp", fs=WRIST_FS["TEMP"]))
    return feats


def extract_chest_features(window):
    c = window["chest"]
    feats = {}
    feats.update(acc_features(c["ACC"], "chest_acc", CHEST_FS))
    feats.update(hr_features_from_peaks(c["ECG"], "chest_ecg", CHEST_FS))
    feats.update(eda_features(c["EDA"], "chest_eda", CHEST_FS))
    feats.update(basic_stats(c["Temp"], "chest_temp", fs=CHEST_FS))
    feats.update(resp_features(c["Resp"], "chest_resp", CHEST_FS))
    feats.update(emg_features(c["EMG"], "chest_emg"))
    return feats


def extract_features(window, mode="full"):
    """mode='wrist' -> wrist channels only (deployable); mode='full' -> wrist + chest (oracle upper bound)."""
    feats = extract_wrist_features(window)
    if mode == "full":
        feats.update(extract_chest_features(window))
    feats["subject"] = window["subject"]
    feats["label"] = window["model_label"]
    return feats


def build_feature_df(windows, mode="full"):
    return pd.DataFrame([extract_features(w, mode=mode) for w in windows])


def feature_columns(df):
    return [c for c in df.columns if c not in ("subject", "label")]
