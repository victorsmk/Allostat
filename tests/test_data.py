import numpy as np
import pytest

from stressnet.constants import CHEST_FS, WRIST_FS
from stressnet.data import quality, wesad
from stressnet.data.splits import label_distribution, loso_folds, subject_level_split, to_binary_windows
from synthetic import DEFAULT_PROTOCOL, make_subject_arrays


def _subject(seed=0):
    d = make_subject_arrays(seed)
    return {"subject": "SX",
            "chest": {k: v.astype(np.float64) for k, v in d["signal"]["chest"].items()},
            "wrist": {k: v.astype(np.float64) for k, v in d["signal"]["wrist"].items()},
            "label": d["label"]}


def test_find_label_runs_matches_protocol():
    runs = wesad.find_label_runs(_subject()["label"])
    assert [(lab, (e - s) // CHEST_FS) for s, e, lab in runs] == [(lab, sec) for lab, sec in DEFAULT_PROTOCOL]


def test_find_label_runs_filter_and_empty():
    label = np.array([0, 0, 1, 1, 1, 2, 0])
    assert wesad.find_label_runs(label, keep_labels=(1, 2)) == [(2, 5, 1), (5, 6, 2)]
    assert wesad.find_label_runs(np.array([])) == []


def test_load_subject_pkl_roundtrip(synthetic_root):
    d = wesad.load_subject_pkl(synthetic_root / "S2" / "S2.pkl")
    assert d["subject"] == "S2"
    assert d["chest"]["ACC"].shape[1] == 3
    assert d["chest"]["ECG"].shape[0] == d["label"].shape[0]


def test_window_count_and_no_label_crossing():
    s = _subject()
    stats = wesad.compute_baseline_stats(s, trim_start_sec=30)
    wins = wesad.generate_windows_for_subject(s, stats, window_sec=60, step_sec=15, keep_labels=(1, 2, 3, 4))
    expected = sum((sec - 60) // 15 + 1 for lab, sec in DEFAULT_PROTOCOL if lab in (1, 2, 3, 4) and sec >= 60)
    assert len(wins) == expected
    for w in wins:
        a, b = int(w["t_start_sec"] * CHEST_FS), int((w["t_start_sec"] + 60) * CHEST_FS)
        assert set(s["label"][a:b]) == {w["label"]}
        assert w["wrist"]["BVP"].shape == (60 * WRIST_FS["BVP"], 1)
        assert w["chest"]["ECG"].shape == (60 * CHEST_FS, 1)


def test_window_is_baseline_zscore_of_raw():
    s = _subject()
    stats = wesad.compute_baseline_stats(s, trim_start_sec=30)
    w = wesad.generate_windows_for_subject(s, stats, keep_labels=(2,))[0]
    mean, std = stats["wrist"]["EDA"]
    a = int(round(w["t_start_sec"] * WRIST_FS["EDA"]))
    raw = s["wrist"]["EDA"][a:a + 60 * WRIST_FS["EDA"]]
    np.testing.assert_allclose(w["wrist"]["EDA"], ((raw - mean) / std).astype(np.float32), rtol=1e-5)
    # stress EDA is elevated relative to this subject's own baseline
    assert w["wrist"]["EDA"].mean() > 3


def test_baseline_trim_too_long_raises():
    with pytest.raises(ValueError, match="leaves no baseline"):
        wesad.compute_baseline_stats(_subject(), trim_start_sec=400)


def test_missing_baseline_raises():
    s = _subject()
    s["label"] = np.zeros_like(s["label"])
    with pytest.raises(ValueError, match="no baseline"):
        wesad.compute_baseline_stats(s)


def test_check_missing_samples_reports_nans():
    s = _subject()
    s["wrist"]["EDA"][5] = np.nan
    assert wesad.check_missing_samples(s) == {"wrist.EDA": 1}


def test_ecg_highpass_removes_dc_offset():
    s = _subject()
    s["chest"]["ECG"] = s["chest"]["ECG"] + 5.0
    out = wesad.apply_ecg_highpass(s)
    assert abs(out["chest"]["ECG"].mean()) < 0.1
    assert abs(s["chest"]["ECG"].mean() - 5.0) < 0.1  # input not mutated


def test_build_flags_and_wrist_only_storage(built_windows):
    windows, report = built_windows
    assert set(report) == {"S2", "S3", "S4"}
    for w in windows:
        assert w["chest"] == {}  # keep_chest=False
        assert {"clip_flag", "motion_energy", "motion_raw", "n_beats", "bad_ibi_frac", "model_label"} <= set(w)
    assert {w["model_label"] for w in windows} == {0, 1, 2, 3}


def test_clean_pulse_has_good_ibi(built_windows):
    windows, _ = built_windows
    assert np.median([w["bad_ibi_frac"] for w in windows]) < 0.15


def test_has_clip_run():
    x = np.zeros(100)
    x[10:30] = 4.0
    assert quality.has_clip_run(x, glob_min=-1.0, glob_max=4.0, min_run=15)
    assert not quality.has_clip_run(x, glob_min=-1.0, glob_max=4.0, min_run=25)
    assert not quality.has_clip_run(x, glob_min=-1.0, glob_max=9.0, min_run=15)


def test_apply_gate_conditions():
    mk = lambda s, mot, ibi, clip: {"subject": s, "label": 1, "model_label": 1, "motion_raw": mot,
                                    "bad_ibi_frac": ibi, "clip_flag": clip}
    wins = [mk("A", i / 10, 0.0, False) for i in range(10)] + [mk("A", 0.0, 0.9, True)]
    kept, rep = quality.apply_gate(wins, clip=True)
    assert len(kept) == 10 and rep["clip"].sum() == 1
    kept, rep = quality.apply_gate(wins, clip=False, ibi=True)
    assert len(kept) == 10 and rep["ibi"].sum() == 1
    kept, rep = quality.apply_gate(wins, clip=False, motion=True, motion_pct=90)
    assert rep["motion"].sum() >= 1 and len(kept) < len(wins)


def test_subject_level_split_is_disjoint(built_windows):
    windows, _ = built_windows
    tr, va, val_subj = subject_level_split(windows, val_frac=0.34, seed=1)
    assert {w["subject"] for w in tr}.isdisjoint({w["subject"] for w in va})
    assert len(tr) + len(va) == len(windows) and len(val_subj) == 1
    with pytest.raises(ValueError):
        subject_level_split(windows, val_subjects=["S99"])


def test_loso_folds_cover_each_subject(built_windows):
    windows, _ = built_windows
    folds = list(loso_folds(windows))
    assert [h for *_, h in folds] == ["S2", "S3", "S4"]
    for tr, va, h in folds:
        assert {w["subject"] for w in va} == {h} and h not in {w["subject"] for w in tr}


def test_binary_relabel_keeps_originals(built_windows):
    windows, _ = built_windows
    binary = to_binary_windows(windows)
    assert set(label_distribution(binary)) == {0, 1}
    assert {w["model_label"] for w in windows} == {0, 1, 2, 3}  # input untouched


def test_cache_roundtrip_and_param_mismatch(synthetic_root, tmp_path):
    cache = tmp_path / "c.pkl"
    kw = dict(subject_ids=["S2"], baseline_trim_start_sec=30, keep_chest=False, verbose=False)
    w1, _ = wesad.load_or_build_final_dataset(cache, synthetic_root, **kw)
    w2, _ = wesad.load_or_build_final_dataset(cache, synthetic_root, **kw)
    assert len(w1) == len(w2) > 0
    with pytest.raises(ValueError, match="different parameters"):
        wesad.load_or_build_final_dataset(cache, synthetic_root, step_sec=30, **kw)
