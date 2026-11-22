import numpy as np
import pytest

from stressnet.constants import WRIST_FS
from stressnet.data import e4


def test_to_float_repairs_stray_dot_and_flags_garbage():
    assert e4._to_float("0.5") == 0.5
    assert e4._to_float("1.038.145") == pytest.approx(1.038145)
    assert e4._to_float("10.123.456") == pytest.approx(10.123456)
    assert np.isnan(e4._to_float("abc"))
    assert np.isnan(e4._to_float("1.2.3.4"))


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n")


def test_read_e4_signal_header_variants(tmp_path):
    _write(tmp_path / "EDA.csv", ["1600000000.0", "4.0", "0.1", "0.2"])           # unix header
    arr, start = e4.read_e4_signal(tmp_path / "EDA.csv", "EDA")
    assert arr.tolist() == [0.1, 0.2] and start == 1600000000.0
    _write(tmp_path / "EDA2.csv", ["2013-02-20 17:55:19", "4.0", "0.1", "0.2"])    # UTC-string header
    arr, start = e4.read_e4_signal(tmp_path / "EDA2.csv", "EDA")
    assert arr.tolist() == [0.1, 0.2] and start.year == 2013
    _write(tmp_path / "ACC.csv", ["32,32,32", "1,2,3", "4,5,6"])                    # lone rate row
    arr, _ = e4.read_e4_signal(tmp_path / "ACC.csv", "ACC")
    assert arr.shape == (2, 3)
    _write(tmp_path / "TEMP.csv", ["33.1", "33.2"])                                  # no header
    arr, start = e4.read_e4_signal(tmp_path / "TEMP.csv", "TEMP")
    assert arr.tolist() == [33.1, 33.2] and start is None


def test_exercise_v1_and_v2_segments():
    v1 = e4.exercise_stress_segments(list(np.arange(13) * 100.0))   # tag k at (k-1)*100 s
    by = {s["name"]: s for s in v1}
    assert (by["Baseline"]["start"], by["Baseline"]["end"]) == (0.0, 100.0)
    assert (by["Stroop"]["start"], by["Stroop"]["end"], by["Stroop"]["label"]) == (200.0, 300.0, 1)
    assert (by["TMCT"]["start"], by["TMCT"]["end"], by["TMCT"]["label"]) == (400.0, 500.0, 1)
    assert by["First Rest"]["label"] == 0 and by["Second Rest"]["start"] == 500.0

    v2 = e4.exercise_stress_segments([500.0 + 100.0 * i for i in range(9)])
    by = {s["name"]: s for s in v2}
    assert (by["Baseline"]["start"], by["Baseline"]["end"], by["Baseline"]["confidence"]) == (320.0, 500.0, "inferred")
    assert (by["TMCT"]["start"], by["TMCT"]["end"], by["TMCT"]["label"]) == (600.0, 700.0, 1)
    assert by["Second Rest"]["start"] == 1100.0 and by["Second Rest"]["end"] == 1200.0
    assert e4.exercise_stress_segments([0.0] * 5) is None


def test_e4stress_segments_cover_schedule():
    total = 2000.0
    segs = e4.e4stress_segments(total)
    assert segs[0]["name"] == "Rest 1" and segs[-1]["end"] == total
    assert [s["name"] for s in segs if s["name"].startswith("Task")] == [f"Task {i}" for i in range(1, 6)]
    assert e4.e4stress_segments(1900.0) is None          # shorter than the 1920 s minimum protocol


def _wrist(seconds, rng):
    return {ch: rng.normal(5.0, 2.0, size=(int(seconds * fs), 3) if ch == "ACC" else int(seconds * fs))
            for ch, fs in WRIST_FS.items()}


def test_segment_windows_inside_segments_and_baseline_normalised():
    rng = np.random.default_rng(0)
    wrist = _wrist(300, rng)
    stats = e4.baseline_stats(wrist, 0.0, 120.0, trim_start_sec=0.0)
    segs = [{"name": "rest", "start": 0.0, "end": 120.0, "label": 0},
            {"name": "task", "start": 120.0, "end": 200.0, "label": 1},
            {"name": "short", "start": 200.0, "end": 230.0, "label": 1}]    # 30 s: no window fits
    w = e4.segment_windows("X", wrist, segs, stats, window_sec=60, step_sec=15, meta={"dataset": "d"})
    assert {x["stage"] for x in w} == {"rest", "task"}
    assert [x["t_start_sec"] for x in w if x["stage"] == "task"] == [120.0, 135.0]
    assert all(x["t_start_sec"] + 60 <= 200.0 for x in w if x["stage"] == "task")
    assert all(x["dataset"] == "d" and x["model_label"] == x["label"] for x in w)
    assert w[0]["wrist"]["BVP"].shape == (60 * 64, 1) and w[0]["wrist"]["ACC"].shape == (60 * 32, 3)
    rest_mean = np.mean([x["wrist"]["EDA"].mean() for x in w if x["stage"] == "rest"])
    assert abs(rest_mean) < 0.2                      # z-scored with the rest segment itself


def test_segment_windows_drop_nan_windows_and_respect_lag():
    rng = np.random.default_rng(1)
    wrist = _wrist(200, rng)
    wrist["EDA"][100 * 4:101 * 4] = np.nan           # NaN at t=100 s
    stats = e4.baseline_stats(_wrist(200, rng), 0.0, 100.0, trim_start_sec=0.0)
    segs = [{"name": "rest", "start": 0.0, "end": 200.0, "label": 0, "lag_sec": 30.0}]
    w = e4.segment_windows("X", wrist, segs, stats, window_sec=60, step_sec=15)
    starts = [x["t_start_sec"] for x in w]
    assert min(starts) == 30.0                       # lag respected
    assert all(not (s <= 100 < s + 60) for s in starts)   # windows covering the NaN are gone
