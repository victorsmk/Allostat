import numpy as np

from stressnet.data import pool


def _w(dataset, label, model_label=None, domain="stress", subject="X"):
    w = {"dataset": dataset, "label": label, "domain": domain, "subject": subject}
    if model_label is not None:
        w["model_label"] = model_label
    return w


def test_three_way_labels():
    assert pool.three_way_label(_w("wesad", 1)) == pool.CALM
    assert pool.three_way_label(_w("wesad", 2)) == pool.STRESS
    assert pool.three_way_label(_w("wesad", 3)) == pool.CALM      # amusement
    assert pool.three_way_label(_w("wesad", 4)) == pool.CALM      # meditation
    assert pool.three_way_label(_w("e4stress", 1, 1)) == pool.STRESS
    assert pool.three_way_label(_w("e4stress", 0, 0)) == pool.CALM
    assert pool.three_way_label(_w("exercise_aerobic", 0, 0, domain="exercise")) == pool.EXERTION
    assert pool.three_way_label(_w("exercise_stress", 1, 1)) == pool.STRESS


def test_pool_windows_copies_and_tags():
    wes = [{"subject": "S2", "label": 2, "wrist": {}}, {"subject": "S2", "label": 3, "wrist": {}}]
    oth = [_w("exercise_anaerobic", 0, 0, domain="exercise", subject="EXR_S01")]
    p = pool.pool_windows(wes, oth)
    assert [w["model_label"] for w in p] == [1, 0, 2]
    assert [pool.dataset_family(w) for w in p] == ["wesad", "wesad", "exercise"]
    assert "dataset" not in wes[0]          # inputs are not mutated
