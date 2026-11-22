import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # lets tests import the `synthetic` helper

from stressnet.constants import LABEL_MAP  # noqa: E402
from stressnet.data import wesad  # noqa: E402
from synthetic import write_subject_pkl  # noqa: E402

SYN_SUBJECTS = ["S2", "S3", "S4"]


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("wesad")
    for i, sid in enumerate(SYN_SUBJECTS):
        write_subject_pkl(root, sid, seed=i)
    return root


@pytest.fixture(scope="session")
def built_windows(synthetic_root):
    """All label-mapped windows (wrist-only storage) for three synthetic subjects."""
    windows, report = wesad.build_dataset_with_quality(
        synthetic_root, subject_ids=SYN_SUBJECTS, baseline_trim_start_sec=30, keep_chest=False, verbose=False)
    return wesad.map_labels(windows, LABEL_MAP), report
