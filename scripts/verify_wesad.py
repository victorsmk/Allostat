"""Regression check: rebuild the WESAD windows from the raw .pkl files and compare with what the original
Colab notebook produced (per-subject window counts, 2737 total, 2733 after dropping clip-flagged windows).

    python scripts/verify_wesad.py --root data/WESAD --subjects S2 S3        # quick
    python scripts/verify_wesad.py --root data/WESAD --cache cache/wesad_wrist.pkl   # all 15, cached

Stores wrist signals only (keep_chest=False): ~110 MB for all subjects instead of ~3.7 GB.
"""

from __future__ import annotations

import argparse
import sys

from stressnet.constants import ALL_SUBJECTS, LABEL_MAP
from stressnet.data import wesad

# Window counts printed by the original notebook (cell 33), 60 s windows / 15 s step.
EXPECTED_WINDOWS = {"S2": 177, "S3": 179, "S4": 181, "S5": 183, "S6": 182, "S7": 182, "S8": 182, "S9": 182,
                    "S10": 187, "S11": 184, "S13": 184, "S14": 184, "S15": 184, "S16": 183, "S17": 183}
EXPECTED_TOTAL, EXPECTED_AFTER_CLIP = 2737, 2733
EXPECTED_CLIP_FLAGGED = {"S3": 4}  # every other subject: 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/WESAD")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--cache", default=None, help="pickle path; reuse if it exists")
    args = ap.parse_args()
    subjects = args.subjects or ALL_SUBJECTS

    if args.cache:
        windows, report = wesad.load_or_build_final_dataset(
            args.cache, args.root, LABEL_MAP, subject_ids=subjects, keep_chest=False)
    else:
        windows, report = wesad.build_dataset_with_quality(args.root, subject_ids=subjects, keep_chest=False)
        windows = wesad.map_labels(windows, LABEL_MAP)

    ok = True
    for sid in subjects:
        got, want = report[sid]["n_windows"], EXPECTED_WINDOWS[sid]
        clip, want_clip = report[sid]["n_clip_flagged"], EXPECTED_CLIP_FLAGGED.get(sid, 0)
        status = "OK " if (got, clip) == (want, want_clip) else "DIFF"
        ok &= status == "OK "
        print(f"{status} {sid}: windows {got} (notebook {want}) | clip-flagged {clip} (notebook {want_clip})")

    if set(subjects) == set(ALL_SUBJECTS):
        kept = [w for w in windows if not w["clip_flag"]]
        print(f"total {len(windows)} (notebook {EXPECTED_TOTAL}) -> after clip gate {len(kept)} "
              f"(notebook {EXPECTED_AFTER_CLIP})")
        ok &= (len(windows), len(kept)) == (EXPECTED_TOTAL, EXPECTED_AFTER_CLIP)
    print("MATCHES NOTEBOOK" if ok else "MISMATCH vs notebook")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
