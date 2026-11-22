# Wearable stress detection (licenta)

The reusable code behind the original Colab notebooks (`notebooks/`, left untouched) lives in the
`stressnet` package. Notebooks become thin: they import from the package instead of redefining everything.

```
src/stressnet/
  constants.py        sampling rates, channels, label map, MODALITY_SPECS registry
  data/wesad.py       .pkl loading, baseline normalisation, windowing, build + cache
  data/quality.py     clip / motion / IBI quality flags and gating
  data/splits.py      subject-level split, LOSO folds, binary relabelling
  data/torch_dataset.py
  features.py         hand-crafted features for classical models
  classical.py        LogReg / RF / HistGB, LOSO, gate ablation
  models.py           per-modality encoders (CNN/TCN), MultiModalNet with missing-modality tokens
  training.py         run_epoch, train_deep_model, loso_deep_cv (per-fold caching), train_final_model
  evaluation.py       fold summaries, paired Wilcoxon comparison
  checkpoint.py       save/load deployable checkpoints
tests/                pytest, runs on synthetic WESAD-shaped data (no dataset needed)
scripts/verify_wesad.py   rebuilds windows from raw data and compares with the notebook's numbers
```

## Local setup (Windows, CPU)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install numpy scipy pandas scikit-learn matplotlib pytest
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pytest
```

## Colab

Colab already has torch. Upload/clone this folder, then `pip install -e .` and import `stressnet`.
See `notebooks/colab_run_loso.ipynb`.

## Behaviour notes

* Model/parameter names and parameter counts match the notebook, so the saved checkpoints
  (`wrist_cnn_binary_full15.pt`, ...) load with `stressnet.checkpoint.load_checkpoint`.
* `keep_chest=False` stores wrist signals only (about 110 MB for all of WESAD instead of about 3.7 GB), while the
  chest-based clip flags are still computed. Use it for every wrist-only experiment.
* Final deployment models are trained on all subjects; their training metrics are not a generalisation estimate.
