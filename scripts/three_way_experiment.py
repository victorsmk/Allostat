"""3-way (calm / stress / exertion) experiment on pooled WESAD + E4 Stress + EXERCISE wrist windows.

Two protocols, both with 1D-CNNs trained from scratch and every split made by whole subject:
  cv      subject-grouped K-fold over the pooled windows (out-of-fold predictions for every window)
  lodo    leave-one-dataset-out: train on the pool minus WESAD (then minus E4 Stress), test on the held-out set
Out-of-fold predictions are written per fold so an interrupted run resumes; ``--summarize`` only reports.

  python scripts/three_way_experiment.py --configs all bvp_eda_acc bvp_eda --epochs 15
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stressnet.constants import LABEL_MAP  # noqa: E402
from stressnet.data import wesad  # noqa: E402
from stressnet.data.pool import CLASS_NAMES_3, dataset_family, pool_windows  # noqa: E402
from stressnet.data.quality import apply_gate  # noqa: E402
from stressnet.data.torch_dataset import WindowDataset  # noqa: E402
from stressnet.training import get_device, move_batch, train_final_model  # noqa: E402

CONFIGS = {"all": ["wrist_bvp", "wrist_eda", "wrist_acc", "wrist_temp"],
           "bvp_eda_acc": ["wrist_bvp", "wrist_eda", "wrist_acc"],
           "bvp_eda": ["wrist_bvp", "wrist_eda"]}
PROBA_COLS = ["p_calm", "p_stress", "p_exertion"]


def load_pool():
    wins, _ = wesad.load_or_build_final_dataset(ROOT / "cache/wesad_wrist.pkl", ROOT / "data/WESAD",
                                                LABEL_MAP, keep_chest=False)
    wesad_ok = apply_gate(wins, clip=True)[0]
    with open(ROOT / "cache/other_datasets.pkl", "rb") as f:
        other = pickle.load(f)
    return pool_windows(wesad_ok, other["e4stress"] + other["exercise"])


@torch.no_grad()
def predict_proba(model, windows, device, batch_size=128):
    model.eval()
    out = []
    for batch in DataLoader(WindowDataset(windows), batch_size=batch_size, shuffle=False):
        inputs, _ = move_batch(batch, device)
        out.append(torch.softmax(model(inputs), dim=1).cpu().numpy())
    return np.concatenate(out)


def thin(windows, stride):
    """Keep every ``stride``-th window of each subject (training only; the 15 s step makes neighbours overlap)."""
    if stride <= 1:
        return windows
    seen, out = {}, []
    for w in windows:
        k = (w["subject"], w["dataset"])
        seen[k] = seen.get(k, -1) + 1
        if seen[k] % stride == 0:
            out.append(w)
    return out


def fit_predict(train, test, mods, args, device, seed):
    model, _ = train_final_model(thin(train, args.train_stride), mods, "cnn", 3, args.epochs, 32, 1e-3, 1e-2, 0.3,
                                 seed=seed, device=device, verbose=False)
    return predict_proba(model, test, device)


def frame(test, proba, tag, fold):
    return pd.DataFrame({
        "protocol": tag, "fold": fold, "subject": [w["subject"] for w in test],
        "dataset": [w["dataset"] for w in test], "family": [dataset_family(w) for w in test],
        "stage": [w["stage"] for w in test], "t": [w["t_start_sec"] for w in test],
        "y": [w["model_label"] for w in test],
        "p_calm": proba[:, 0], "p_stress": proba[:, 1], "p_exertion": proba[:, 2]})


def run(args):
    pool = load_pool()
    y = np.array([w["model_label"] for w in pool])
    subj = np.array([w["dataset"].split("_")[0] + ":" + w["subject"] for w in pool])
    print("pool:", len(pool), "windows |", {CLASS_NAMES_3[k]: int((y == k).sum()) for k in range(3)})
    device = get_device()
    folds = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0).split(pool, y, subj))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for cfg in args.configs:
        mods = CONFIGS[cfg]
        for k, (tr, te) in enumerate(folds):
            f = out / f"{cfg}_cv_fold{k}.csv"
            if f.exists():
                continue
            print(f"=== {cfg} cv fold {k + 1}/{len(folds)} | elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
            test = [pool[i] for i in te]
            p = fit_predict([pool[i] for i in tr], test, mods, args, device, seed=k)
            frame(test, p, "cv", k).to_csv(f, index=False)
        for held in ("wesad", "e4stress"):
            f = out / f"{cfg}_lodo_{held}.csv"
            if f.exists():
                continue
            print(f"=== {cfg} leave-out {held} | elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
            train = [w for w in pool if dataset_family(w) != held]
            test = [w for w in pool if dataset_family(w) == held]
            p = fit_predict(train, test, mods, args, device, seed=100)
            frame(test, p, f"lodo_{held}", -1).to_csv(f, index=False)
    print("training finished", flush=True)


def metrics_3way(d):
    y, pred = d["y"].values, d[PROBA_COLS].values.argmax(1)
    rec = {f"recall_{CLASS_NAMES_3[k]}": float((pred[y == k] == k).mean()) for k in range(3) if (y == k).any()}
    return {"n": len(d), "macro_f1_3way": f1_score(y, pred, average="macro"),
            "bal_acc": balanced_accuracy_score(y, pred), **rec}


def summarize(args):
    out = Path(args.out)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for cfg in args.configs:
        files = sorted(out.glob(f"{cfg}_cv_fold*.csv"))
        if not files:
            continue
        cv = pd.concat(map(pd.read_csv, files))
        pred = cv[PROBA_COLS].values.argmax(1)
        print(f"\n################ {cfg}: subject-grouped CV ({len(files)} folds) ################")
        rows = {"ALL": metrics_3way(cv)}
        for fam, d in cv.groupby("family"):
            rows[fam] = metrics_3way(d)
        print(pd.DataFrame(rows).T.round(3))
        print("confusion (rows=true, cols=pred; calm/stress/exertion):\n",
              confusion_matrix(cv["y"], pred, labels=[0, 1, 2]))
        ys, ps = (cv["y"] == 1).astype(int).values, (pred == 1).astype(int)
        print(f"binary stress-vs-rest: macro-F1 {f1_score(ys, ps, average='macro'):.3f} | "
              f"exercise windows called stress: {float(ps[cv['y'].values == 2].mean()):.3f}")
        per = cv.assign(ok=(pred == cv["y"].values)).groupby(["family", "subject"]).ok.mean()
        print("per-subject 3-way accuracy by dataset:\n", per.groupby("family").describe().round(2))
        for held in ("wesad", "e4stress"):
            f = out / f"{cfg}_lodo_{held}.csv"
            if f.exists():
                d = pd.read_csv(f)
                pr = d[PROBA_COLS].values.argmax(1)
                yb, pb = (d["y"] == 1).astype(int).values, (pr == 1).astype(int)
                print(f"[leave-out {held}] binary stress macro-F1 {f1_score(yb, pb, average='macro'):.3f} "
                      f"bal_acc {balanced_accuracy_score(yb, pb):.3f} | stress recall {pb[yb == 1].mean():.3f} "
                      f"| false-alarm rate {pb[yb == 0].mean():.3f} | predicted exertion {float((pr == 2).mean()):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["all", "bvp_eda_acc", "bvp_eda"], choices=list(CONFIGS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results/three_way"))
    ap.add_argument("--summarize", action="store_true", help="only print the summary of finished runs")
    a = ap.parse_args()
    if not a.summarize:
        run(a)
    summarize(a)
