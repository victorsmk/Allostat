"""Zero-shot cross-dataset test: train on WESAD only, evaluate on E4 Stress and EXERCISE (no fine-tuning).

For the exercise sessions every window is a *negative* (physical exertion is not psychological stress), so the
interesting number there is the false-positive rate (share of windows predicted "stress").

  python scripts/cross_dataset_eval.py --seeds 0 1 2
Needs cache/wesad_wrist.pkl (see notebooks) and cache/other_datasets.pkl (see stressnet.data.e4).
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stressnet.constants import LABEL_MAP  # noqa: E402
from stressnet.data import wesad  # noqa: E402
from stressnet.data.quality import apply_gate  # noqa: E402
from stressnet.data.splits import to_binary_windows  # noqa: E402
from stressnet.data.torch_dataset import WindowDataset  # noqa: E402
from stressnet.training import get_device, move_batch, train_final_model  # noqa: E402

CONFIGS = {"all": ["wrist_bvp", "wrist_eda", "wrist_acc", "wrist_temp"],
           "bvp_eda": ["wrist_bvp", "wrist_eda"]}


@torch.no_grad()
def predict_stress_prob(model, windows, device, batch_size=128):
    model.eval()
    probs = []
    for batch in DataLoader(WindowDataset(windows), batch_size=batch_size, shuffle=False):
        inputs, _ = move_batch(batch, device)
        probs.append(torch.softmax(model(inputs), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def score_labelled(y, p):
    pred = (p >= 0.5).astype(int)
    neg = y == 0
    return {"n": len(y), "n_stress": int(y.sum()), "macro_f1": f1_score(y, pred, average="macro"),
            "bal_acc": balanced_accuracy_score(y, pred), "auroc": roc_auc_score(y, p),
            "stress_recall": float(pred[y == 1].mean()), "fpr": float(pred[neg].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "results/cross_dataset"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wins, _ = wesad.load_or_build_final_dataset(ROOT / "cache/wesad_wrist.pkl", ROOT / "data/WESAD",
                                                LABEL_MAP, keep_chest=False)
    wesad_train = to_binary_windows(apply_gate(wins, clip=True)[0])
    with open(ROOT / "cache/other_datasets.pkl", "rb") as f:
        other = pickle.load(f)
    e4s, exr = other["e4stress"], other["exercise"]
    groups = {
        "e4stress": [w for w in e4s],
        "exercise_stress": [w for w in exr if w["dataset"] == "exercise_stress"],
        "exercise_aerobic": [w for w in exr if w["dataset"] == "exercise_aerobic"],
        "exercise_anaerobic": [w for w in exr if w["dataset"] == "exercise_anaerobic"],
    }
    print({k: len(v) for k, v in groups.items()}, "| WESAD train windows:", len(wesad_train))
    device = get_device()

    rows, subj_rows = [], []
    for cfg, mods in CONFIGS.items():
        for seed in args.seeds:
            print(f"=== train on WESAD: {cfg}, seed {seed}")
            model, _ = train_final_model(wesad_train, mods, "cnn", 2, args.epochs, 32, 1e-3, 1e-2, 0.3,
                                         seed=seed, device=device, verbose=False)
            for name, ws in groups.items():
                p = predict_stress_prob(model, ws, device)
                y = np.array([w["model_label"] for w in ws])
                pred = p >= 0.5
                if name in ("exercise_aerobic", "exercise_anaerobic"):
                    rows.append({"config": cfg, "seed": seed, "group": name, "n": len(y),
                                 "fpr": float(pred.mean()), "mean_p_stress": float(p.mean())})
                else:
                    rows.append({"config": cfg, "seed": seed, "group": name, **score_labelled(y, p)})
                subs = np.array([w["subject"] for w in ws])
                for s in sorted(set(subs)):
                    m = subs == s
                    r = {"config": cfg, "seed": seed, "group": name, "subject": s, "n": int(m.sum()),
                         "pos_rate": float(pred[m].mean())}
                    if y[m].min() != y[m].max():
                        r["macro_f1"] = f1_score(y[m], pred[m].astype(int), average="macro")
                    subj_rows.append(r)
            pd.DataFrame(rows).to_csv(out / "summary_per_run.csv", index=False)
            pd.DataFrame(subj_rows).to_csv(out / "per_subject.csv", index=False)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== mean over seeds ===")
    print(df.groupby(["group", "config"]).mean(numeric_only=True).drop(columns=["seed"]).round(3))
    print("\n=== std over seeds ===")
    print(df.groupby(["group", "config"]).std(numeric_only=True).drop(columns=["seed", "n"], errors="ignore").round(3))


if __name__ == "__main__":
    main()
