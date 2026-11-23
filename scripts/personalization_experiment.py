"""Personalization experiment: how much does a few minutes of a new user's own labelled data help?

Protocol (every split by whole person, same 5 subject-grouped folds as scripts/three_way_experiment.py):
  1. train a global 3-class model (calm / stress / exertion) on the training folds' people
  2. for every held-out person, draw K labelled calibration windows per class (K = 1, 3, 5, 10 one-minute
     windows; no calibration window overlaps any test window), adapt with
       bias  - per-class logit offset (personal threshold)
       head  - fine-tuned classifier head, L2 penalty towards the global weights
     and test on the person's remaining windows; the global model is scored on the SAME test windows
  3. repeat the draw ``--draws`` times; report balanced accuracy (3-way and stress-vs-rest) per dataset,
     paired against the global model.
Hyper-parameters of the adaptation are fixed in advance (stressnet/personalization.py), not tuned on test data.

  python scripts/personalization_experiment.py --configs bvp_eda_acc --seeds 0 1
"""
import argparse
import itertools
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import three_way_experiment as twe  # noqa: E402
from stressnet import personalization as P  # noqa: E402
from stressnet.data.pool import dataset_family  # noqa: E402
from stressnet.models import MultiModalNet  # noqa: E402
from stressnet.training import get_device, train_final_model  # noqa: E402

BUDGETS = [1, 3, 5, 10]          # labelled one-minute windows per class
METHODS = ["global", "bias", "head"]


def get_model(cfg, fold, seed, train, args, device, model_dir):
    mods = twe.CONFIGS[cfg]
    path = model_dir / f"{cfg}_s{seed}_fold{fold}.pt"
    if path.exists():
        model = MultiModalNet(mods, body="cnn", num_classes=3, dropout=0.3).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        return model.eval()
    model, _ = train_final_model(twe.thin(train, args.train_stride), mods, "cnn", 3, args.epochs, 32, 1e-3, 1e-2,
                                 0.3, seed=seed * 10 + fold, device=device, verbose=False)
    torch.save(model.state_dict(), path)
    return model.eval()


def scores(y, pred):
    out = {"bal_acc3": balanced_accuracy_score(y, pred)}
    if (y == 1).any() and (y != 1).any():
        out["bal_acc_stress"] = balanced_accuracy_score(y == 1, pred == 1)
    return out


def personalize_person(model, emb, windows, args, rng):
    """Rows for one person. ``emb`` are the (frozen-encoder) embeddings of ``windows``."""
    y_all = np.array([w["model_label"] for w in windows])
    with torch.no_grad():
        logits_all = model.head(emb)
    rows = []
    for gap, k, draw in itertools.product(args.gaps, BUDGETS, range(args.draws)):
        cal, test = P.sample_calibration(windows, k, rng, min_gap_sec=gap)
        y_test = y_all[test]
        if len(test) < 20 or len(set(y_test.tolist())) < 2:
            continue
        y_cal = y_all[cal]
        lg_test = logits_all[test]
        preds = {"global": lg_test.argmax(1).numpy()}
        bias = P.fit_bias(logits_all[cal], y_cal)
        preds["bias"] = (lg_test + bias).argmax(1).numpy()
        head = P.fit_head(model.head, emb[cal], y_cal)
        with torch.no_grad():
            preds["head"] = head(emb[test]).argmax(1).numpy()
        for m, pr in preds.items():
            rows.append({"gap": gap, "k": k, "draw": draw, "method": m, "n_cal": len(cal),
                         "n_test": len(test), **scores(y_test, pr)})
    return rows


def run(args):
    pool = twe.load_pool()
    y = np.array([w["model_label"] for w in pool])
    person = np.array([w["dataset"].split("_")[0] + ":" + w["subject"] for w in pool])
    folds = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0).split(pool, y, person))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    t0 = time.time()
    for cfg in args.configs:
        for seed in args.seeds:
            for k, (tr, te) in enumerate(folds[: args.max_folds]):
                f = out / f"{cfg}_s{seed}_fold{k}.csv"
                if f.exists():
                    continue
                print(f"=== {cfg} seed {seed} fold {k + 1}/{len(folds)} | elapsed {(time.time() - t0) / 60:.1f} min",
                      flush=True)
                model = get_model(cfg, k, seed, [pool[i] for i in tr], args, device, model_dir)
                held = [pool[i] for i in te]
                emb = P.embed(model, held, device)
                held_person = person[te]
                rows = []
                for pid in sorted(set(held_person)):
                    idx = np.where(held_person == pid)[0]
                    rng = np.random.default_rng(zlib.crc32(f"{seed}|{k}|{pid}".encode()))
                    rs = personalize_person(model, emb[idx], [held[i] for i in idx], args, rng)
                    fam = dataset_family(held[idx[0]])
                    rows += [{"config": cfg, "seed": seed, "fold": k, "person": pid, "family": fam, **r}
                             for r in rs]
                pd.DataFrame(rows).to_csv(f, index=False)
    print("finished", flush=True)


def summarize(args):
    out = Path(args.out)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for cfg in args.configs:
        files = sorted(out.glob(f"{cfg}_s*_fold*.csv"))
        if not files:
            continue
        d = pd.concat(map(pd.read_csv, files))
        for gap in sorted(d.gap.unique()):
            dg = d[d.gap == gap]
            # average over draws and seeds first: one number per person x budget x method
            per = dg.groupby(["family", "person", "k", "method"])[["bal_acc3", "bal_acc_stress"]].mean().reset_index()
            print()
            print(f"################ {cfg} | calibration-test gap {gap:.0f} s | {per.person.nunique()} people, "
                  f"{dg.seed.nunique()} global-model seed(s), {dg.draw.nunique()} draws ################")
            for metric in ("bal_acc3", "bal_acc_stress"):
                print()
                print(f"--- mean {metric} over people (cols: labelled one-minute windows per class) ---")
                by_fam = per.groupby(["family", "method", "k"])[metric].mean().unstack("k")
                overall = per.groupby(["method", "k"])[metric].mean().unstack("k")
                overall.index = pd.MultiIndex.from_product([["ALL"], overall.index])
                print(pd.concat([overall, by_fam]).round(3))
            print()
            print("--- paired change vs the global model on the same test windows (bal_acc3) ---")
            rows = []
            for fam in ["ALL"] + sorted(per.family.unique()):
                sub = per if fam == "ALL" else per[per.family == fam]
                for m in ("bias", "head"):
                    for k in BUDGETS:
                        a_ = sub[(sub.method == "global") & (sub.k == k)].set_index("person").bal_acc3
                        b_ = sub[(sub.method == m) & (sub.k == k)].set_index("person").bal_acc3
                        diff = (b_ - a_).dropna()
                        if len(diff) < 5:
                            continue
                        try:
                            p = wilcoxon(diff).pvalue
                        except ValueError:
                            p = np.nan
                        rows.append({"family": fam, "method": m, "k": k, "n": len(diff),
                                     "mean_change": diff.mean(),
                                     "people_improved": f"{int((diff > 1e-9).sum())}/{len(diff)}",
                                     "wilcoxon_p": p})
            print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["bvp_eda_acc"], choices=list(twe.CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-folds", type=int, default=5, help="use fewer folds (smoke tests)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--gaps", type=float, nargs="+", default=[60.0, 300.0],
                    help="minimum seconds between a calibration window and any test window of the same recording")
    ap.add_argument("--out", default=str(ROOT / "results/personalization"))
    ap.add_argument("--model-dir", default=str(ROOT / "cache/personalization_models"))
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if not a.summarize:
        run(a)
    summarize(a)
