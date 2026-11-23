"""Cross-session personalization: calibrate on one recording session of a person, test on another.

EXERCISE people have three separate recordings: a stress protocol (calm + stress windows, session S), an
aerobic session (A) and an anaerobic session (N), the last two being pure exertion. So the test windows here
never come from a session used for calibration. Three scenarios:

  S_to_exercise   calibrate on S only (calm/stress labels, as an app's feedback prompts would give), test on A+N.
                  Question: does adapting to calm/stress labels break exertion detection / cause false alarms?
  SA_to_N         calibrate on S + exertion windows of A, test on N.
  SN_to_A         calibrate on S + exertion windows of N, test on A.

All test windows of the scenarios are exertion (class 2), so the metrics are the share of test windows predicted
exertion (recall) and predicted stress (the false-alarm rate). Uses the global models cached by
scripts/personalization_experiment.py (same folds, same people held out).

  python scripts/cross_session_experiment.py --configs bvp_eda_acc --seeds 0 1
"""
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import personalization_experiment as pe  # noqa: E402
import three_way_experiment as twe  # noqa: E402
from stressnet import personalization as P  # noqa: E402
from stressnet.data.pool import CALM, EXERTION, STRESS, dataset_family  # noqa: E402
from stressnet.training import get_device  # noqa: E402

S_, A_, N_ = "exercise_stress", "exercise_aerobic", "exercise_anaerobic"
BUDGETS = [1, 3, 5]
SCENARIOS = ["S_to_exercise", "SA_to_N", "SN_to_A"]


def calibrate_idx(windows, idx, k, rng):
    """Up to k windows per class present among ``windows[idx]`` (>= 60 s apart); returns indices into windows."""
    sub = [windows[i] for i in idx]
    cal, _ = P.sample_calibration(sub, k, rng)
    return idx[cal]


def predict_all(model, emb, logits, cal, test, y):
    y_cal = y[cal]
    out = {"global": logits[test].argmax(1).numpy()}
    out["bias"] = (logits[test] + P.fit_bias(logits[cal], y_cal)).argmax(1).numpy()
    head = P.fit_head(model.head, emb[cal], y_cal)
    with torch.no_grad():
        out["head"] = head(emb[test]).argmax(1).numpy()
    return out


def person_rows(model, emb, windows, args, rng):
    y = np.array([w["model_label"] for w in windows])
    sess = np.array([w["dataset"] for w in windows])
    idx = {s: np.where(sess == s)[0] for s in (S_, A_, N_)}
    if len(idx[S_]) == 0 or len(set(y[idx[S_]].tolist()) & {CALM, STRESS}) < 2:
        return []
    with torch.no_grad():
        logits = model.head(emb)
    rows = []
    for k in BUDGETS:
        for draw in range(args.draws):
            calS = calibrate_idx(windows, idx[S_], k, rng)
            jobs = []
            if len(idx[A_]) + len(idx[N_]) >= 20:
                jobs.append(("S_to_exercise", calS, np.concatenate([idx[A_], idx[N_]])))
            if len(idx[A_]) and len(idx[N_]) >= 20:
                jobs.append(("SA_to_N", np.concatenate([calS, calibrate_idx(windows, idx[A_], k, rng)]), idx[N_]))
            if len(idx[N_]) and len(idx[A_]) >= 20:
                jobs.append(("SN_to_A", np.concatenate([calS, calibrate_idx(windows, idx[N_], k, rng)]), idx[A_]))
            for scen, cal, test in jobs:
                for m, pr in predict_all(model, emb, logits, cal, test, y).items():
                    rows.append({"scenario": scen, "k": k, "draw": draw, "method": m, "n_cal": len(cal),
                                 "n_test": len(test), "frac_calm": float((pr == CALM).mean()),
                                 "frac_stress": float((pr == STRESS).mean()),
                                 "frac_exertion": float((pr == EXERTION).mean())})
    return rows


def run(args):
    pool = twe.load_pool()
    y = np.array([w["model_label"] for w in pool])
    person = np.array([w["dataset"].split("_")[0] + ":" + w["subject"] for w in pool])
    folds = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0).split(pool, y, person))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    device = get_device()
    for cfg in args.configs:
        for seed in args.seeds:
            for k, (tr, te) in enumerate(folds):
                f = out / f"{cfg}_s{seed}_fold{k}.csv"
                if f.exists():
                    continue
                path = model_dir / f"{cfg}_s{seed}_fold{k}.pt"
                if not path.exists():
                    raise SystemExit(f"missing global model {path}: run scripts/personalization_experiment.py first")
                print(f"=== {cfg} seed {seed} fold {k + 1}/{len(folds)}", flush=True)
                model = pe.get_model(cfg, k, seed, None, args, device, model_dir)
                held = [pool[i] for i in te]
                hp = person[te]
                ex = np.array([dataset_family(w) == "exercise" for w in held])
                emb = P.embed(model, [w for w, e in zip(held, ex) if e], device)
                held_ex = [w for w, e in zip(held, ex) if e]
                hp_ex = hp[ex]
                rows = []
                for pid in sorted(set(hp_ex)):
                    idx = np.where(hp_ex == pid)[0]
                    rng = np.random.default_rng(zlib.crc32(f"{seed}|{k}|{pid}|xs".encode()))
                    rs = person_rows(model, emb[idx], [held_ex[i] for i in idx], args, rng)
                    rows += [{"config": cfg, "seed": seed, "fold": k, "person": pid, **r} for r in rs]
                pd.DataFrame(rows).to_csv(f, index=False)
    print("finished", flush=True)


def summarize(args):
    out = Path(args.out)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for cfg in args.configs:
        files = sorted(out.glob(f"{cfg}_s*_fold*.csv"))
        files = [f for f in files if f.stat().st_size > 5]
        if not files:
            continue
        d = pd.concat(map(pd.read_csv, files))
        per = (d.groupby(["scenario", "person", "k", "method"])[["frac_calm", "frac_stress", "frac_exertion"]]
               .mean().reset_index())
        print(f"\n################ {cfg}: {per.person.nunique()} people, {d.seed.nunique()} seed(s), "
              f"{d.draw.nunique()} draws ################")
        for scen in SCENARIOS:
            sc = per[per.scenario == scen]
            if sc.empty:
                continue
            print()
            print(f"--- {scen}: all test windows are exertion; "
                  f"share predicted exertion (higher is better) / stress (false alarm, lower is better) ---")
            tab = sc.groupby(["method", "k"])[["frac_exertion", "frac_stress", "frac_calm"]].mean().round(3)
            n = sc.groupby(["method", "k"]).person.nunique().rename("people")
            print(tab.join(n).unstack("k").to_string())
            rows = []
            for m in ("bias", "head"):
                for k in BUDGETS:
                    g = sc[(sc.method == "global") & (sc.k == k)].set_index("person")
                    a = sc[(sc.method == m) & (sc.k == k)].set_index("person")
                    for col, better in (("frac_exertion", 1), ("frac_stress", -1)):
                        diff = (a[col] - g[col]).dropna()
                        if len(diff) < 5:
                            continue
                        try:
                            p = wilcoxon(diff).pvalue
                        except ValueError:
                            p = np.nan
                        rows.append({"method": m, "k": k, "metric": col, "n": len(diff),
                                     "mean_change": diff.mean(),
                                     "people_better": f"{int(((diff * better) > 1e-9).sum())}/{len(diff)}",
                                     "wilcoxon_p": p})
            print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["bvp_eda_acc"], choices=list(twe.CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument("--model-dir", default=str(ROOT / "cache/personalization_models"))
    ap.add_argument("--out", default=str(ROOT / "results/cross_session"))
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if not a.summarize:
        run(a)
    summarize(a)
