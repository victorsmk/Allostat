"""Can personalization from calm/stress feedback be made to keep exertion detection intact?

Same people, folds and cached global models as scripts/cross_session_experiment.py. Compared on the same
calibration draws:
  global         no adaptation
  bias / head    the plain adapters (the ones that broke exertion)
  bias_masked    classes without calibration labels are excluded from the loss and keep offset 0
  head_masked    same for the head fine-tuning
  head_replay    + a few exemplars of EVERY class from other (training-fold) people, weight 0.3
  head_pseudo    + the person's own unlabeled windows the global model is confident about (p >= 0.8), labelled with
                 its prediction, weight 0.3.  SEMI-SUPERVISED: it looks at unlabeled test-session inputs, so it is
                 an upper bound, not a clean cross-session result.
Scenarios: S_to_exercise, SA_to_N, SN_to_A (test windows are all exertion: share predicted exertion / stress), and
S_within (calibrate on the stress session, test on its windows >= 300 s away: does protection hurt the calm/stress
gain? metric: balanced accuracy over calm/stress, plus how much of it is wrongly called exertion).

  python scripts/cross_session_fix.py --seeds 0 1
"""
import argparse
import sys
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

import cross_session_experiment as cs  # noqa: E402
import personalization_experiment as pe  # noqa: E402
import three_way_experiment as twe  # noqa: E402
from stressnet import personalization as P  # noqa: E402
from stressnet.data.pool import CALM, EXERTION, STRESS, dataset_family  # noqa: E402
from stressnet.training import get_device  # noqa: E402

S_, A_, N_ = cs.S_, cs.A_, cs.N_
BUDGETS = cs.BUDGETS
METHODS = ["global", "bias", "bias_masked", "head", "head_masked", "head_replay", "head_pseudo"]
SCENARIOS = ["S_to_exercise", "SA_to_N", "SN_to_A", "S_within"]
N_REPLAY_PER_CLASS = 15
PSEUDO_CONF, PSEUDO_CAP = 0.8, 30


def adapt_all(model, emb, logits, cal, test, y, replay, rng):
    y_cal = y[cal]
    preds = {"global": logits[test].argmax(1).numpy()}

    def head_pred(h):
        with torch.no_grad():
            return h(emb[test]).argmax(1).numpy()

    preds["bias"] = (logits[test] + P.fit_bias(logits[cal], y_cal)).argmax(1).numpy()
    preds["bias_masked"] = (logits[test] + P.fit_bias(logits[cal], y_cal, mask_absent=True)).argmax(1).numpy()
    preds["head"] = head_pred(P.fit_head(model.head, emb[cal], y_cal))
    preds["head_masked"] = head_pred(P.fit_head(model.head, emb[cal], y_cal, mask_absent=True))
    preds["head_replay"] = head_pred(P.fit_head(model.head, emb[cal], y_cal,
                                                extra_emb=replay[0], extra_y=replay[1]))
    probs = torch.softmax(logits[test], dim=1)
    conf, lab = probs.max(1)
    sel = []
    for c in range(3):
        ids = np.where((lab.numpy() == c) & (conf.numpy() >= PSEUDO_CONF))[0]
        sel.extend(rng.permutation(ids)[:PSEUDO_CAP].tolist())
    sel = np.array(sel, dtype=int)
    if len(sel):
        preds["head_pseudo"] = head_pred(P.fit_head(model.head, emb[cal], y_cal, extra_emb=emb[test][sel],
                                                    extra_y=lab.numpy()[sel]))
    else:
        preds["head_pseudo"] = preds["head"]
    return preds


def person_rows(model, emb, windows, replay, args, rng):
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
            calS = cs.calibrate_idx(windows, idx[S_], k, rng)
            jobs = []
            if len(idx[A_]) + len(idx[N_]) >= 20:
                jobs.append(("S_to_exercise", calS, np.concatenate([idx[A_], idx[N_]])))
            if len(idx[A_]) and len(idx[N_]) >= 20:
                jobs.append(("SA_to_N", np.concatenate([calS, cs.calibrate_idx(windows, idx[A_], k, rng)]), idx[N_]))
            if len(idx[N_]) and len(idx[A_]) >= 20:
                jobs.append(("SN_to_A", np.concatenate([calS, cs.calibrate_idx(windows, idx[N_], k, rng)]), idx[A_]))
            # S_within: calibration and test both in the stress session, test >= 300 s from any calibration window
            sub = [windows[i] for i in idx[S_]]
            cal_w, test_w = P.sample_calibration(sub, k, rng, min_gap_sec=300.0)
            if len(test_w) >= 20 and len(set(y[idx[S_]][test_w].tolist())) >= 2:
                jobs.append(("S_within", idx[S_][cal_w], idx[S_][test_w]))
            for scen, cal, test in jobs:
                for m, pr in adapt_all(model, emb, logits, cal, test, y, replay, rng).items():
                    row = {"scenario": scen, "k": k, "draw": draw, "method": m, "n_cal": len(cal),
                           "n_test": len(test), "frac_calm": float((pr == CALM).mean()),
                           "frac_stress": float((pr == STRESS).mean()), "frac_exertion": float((pr == EXERTION).mean())}
                    if scen == "S_within":
                        row["bal_acc_cs"] = balanced_accuracy_score(y[test], pr)
                    rows.append(row)
    return rows


def run(args):
    pool = twe.load_pool()
    y = np.array([w["model_label"] for w in pool])
    person = np.array([w["dataset"].split("_")[0] + ":" + w["subject"] for w in pool])
    folds = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0).split(pool, y, person))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    cfg = args.config
    for seed in args.seeds:
        for k, (tr, te) in enumerate(folds):
            f = out / f"{cfg}_s{seed}_fold{k}.csv"
            if f.exists():
                continue
            print(f"=== {cfg} seed {seed} fold {k + 1}/{len(folds)}", flush=True)
            model = pe.get_model(cfg, k, seed, None, args, device, Path(args.model_dir))
            # replay exemplars: a few random training-fold windows of every class
            rng_r = np.random.default_rng(zlib.crc32(f"replay|{seed}|{k}".encode()))
            sel = []
            for c in range(3):
                ids = [i for i in tr if y[i] == c]
                sel += rng_r.choice(ids, size=min(N_REPLAY_PER_CLASS, len(ids)), replace=False).tolist()
            replay = (P.embed(model, [pool[i] for i in sel], device), np.array([y[i] for i in sel]))
            held = [pool[i] for i in te if dataset_family(pool[i]) == "exercise"]
            hp = np.array([person[i] for i in te if dataset_family(pool[i]) == "exercise"])
            emb = P.embed(model, held, device)
            rows = []
            for pid in sorted(set(hp)):
                idx = np.where(hp == pid)[0]
                rng = np.random.default_rng(zlib.crc32(f"{seed}|{k}|{pid}|fix".encode()))
                rows += [{"config": cfg, "seed": seed, "fold": k, "person": pid, **r}
                         for r in person_rows(model, emb[idx], [held[i] for i in idx], replay, args, rng)]
            pd.DataFrame(rows).to_csv(f, index=False)
    print("finished", flush=True)


def paired(sc, a_name, b_name, col, better, k):
    a = sc[(sc.method == a_name) & (sc.k == k)].set_index("person")[col]
    b = sc[(sc.method == b_name) & (sc.k == k)].set_index("person")[col]
    diff = (b - a).dropna()
    if len(diff) < 5:
        return None
    try:
        p = wilcoxon(diff).pvalue
    except ValueError:
        p = np.nan
    return {"k": k, "n": len(diff), "mean_change": diff.mean(),
            "people_better": f"{int(((diff * better) > 1e-9).sum())}/{len(diff)}", "wilcoxon_p": p}


def summarize(args):
    out = Path(args.out)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    files = [f for f in sorted(out.glob(f"{args.config}_s*_fold*.csv")) if f.stat().st_size > 5]
    if not files:
        return
    d = pd.concat(map(pd.read_csv, files))
    cols = ["frac_exertion", "frac_stress", "frac_calm"] + (["bal_acc_cs"] if "bal_acc_cs" in d else [])
    per = d.groupby(["scenario", "person", "k", "method"])[cols].mean().reset_index()
    print(f"################ {args.config}: {per.person.nunique()} people, {d.seed.nunique()} seed(s), "
          f"{d.draw.nunique()} draws ################")
    for scen in SCENARIOS:
        sc = per[per.scenario == scen]
        if sc.empty:
            continue
        print()
        if scen == "S_within":
            print("--- S_within: test = stress-session windows >= 300 s from calibration; "
                  "bal_acc_cs = balanced accuracy over calm/stress (higher better); frac_exertion = wrongly called exertion ---")
            show = ["bal_acc_cs", "frac_exertion"]
        else:
            print(f"--- {scen}: all test windows are exertion: frac_exertion (higher better), "
                  f"frac_stress = false alarm (lower better) ---")
            show = ["frac_exertion", "frac_stress"]
        tab = sc.groupby(["method", "k"])[show].mean().round(3).unstack("k")
        print(tab.reindex(METHODS).to_string())
        print("people:", sc.person.nunique())
        rows = []
        for m in METHODS[1:]:
            for k in BUDGETS:
                for col, better in ((show[0], 1 if scen != "S_within" or True else 1),):
                    r = paired(sc, "global", m, col, better, k)
                    if r:
                        rows.append({"method": m, "vs": "global", "metric": col, **r})
        if scen == "S_to_exercise":
            for m in ("bias_masked", "head_masked", "head_replay", "head_pseudo"):
                for k in BUDGETS:
                    r = paired(sc, "head", m, "frac_exertion", 1, k)
                    if r:
                        rows.append({"method": m, "vs": "head", "metric": "frac_exertion", **r})
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="bvp_eda_acc", choices=list(twe.CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument("--model-dir", default=str(ROOT / "cache/personalization_models"))
    ap.add_argument("--out", default=str(ROOT / "results/cross_session_fix"))
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if not a.summarize:
        run(a)
    summarize(a)
