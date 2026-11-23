"""FedAvg vs centralized training, and personalization on top of each.

Same pooled windows (WESAD + E4 Stress + EXERCISE), same 5 person-grouped folds and same 3 classes
(calm / stress / exertion) as scripts/three_way_experiment.py. Every training person is a federated client that
shares weights only. Per fold:
  1. FedAvg global model: ``--rounds`` rounds, 1 local epoch, all clients each round, final-round model used
  2. centralized reference: the cached model of scripts/personalization_experiment.py (seed 0; trained on the
     pooled windows with a class-weighted loss, 15 epochs)
  3. both are scored on the held-out people, and personalized with the plain head fine-tuning and the replay
     variant (stressnet.personalization) on identical calibration draws
Learning curves (held-out balanced accuracy every 5 rounds) are logged for plotting only; nothing is selected on them.

  python scripts/fedavg_experiment.py --rounds 30
"""
import argparse
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

import personalization_experiment as pe  # noqa: E402
import three_way_experiment as twe  # noqa: E402
from stressnet import federated as fed  # noqa: E402
from stressnet import personalization as P  # noqa: E402
from stressnet.data.pool import dataset_family  # noqa: E402
from stressnet.models import MultiModalNet  # noqa: E402
from stressnet.training import get_device  # noqa: E402

CFG = "bvp_eda_acc"
BUDGETS, GAPS = [1, 3, 5], [60.0, 300.0]
N_REPLAY_PER_CLASS = 15


def personalization_rows(model, mtype, emb, windows, replay, args, rng_seed):
    """Rows for one person and one model; the draws depend only on rng_seed so both models see the same ones."""
    y = np.array([w["model_label"] for w in windows])
    with torch.no_grad():
        logits = model.head(emb)
    rows = []
    rng = np.random.default_rng(rng_seed)
    for gap in GAPS:
        for k in BUDGETS:
            for draw in range(args.draws):
                cal, test = P.sample_calibration(windows, k, rng, min_gap_sec=gap)
                if len(test) < 20 or len(set(y[test].tolist())) < 2:
                    continue
                preds = {"global": logits[test].argmax(1).numpy()}
                for name, kw in (("head", {}), ("head_replay", {"extra_emb": replay[0], "extra_y": replay[1]})):
                    h = P.fit_head(model.head, emb[cal], y[cal], **kw)
                    with torch.no_grad():
                        preds[name] = h(emb[test]).argmax(1).numpy()
                for m, pr in preds.items():
                    rows.append({"model": mtype, "gap": gap, "k": k, "draw": draw, "method": m,
                                 "bal_acc3": balanced_accuracy_score(y[test], pr), "n_test": len(test)})
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
    mods = twe.CONFIGS[CFG]
    t0 = time.time()
    for k, (tr, te) in enumerate(folds):
        f_pers = out / f"pers_fold{k}.csv"
        if f_pers.exists() or (args.only_folds and k not in args.only_folds):
            continue
        print(f"=== fold {k + 1}/{len(folds)} | elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        held = [pool[i] for i in te]
        held_y = np.array([w["model_label"] for w in held])

        def curve(model):
            p = twe.predict_proba(model, held, device)
            return {"bal_acc3": balanced_accuracy_score(held_y, p.argmax(1))}

        # ---- FedAvg global model
        path = model_dir / f"{CFG}_fold{k}.pt"
        if path.exists():
            fed_model = MultiModalNet(mods, body="cnn", num_classes=3, dropout=0.3).to(device)
            fed_model.load_state_dict(torch.load(path, map_location=device))
            fed_model.eval()
        else:
            clients = {}
            for i in tr:
                clients.setdefault(person[i], []).append(pool[i])
            clients = {c: twe.thin(ws, args.train_stride) for c, ws in clients.items()}
            fed_model, hist = fed.fedavg(clients, mods, rounds=args.rounds, local_epochs=args.local_epochs,
                                         participation=args.participation, device=device, seed=k,
                                         eval_fn=curve, eval_every=5, verbose=True)
            torch.save(fed_model.state_dict(), path)
            pd.DataFrame(hist).assign(fold=k, n_clients=len(clients)).to_csv(out / f"history_fold{k}.csv", index=False)
            print("  communication:", fed.communication_per_round(fed_model, len(clients)), flush=True)
        central = pe.get_model(CFG, k, 0, None, args, device, ROOT / "cache/personalization_models")

        # ---- held-out predictions of both models (summarised with three_way_experiment.summarize)
        for name, m in (("fedavg", fed_model), ("central", central)):
            twe.frame(held, twe.predict_proba(m, held, device), "cv", k).to_csv(out / f"{name}_cv_fold{k}.csv",
                                                                                 index=False)

        # ---- personalization on top of each global model (same replay exemplars, same calibration draws)
        rng_r = np.random.default_rng(zlib.crc32(f"replay|{k}".encode()))
        sel = []
        for c in range(3):
            ids = [i for i in tr if y[i] == c]
            sel += rng_r.choice(ids, size=min(N_REPLAY_PER_CLASS, len(ids)), replace=False).tolist()
        replay_w, replay_y = [pool[i] for i in sel], np.array([y[i] for i in sel])
        hp = person[te]
        rows = []
        for mtype, m in (("central", central), ("fedavg", fed_model)):
            emb = P.embed(m, held, device)
            replay = (P.embed(m, replay_w, device), replay_y)
            for pid in sorted(set(hp)):
                idx = np.where(hp == pid)[0]
                rs = personalization_rows(m, mtype, emb[idx], [held[i] for i in idx], replay, args,
                                          zlib.crc32(f"{k}|{pid}|fedavg".encode()))
                fam = dataset_family(held[idx[0]])
                rows += [{"fold": k, "person": pid, "family": fam, **r} for r in rs]
        pd.DataFrame(rows).to_csv(f_pers, index=False)
    print("finished", flush=True)


def summarize(args):
    out = Path(args.out)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    hist = [pd.read_csv(f) for f in sorted(out.glob("history_fold*.csv"))]
    if hist:
        h = pd.concat(hist)
        print("################ FedAvg learning curve: held-out balanced accuracy (3-way), mean over folds ################")
        print(h.groupby("round").bal_acc3.agg(["mean", "std"]).round(3).T.to_string())
        n_clients = int(h.groupby("fold").n_clients.first().mean())
        m = MultiModalNet(twe.CONFIGS[CFG], body="cnn", num_classes=3)
        c = fed.communication_per_round(m, n_clients)
        print(f"\ncommunication: {c['parameters']:,} parameters = {c['bytes_per_client'] / 1e3:.0f} kB per client per "
              f"round (down + up); ~{c['bytes_per_round'] / 1e6:.1f} MB per round with {n_clients} clients; "
              f"{int(h['round'].max())} rounds = {c['bytes_per_client'] * h['round'].max() / 1e6:.1f} MB per client in total")
    class A:        # reuse three_way_experiment.summarize on the per-fold prediction files
        pass
    a = A()
    a.out, a.configs = str(out), ["central", "fedavg"]
    twe.summarize(a)
    files = sorted(out.glob("pers_fold*.csv"))
    if not files:
        return
    d = pd.concat(map(pd.read_csv, files))
    per = d.groupby(["family", "person", "model", "gap", "k", "method"]).bal_acc3.mean().reset_index()
    for gap in sorted(per.gap.unique()):
        pg = per[per.gap == gap]
        print(f"\n################ personalization, calibration-test gap {gap:.0f} s: mean 3-way balanced accuracy "
              f"over {pg.person.nunique()} people ################")
        print(pg.groupby(["model", "method", "k"]).bal_acc3.mean().unstack("k").round(3).to_string())
        rows = []
        for method in ("global", "head", "head_replay"):
            for k in BUDGETS:
                a_ = pg[(pg.model == "central") & (pg.method == method) & (pg.k == k)].set_index("person").bal_acc3
                b_ = pg[(pg.model == "fedavg") & (pg.method == method) & (pg.k == k)].set_index("person").bal_acc3
                diff = (b_ - a_).dropna()
                if len(diff) < 5:
                    continue
                try:
                    p = wilcoxon(diff).pvalue
                except ValueError:
                    p = np.nan
                rows.append({"method": method, "k": k, "n": len(diff), "fedavg_minus_central": diff.mean(),
                             "fedavg_better_for": f"{int((diff > 1e-9).sum())}/{len(diff)}", "wilcoxon_p": p})
        print("\nFedAvg vs centralized under the same method (paired over people):")
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--participation", type=float, default=1.0)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results/fedavg"))
    ap.add_argument("--model-dir", default=str(ROOT / "cache/fedavg_models"))
    ap.add_argument("--only-folds", type=int, nargs="+", default=None)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if not a.summarize:
        run(a)
    summarize(a)
