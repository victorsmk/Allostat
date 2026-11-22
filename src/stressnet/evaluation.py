"""Helpers for summarizing deep-model LOSO results and comparing conditions."""

from __future__ import annotations

import pandas as pd
from scipy.stats import wilcoxon


def extract_metrics(loso_results, subset_key="full"):
    """Per-fold macro-F1 / balanced accuracy for one eval subset of loso_deep_cv output."""
    return pd.DataFrame([
        {"held_out": f["held_out_subject"],
         "macro_f1": f["subsets"][subset_key]["macro_f1"],
         "balanced_accuracy": f["subsets"][subset_key]["balanced_accuracy"]}
        for f in loso_results
    ])


def summarize_subsets(loso_results):
    """mean/std of macro-F1 and balanced accuracy per eval subset."""
    rows = [
        {"held_out": f["held_out_subject"], "subset": name,
         "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"]}
        for f in loso_results for name, m in f["subsets"].items()
    ]
    return pd.DataFrame(rows).groupby("subset")[["macro_f1", "balanced_accuracy"]].agg(["mean", "std"])


def paired_comparison(df_a, df_b, name_a, name_b, col="macro_f1", verbose=True):
    """Paired per-subject comparison with a Wilcoxon signed-rank test. Returns (table, p_value)."""
    m = (df_a.set_index("held_out")[col].rename(name_a).to_frame()
         .join(df_b.set_index("held_out")[col].rename(name_b)))
    m["diff"] = m[name_b] - m[name_a]
    p = float(wilcoxon(m[name_a], m[name_b]).pvalue)
    if verbose:
        print(m.round(3).sort_values("diff"))
        print(f"mean diff {m['diff'].mean():+.3f} | median {m['diff'].median():+.3f} | wilcoxon p={p:.3f}")
    return m, p
