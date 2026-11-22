"""Classical (feature-based) baselines: models, LOSO evaluation, and the quality-gate ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .data.quality import apply_gate
from .data.splits import to_binary_windows
from .features import build_feature_df, feature_columns


def get_models():
    """Fresh, unfit pipelines. Call every time you are about to fit so no state leaks between folds."""
    return {
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                           random_state=42, n_jobs=-1)),
        ]),
        "hist_gb": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05,
                                                   random_state=42)),
        ]),
        "logreg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
    }


def fit_with_sample_weight(pipeline, X, y):
    pipeline.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    return pipeline


def evaluate(pipeline, X, y, labels):
    preds = pipeline.predict(X)
    return {
        "balanced_accuracy": balanced_accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        "per_class_f1": f1_score(y, preds, average=None, labels=labels),
        "confusion_matrix": confusion_matrix(y, preds, labels=labels),
        "y_true": y, "y_pred": preds,
    }


def loso_cv(df, feature_cols, model_name, labels):
    """Leave-one-subject-out CV on a feature DataFrame with 'subject' and 'label' columns.
    ``labels``: all class ids, e.g. [0, 1, 2, 3] (4-class) or [0, 1] (binary)."""
    fold_metrics = []
    for held_out in sorted(df["subject"].unique()):
        tr, va = df["subject"] != held_out, df["subject"] == held_out
        pipeline = fit_with_sample_weight(get_models()[model_name], df.loc[tr, feature_cols], df.loc[tr, "label"])
        m = evaluate(pipeline, df.loc[va, feature_cols], df.loc[va, "label"], labels)
        m["held_out_subject"] = held_out
        fold_metrics.append(m)
    return fold_metrics


def summarize_folds(loso_results, class_names):
    """{(features, model): folds} -> DataFrame of mean/std balanced accuracy and macro-F1 + per-class F1."""
    rows = []
    for (feat_name, model_name), folds in loso_results.items():
        bal = np.array([f["balanced_accuracy"] for f in folds])
        f1 = np.array([f["macro_f1"] for f in folds])
        per_class = np.array([f["per_class_f1"] for f in folds])
        row = {"features": feat_name, "model": model_name,
               "balanced_acc_mean": bal.mean(), "balanced_acc_std": bal.std(),
               "macro_f1_mean": f1.mean(), "macro_f1_std": f1.std()}
        for i, cname in enumerate(class_names):
            row[f"f1_{cname}"] = per_class[:, i].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def gate_ablation_classical(windows_bin, model_names=("logreg", "random_forest"), motion_pct=90,
                            max_bad_ibi=0.15):
    """Train under each quality gate (clip_only / motion / ibi / motion+ibi) and score LOSO F1 on the gate's own
    kept windows ('own'), the strictest common set ('common'), and everything ('all')."""
    base = [w for w in windows_bin if not w["clip_flag"]]
    _, rep = apply_gate(base, clip=False, motion=True, ibi=True, motion_pct=motion_pct, max_bad_ibi=max_bad_ibi)
    df = build_feature_df(base, mode="wrist")
    ok_m, ok_i = ~rep["motion"].values, ~rep["ibi"].values
    conditions = {"clip_only": np.ones(len(df), bool), "motion": ok_m, "ibi": ok_i, "motion+ibi": ok_m & ok_i}
    strict = conditions["motion+ibi"]
    cols = feature_columns(df)
    subj = df["subject"].values

    rows = []
    for model_name in model_names:
        for cname, train_ok in conditions.items():
            f1s = {"own": [], "common": [], "all": []}
            for held in sorted(set(subj)):
                tr = (subj != held) & train_ok
                pipe = fit_with_sample_weight(get_models()[model_name], df.loc[tr, cols], df.loc[tr, "label"])
                for ename, emask in {"own": train_ok, "common": strict, "all": np.ones(len(df), bool)}.items():
                    vm = (subj == held) & emask
                    if vm.sum() == 0 or df.loc[vm, "label"].nunique() < 2:
                        f1s[ename].append(np.nan)
                        continue
                    f1s[ename].append(f1_score(df.loc[vm, "label"], pipe.predict(df.loc[vm, cols]), average="macro"))
            rows.append({"model": model_name, "train_gate": cname, "kept_frac": round(train_ok.mean(), 3),
                         **{f"f1_{k}": round(float(np.nanmean(v)), 3) for k, v in f1s.items()}})
    return pd.DataFrame(rows)


__all__ = ["get_models", "fit_with_sample_weight", "evaluate", "loso_cv", "summarize_folds",
           "gate_ablation_classical", "to_binary_windows"]
