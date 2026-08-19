"""Section 12 — Explication et audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score


def permutation_importance_report(
    model, X_test, y_test, feature_names: list[str] | None = None, n_repeats: int = 3
) -> pd.DataFrame:
    """Importance par permutation — ablation colonne par colonne."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=0, n_jobs=-1
    )
    names = feature_names or list(getattr(X_test, "columns", range(X_test.shape[1])))
    return (
        pd.DataFrame(
            {
                "feature": names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def ablation_by_column(
    pipeline, X_test, y_test, columns: list[str]
) -> pd.DataFrame:
    """Abîme une colonne à la fois et mesure la chute de PR-AUC."""
    baseline = average_precision_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    rows = []

    for col in columns:
        X_perm = X_test.copy()
        X_perm[col] = np.random.permutation(X_perm[col].values)
        score = average_precision_score(y_test, pipeline.predict_proba(X_perm)[:, 1])
        rows.append(
            {
                "colonne": col,
                "pr_auc": score,
                "chute_pr_auc": baseline - score,
            }
        )

    return pd.DataFrame(rows).sort_values("chute_pr_auc", ascending=False)


def audit_by_country(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    country_col: str = "country",
    min_count: int = 100,
) -> pd.DataFrame:
    """Recalcule precision/recall par zone géographique."""
    tmp = df.copy()
    tmp["y_true"] = y_true.values
    tmp["y_pred"] = y_pred

    rows = []
    for country, grp in tmp.groupby(country_col):
        if len(grp) < min_count:
            continue
        yt = grp["y_true"]
        yp = grp["y_pred"]
        tp = ((yt == 1) & (yp == 1)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append(
            {
                "country": country,
                "n_releves": len(grp),
                "n_canulars": int(yt.sum()),
                "precision": precision,
                "recall": recall,
            }
        )

    return pd.DataFrame(rows).sort_values("n_releves", ascending=False)


def monitoring_report(y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Surveillance sans étiquette — indicateurs en production."""
    return {
        "taux_signale": float((y_proba >= threshold).mean()),
        "prob_moyenne": float(y_proba.mean()),
        "prob_mediane": float(np.median(y_proba)),
        "prob_p95": float(np.percentile(y_proba, 95)),
    }
