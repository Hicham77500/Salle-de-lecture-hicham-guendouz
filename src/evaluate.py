"""Sections 4-6 et 11 — Métriques, baseline, seuil, calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)


@dataclass
class CostGrid:
    cost_missed_hoax: float = 30.0
    cost_false_alarm: float = 2.0


def evaluate_model(y_true, y_pred, y_proba=None) -> dict:
    """Calcule les métriques principales."""
    result = {
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None:
        result["pr_auc"] = float(average_precision_score(y_true, y_proba))
    result["classification_report"] = classification_report(
        y_true, y_pred, zero_division=0
    )
    return result


def baseline_report(X_train, y_train, X_test, y_test) -> dict:
    """Modèle bête DummyClassifier pour référence."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    return {
        "accuracy": float(dummy.score(X_test, y_test)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }


def optimize_threshold(y_true, y_proba, grid: CostGrid | None = None) -> dict:
    """Trouve le seuil optimal selon la grille de coûts du Conseil."""
    grid = grid or CostGrid()
    thresholds = np.linspace(0.01, 0.99, 99)
    best = {"seuil": 0.5, "cout": float("inf"), "fn": 0, "fp": 0}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cout = fn * grid.cost_missed_hoax + fp * grid.cost_false_alarm
        if cout < best["cout"]:
            best = {"seuil": float(t), "cout": float(cout), "fn": int(fn), "fp": int(fp)}

    return best


def threshold_comparison(y_true, y_proba, grid: CostGrid | None = None) -> pd.DataFrame:
    """Compare les coûts aux seuils 0.5 et optimal."""
    grid = grid or CostGrid()
    rows = []
    optimal = optimize_threshold(y_true, y_proba, grid)

    for label, t in [("0.50", 0.5), ("optimal", optimal["seuil"])]:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "seuil": label,
                "canulars_rates": fn,
                "fausses_alertes": fp,
                "facture": fn * grid.cost_missed_hoax + fp * grid.cost_false_alarm,
            }
        )
    return pd.DataFrame(rows)


def calibrate_model(pipeline, X_train, y_train, X_test, y_test, method="isotonic"):
    """Calibre les probabilités et retourne les scores Brier."""
    calibrated = CalibratedClassifierCV(pipeline, method=method, cv=3)
    calibrated.fit(X_train, y_train)

    proba_raw = pipeline.predict_proba(X_test)[:, 1]
    proba_cal = calibrated.predict_proba(X_test)[:, 1]

    return {
        "brier_raw": float(brier_score_loss(y_test, proba_raw)),
        "brier_calibrated": float(brier_score_loss(y_test, proba_cal)),
        "model": calibrated,
        "proba_calibrated": proba_cal,
    }


def calibration_bins(y_true, y_proba, n_bins: int = 10) -> pd.DataFrame:
    """Proportion réelle vs probabilité annoncée."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    return pd.DataFrame(
        {"probabilite_annoncee": prob_pred, "proportion_reelle": prob_true}
    )


def repeated_cv_pr_auc(
    pipeline, X, y, n_splits: int = 5, random_state: int = 0
) -> dict:
    """Intervalle de confiance PR-AUC via validation croisée stratifiée répétée."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        from sklearn.base import clone
        pipe = clone(pipeline)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        scores.append(average_precision_score(y_te, proba))

    scores = np.array(scores)
    return {
        "pr_auc_mean": float(scores.mean()),
        "pr_auc_std": float(scores.std()),
        "pr_auc_min": float(scores.min()),
        "pr_auc_max": float(scores.max()),
        "n_splits": n_splits,
    }


def precision_recall_data(y_true, y_proba) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Données pour la courbe precision-recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return precision, recall, thresholds
