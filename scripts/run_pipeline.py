#!/usr/bin/env python3
"""Script principal — exécute le pipeline complet et génère les figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clean import coerce_types, create_label, leakage_report, prepare_features
from src.evaluate import (
    baseline_report,
    calibrate_model,
    evaluate_model,
    optimize_threshold,
    precision_recall_data,
    repeated_cv_pr_auc,
    threshold_comparison,
)
from src.explain import ablation_by_column, audit_by_country, monitoring_report
from src.features import missing_signal_report
from src.load_data import load_report
from src.pipeline import build_pipeline, get_X
from src.split import group_split, label_shift_report, temporal_split, yearly_canular_rate

FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "releves_klaxo3.csv"


def plot_confusion_matrix(cm, path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-canular", "Canular"],
        yticklabels=["Non-canular", "Canular"],
        ax=ax,
    )
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title("Matrice de confusion")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curve(y_true, y_proba, path: Path):
    precision, recall, _ = precision_recall_data(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Courbe Precision-Recall")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(y_true, y_proba, path: Path):
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_pred, prob_true, marker="o", label="Modèle")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Parfait")
    ax.set_xlabel("Probabilité annoncée")
    ax.set_ylabel("Proportion réelle")
    ax.set_title("Diagramme de calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        print("Données absentes — lancement du téléchargement...")
        from scripts.download_data import download

        download(DATA_PATH)

    print("=== Section 1 : Chargement CSV ===")
    report = load_report(DATA_PATH)
    df = report["df"]
    print(f"  Gardées : {report['n_gardees']:,} | Écartées : {report['n_ecartees']:,}")

    print("=== Sections 2-3, 5 : Nettoyage et étiquetage ===")
    df = prepare_features(df)
    print(f"  Canulars : {df['canular'].sum()} ({100 * df['canular'].mean():.2f}%)")
    print(f"  Anti-fuite : {leakage_report(df)}")

    print("=== Section 8 : Signal des valeurs manquantes ===")
    missing_df = missing_signal_report(df)
    print(missing_df.to_string(index=False))

    print("=== Section 7 : Découpe temporelle ===")
    i_train, i_test = temporal_split(df)
    shift = label_shift_report(df, i_train, i_test)
    print(f"  Label shift (temporel) : {shift}")

    i_train_g, i_test_g = group_split(df)
    shift_g = label_shift_report(df, i_train_g, i_test_g)
    print(f"  Label shift (groupe)   : {shift_g}")

    X = get_X(df)
    y = df["canular"]
    X_train, X_test = X.loc[i_train], X.loc[i_test]
    y_train, y_test = y.loc[i_train], y.loc[i_test]

    print("=== Section 6 : Baseline ===")
    baseline = baseline_report(X_train, y_train, X_test, y_test)
    print(f"  DummyClassifier accuracy : {baseline['accuracy']:.4f}")

    print("=== Sections 9-10 : Entraînement pipeline ===")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("=== Sections 4, 11 : Évaluation ===")
    metrics = evaluate_model(y_test, y_pred, y_proba)
    print(f"  Precision : {metrics['precision']:.3f}")
    print(f"  Recall    : {metrics['recall']:.3f}")
    print(f"  PR-AUC    : {metrics['pr_auc']:.3f}")

    optimal = optimize_threshold(y_test, y_proba)
    print(f"  Seuil optimal : {optimal['seuil']:.2f} (coût {optimal['cout']:.0f})")
    print(threshold_comparison(y_test, y_proba).to_string(index=False))

    calib = calibrate_model(pipeline, X_train, y_train, X_test, y_test)
    print(
        f"  Brier raw : {calib['brier_raw']:.4f} | "
        f"calibré : {calib['brier_calibrated']:.4f}"
    )

    print("=== Section 11 : Intervalle PR-AUC (CV stratifiée) ===")
    cv_interval = repeated_cv_pr_auc(pipeline, X, y, n_splits=5)
    print(
        f"  PR-AUC : {cv_interval['pr_auc_mean']:.3f} "
        f"± {cv_interval['pr_auc_std']:.3f} "
        f"[{cv_interval['pr_auc_min']:.3f}, {cv_interval['pr_auc_max']:.3f}]"
    )

    print("=== Section 7b : Comparaison découpe groupe vs temporelle ===")
    pipeline_g = build_pipeline()
    pipeline_g.fit(X.loc[i_train_g], y.loc[i_train_g])
    y_proba_g = pipeline_g.predict_proba(X.loc[i_test_g])[:, 1]
    y_pred_g = (y_proba_g >= 0.5).astype(int)
    metrics_g = evaluate_model(y.loc[i_test_g], y_pred_g, y_proba_g)
    print(
        f"  Groupe   — precision: {metrics_g['precision']:.3f}, "
        f"recall: {metrics_g['recall']:.3f}, pr_auc: {metrics_g['pr_auc']:.3f}"
    )
    print(
        f"  Temporel — precision: {metrics['precision']:.3f}, "
        f"recall: {metrics['recall']:.3f}, pr_auc: {metrics['pr_auc']:.3f}"
    )

    print("=== Section 12 : Audit ===")
    ablation = ablation_by_column(pipeline, X_test, y_test, list(X_test.columns))
    print(ablation.to_string(index=False))
    audit = audit_by_country(df.loc[i_test], y_test, y_pred)
    print(audit.head(10).to_string(index=False))
    print(f"  Monitoring : {monitoring_report(y_proba, optimal['seuil'])}")

    # Figures
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, FIGURES_DIR / "confusion_matrix.png")
    plot_pr_curve(y_test, y_proba, FIGURES_DIR / "precision_recall.png")
    plot_calibration(y_test, y_proba, FIGURES_DIR / "calibration.png")

    # Sauvegarde
    joblib.dump(pipeline, MODELS_DIR / "pipeline_canular.joblib")
    joblib.dump(calib["model"], MODELS_DIR / "pipeline_calibrated.joblib")

    results = {
        "n_rows": len(df),
        "n_canulars": int(df["canular"].sum()),
        "baseline": baseline,
        "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
        "optimal_threshold": optimal,
        "calibration": {
            "brier_raw": calib["brier_raw"],
            "brier_calibrated": calib["brier_calibrated"],
        },
        "label_shift": shift,
        "label_shift_group": shift_g,
        "cv_interval": cv_interval,
        "metrics_group_split": {
            k: v for k, v in metrics_g.items() if k != "classification_report"
        },
        "yearly_rates": yearly_canular_rate(df).to_dict(orient="records"),
    }
    with open(ROOT / "reports" / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nPipeline terminé. Modèle et figures sauvegardés.")
    return results


if __name__ == "__main__":
    main()
