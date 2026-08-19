"""Chargement et préparation des données pour la tâche shape ← comments."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from src.load_data import CHAMPS, load_csv_robust

# Fusions de doublons sémantiques (Phase 3)
SHAPE_MERGE_MAP = {
    "round": "circle",
    "changed": "changing",
}

FOURRE_TOUT = {"unknown", "other", "inconnu"}


def load_shape_data(filepath: str | Path) -> pd.DataFrame:
    """Charge les relevés et prépare texte + cible shape."""
    df, _ = load_csv_robust(filepath)
    df["comments"] = df["comments"].fillna("").map(html.unescape).str.strip()
    df["shape"] = df["shape"].fillna("").str.lower().str.strip()
    df["shape"] = df["shape"].replace(SHAPE_MERGE_MAP)
    return df


def prepare_shape_labels(
    df: pd.DataFrame,
    drop_missing: bool = True,
    merge_fourre_tout: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Prépare les labels shape avec décisions documentées (Phase 3).

    Returns
    -------
    df_filtered, decisions
    """
    decisions = {}
    out = df.copy()
    n_missing = out["shape"].eq("").sum()
    decisions["missing_shape"] = int(n_missing)

    if drop_missing:
        out = out[out["shape"] != ""].copy()
        decisions["action_missing"] = "exclure"

    if merge_fourre_tout:
        out.loc[out["shape"].isin(FOURRE_TOUT), "shape"] = "other_merged"
        decisions["action_fourre_tout"] = "fusionner en other_merged"

    decisions["n_classes"] = out["shape"].nunique()
    decisions["n_rows"] = len(out)
    return out, decisions
