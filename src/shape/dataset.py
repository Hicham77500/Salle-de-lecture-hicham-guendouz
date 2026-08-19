"""Chargement et préparation des données pour la tâche shape ← comments."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from src.load_data import load_csv_robust

# Fusions de doublons sémantiques (Phase 3)
SHAPE_MERGE_MAP = {
    "round": "circle",
    "changed": "changing",
}

FOURRE_TOUT = {"unknown", "other", "inconnu"}
RARE_LABEL = "rare"


def load_shape_data(filepath: str | Path) -> pd.DataFrame:
    """Charge les relevés et prépare texte + cible shape."""
    df, _ = load_csv_robust(filepath)
    df["comments"] = df["comments"].fillna("").map(html.unescape).str.strip()
    df["shape"] = df["shape"].fillna("").str.lower().str.strip()
    df["shape"] = df["shape"].replace(SHAPE_MERGE_MAP)
    df["dt"] = pd.to_datetime(df["datetime"], format="%m/%d/%Y %H:%M", errors="coerce")
    return df


def prepare_shape_labels(
    df: pd.DataFrame,
    drop_missing: bool = True,
    merge_fourre_tout: bool = True,
    min_class_count: int = 50,
    min_comment_len: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """
    Prépare les labels shape avec décisions documentées (Phase 3).

    Returns
    -------
    df_filtered, decisions
    """
    decisions: dict = {}
    out = df.copy()
    decisions["rows_raw"] = len(out)

    n_missing = int(out["shape"].eq("").sum())
    decisions["missing_shape"] = n_missing
    if drop_missing:
        out = out[out["shape"] != ""].copy()
        decisions["action_missing"] = "exclure (2922 attendus sur fichier cours)"

    n_empty_comments = int((out["comments"].str.len() < min_comment_len).sum())
    decisions["empty_short_comments"] = n_empty_comments
    out = out[out["comments"].str.len() >= min_comment_len].copy()
    decisions["action_comments"] = f"exclure commentaires < {min_comment_len} caractères"

    if merge_fourre_tout:
        n_fourre = int(out["shape"].isin(FOURRE_TOUT).sum())
        decisions["fourre_tout_count"] = n_fourre
        out.loc[out["shape"].isin(FOURRE_TOUT), "shape"] = "other_merged"
        decisions["action_fourre_tout"] = "fusionner unknown/other/inconnu → other_merged"

    decisions["action_doublons"] = "round→circle, changed→changing"

    counts = out["shape"].value_counts()
    rare_shapes = counts[counts < min_class_count].index.tolist()
    decisions["rare_shapes_merged"] = rare_shapes
    decisions["min_class_count"] = min_class_count
    out.loc[out["shape"].isin(rare_shapes), "shape"] = RARE_LABEL

    decisions["n_classes"] = int(out["shape"].nunique())
    decisions["n_rows"] = len(out)
    decisions["class_counts"] = out["shape"].value_counts().to_dict()
    return out, decisions


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.75,
    time_col: str = "dt",
) -> tuple[np.ndarray, np.ndarray]:
    """Découpe temporelle sur la date d'observation."""
    ordered = df.sort_values(time_col).index.to_numpy()
    cut = int(train_ratio * len(ordered))
    return ordered[:cut], ordered[cut:]


def encode_labels(
    y_train: pd.Series,
    y_val: pd.Series,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Encode les labels sur les classes présentes dans le train."""
    classes = sorted(y_train.unique())
    label_to_idx = {label: i for i, label in enumerate(classes)}
    y_train_enc = y_train.map(label_to_idx).to_numpy()
    y_val_enc = y_val.map(label_to_idx).fillna(-1).astype(int).to_numpy()
    if (y_val_enc < 0).any():
        unknown = y_val[y_val_enc < 0].unique().tolist()
        raise ValueError(f"Classes val absentes du train : {unknown}")
    return y_train_enc, y_val_enc, label_to_idx
