"""Section 7 — Découpes honnêtes (groupe et temporelle)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_groups(df: pd.DataFrame) -> pd.Series:
    """Groupe = ville + date de l'observation."""
    return (
        df["city"].str.lower().str.strip()
        + "|"
        + df["dt"].dt.date.astype(str)
    )


def group_split(
    df: pd.DataFrame,
    y_col: str = "canular",
    test_size: float = 0.25,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Découpe par groupe pour éviter le group leakage."""
    groups = make_groups(df)
    feature_cols = ["temoin", "shape", "country", "heure", "duree"]
    X = df[feature_cols]
    y = df[y_col]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(gss.split(X, y, groups=groups))


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Découpe temporelle sur date_posted."""
    ordre = df.sort_values("date_posted_dt").index
    coupe = int(train_ratio * len(ordre))
    i_train = ordre[:coupe].to_numpy()
    i_test = ordre[coupe:].to_numpy()
    return i_train, i_test


def label_shift_report(
    df: pd.DataFrame,
    i_train: np.ndarray,
    i_test: np.ndarray,
    target_col: str = "canular",
) -> dict:
    """Analyse du label shift entre train et test."""
    y_train = df.loc[i_train, target_col]
    y_test = df.loc[i_test, target_col]
    return {
        "taux_train": float(y_train.mean()),
        "taux_test": float(y_test.mean()),
        "n_canular_train": int(y_train.sum()),
        "n_canular_test": int(y_test.sum()),
    }


def yearly_canular_rate(df: pd.DataFrame, target_col: str = "canular") -> pd.DataFrame:
    """Taux de canulars par année de publication."""
    tmp = df.copy()
    tmp["annee"] = tmp["date_posted_dt"].dt.year
    return (
        tmp.groupby("annee")[target_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "taux_canular", "count": "n_releves"})
        .reset_index()
    )
