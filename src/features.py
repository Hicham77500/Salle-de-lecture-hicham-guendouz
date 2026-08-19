"""Section 10 — Feature engineering."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd


def unescape_html(text: str) -> str:
    return html.unescape(text)


def heure_cyclique(h: np.ndarray) -> np.ndarray:
    """Encode l'heure sur un cercle (sin/cos)."""
    h = np.asarray(h, dtype=float)
    return np.c_[np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)]


def add_cyclical_hour(df: pd.DataFrame, hour_col: str = "heure") -> pd.DataFrame:
    out = df.copy()
    h = out[hour_col].astype(float)
    out["heure_sin"] = np.sin(2 * np.pi * h / 24)
    out["heure_cos"] = np.cos(2 * np.pi * h / 24)
    return out


def missing_signal_report(df: pd.DataFrame, target_col: str = "canular") -> pd.DataFrame:
    """Section 8 — Taux de canulars avec/sans valeur manquante."""
    rows = []
    for col in ("country", "state", "shape"):
        missing = df[col].isna() | (df[col].astype(str).str.strip() == "")
        rate_if_missing = df.loc[missing, target_col].mean() if missing.any() else np.nan
        rate_if_present = df.loc[~missing, target_col].mean() if (~missing).any() else np.nan
        rows.append(
            {
                "colonne": col,
                "releves_troues": int(missing.sum()),
                "taux_canular_si_trou": rate_if_missing,
                "taux_canular_sinon": rate_if_present,
            }
        )
    return pd.DataFrame(rows)
