"""Sections 2-3 et 5 — Nettoyage, coercion, étiquetage, anti-fuite."""

from __future__ import annotations

import html

import pandas as pd


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce les types numériques et datetime avec errors='coerce'."""
    out = df.copy()

    for col in ("latitude", "longitude", "duration_seconds"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["dt"] = pd.to_datetime(
        out["datetime"], format="%m/%d/%Y %H:%M", errors="coerce"
    )
    out["date_posted_dt"] = pd.to_datetime(
        out["date_posted"], format="%m/%d/%Y", errors="coerce"
    )
    return out


def create_label(df: pd.DataFrame) -> pd.DataFrame:
    """Crée la cible 'canular' via weak supervision (mot hoax dans comments)."""
    out = df.copy()
    com = out["comments"].fillna("").map(html.unescape)
    out["canular"] = com.str.lower().str.contains("hoax", regex=False).astype(int)
    return out


def remove_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les notes de modération Bureau ((...)) du témoignage."""
    out = df.copy()
    com = out["comments"].fillna("").map(html.unescape)
    out["temoin"] = (
        com.str.replace(r"\(\(.*?\)\)", " ", regex=True).str.strip()
    )
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de préparation : coercion + label + anti-fuite + features dérivées."""
    out = coerce_types(df)
    out = create_label(out)
    out = remove_leakage(out)
    out["heure"] = out["dt"].dt.hour.fillna(12).astype(float)
    out["duree"] = out["duration_seconds"]
    out["shape"] = out["shape"].fillna("inconnu").astype(str)
    out["country"] = out["country"].fillna("inconnu").astype(str)
    return out


def coercion_report(df: pd.DataFrame) -> dict:
    """Rapport sur les valeurs fautives après coercion."""
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    dt = pd.to_datetime(df["datetime"], format="%m/%d/%Y %H:%M", errors="coerce")
    dur = pd.to_numeric(df["duration_seconds"], errors="coerce")

    return {
        "latitude_na": int(lat.isna().sum()),
        "latitude_bad": df.loc[lat.isna(), "latitude"].unique().tolist()[:5],
        "datetime_na": int(dt.isna().sum()),
        "duration_na": int(dur.isna().sum()),
    }


def leakage_report(df: pd.DataFrame) -> dict:
    """Statistiques sur le retrait des notes de modération."""
    com = df["comments"].fillna("").map(html.unescape)
    temoin = com.str.replace(r"\(\(.*?\)\)", " ", regex=True).str.strip()
    return {
        "temoins_vides": int((temoin.str.len() == 0).sum()),
        "hoax_dans_temoin": int(temoin.str.lower().str.contains("hoax", regex=False).sum()),
    }
