"""Section 1 — Chargement CSV avec comptage des lignes écartées."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

CHAMPS = [
    "datetime",
    "city",
    "state",
    "country",
    "shape",
    "duration_seconds",
    "duration_hours_min",
    "comments",
    "date_posted",
    "latitude",
    "longitude",
]


def load_csv_robust(filepath: str | Path) -> tuple[pd.DataFrame, list[list[str]]]:
    """Charge le CSV ligne par ligne sans perte silencieuse."""
    filepath = Path(filepath)
    gardees: list[list[str]] = []
    ecartees: list[list[str]] = []

    with open(filepath, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for ligne in reader:
            if len(ligne) == len(CHAMPS):
                gardees.append(ligne)
            else:
                ecartees.append(ligne)

    df = pd.DataFrame(gardees, columns=CHAMPS)
    return df, ecartees


def load_report(filepath: str | Path) -> dict:
    """Retourne un rapport de chargement."""
    df, ecartees = load_csv_robust(filepath)
    return {
        "n_gardees": len(df),
        "n_ecartees": len(ecartees),
        "n_total": len(df) + len(ecartees),
        "df": df,
        "ecartees": ecartees,
    }
