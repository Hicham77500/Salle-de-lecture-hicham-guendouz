#!/usr/bin/env python3
"""Télécharge le dataset NUFORC public et le sauvegarde comme releves_klaxo3.csv."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

URL = (
    "https://raw.githubusercontent.com/planetsig/ufo-reports/master/"
    "csv-data/ufo-scrubbed-geocoded-time-standardized.csv"
)

COLUMN_MAP = {
    "datetime": "datetime",
    "city": "city",
    "state": "state",
    "country": "country",
    "shape": "shape",
    "duration (seconds)": "duration_seconds",
    "duration (hours/min)": "duration_hours_min",
    "comments": "comments",
    "date posted": "date_posted",
    "latitude": "latitude",
    "longitude": "longitude",
}

EXPECTED_COLUMNS = list(COLUMN_MAP.values())


def download(output_path: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[1]
    output_path = output_path or root / "data" / "releves_klaxo3.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Téléchargement depuis {URL} ...")
    # Le fichier NUFORC n'a pas de ligne d'en-tête
    df = pd.read_csv(URL, header=None, names=EXPECTED_COLUMNS, low_memory=False)
    df.to_csv(output_path, index=False)
    print(f"Sauvegardé : {output_path} ({len(df):,} lignes, {len(df.columns)} colonnes)")
    return output_path


if __name__ == "__main__":
    try:
        download()
    except Exception as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        sys.exit(1)
