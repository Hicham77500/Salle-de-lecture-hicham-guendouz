"""Phase 0 — Analyse exploratoire du dossier du 4 juillet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.load_data import load_csv_robust


def phase0_report(
    filepath: str | Path,
    date_col: str = "datetime",
    year_min: int = 1990,
    year_max: int = 2014,
) -> dict:
    """
    Reproduit les calculs du dossier du disparu.

    Parameters
    ----------
    date_col : 'datetime' (observation) ou 'date_posted' (publication)
    year_min, year_max : fenêtre temporelle du dossier (1990-2014)
    """
    df, _ = load_csv_robust(filepath)

    if date_col == "datetime":
        dates = pd.to_datetime(df["datetime"], format="%m/%d/%Y %H:%M", errors="coerce")
    else:
        dates = pd.to_datetime(df["date_posted"], format="%m/%d/%Y", errors="coerce")

    df = df.assign(_date=dates).dropna(subset=["_date"])
    df = df[(df["_date"].dt.year >= year_min) & (df["_date"].dt.year <= year_max)]
    df["_day"] = df["_date"].dt.date
    df["_dow"] = df["_date"].dt.dayofweek  # 0=lundi
    df["_month"] = df["_date"].dt.month
    df["_year"] = df["_date"].dt.year

    daily = df.groupby("_day").size()
    n_days = daily.index.nunique()
    mean_per_day = len(df) / n_days

    july4 = df[(df["_date"].dt.month == 7) & (df["_date"].dt.day == 4)]
    n_july4 = len(july4)
    n_july4_years = july4["_date"].dt.year.nunique()
    july4_mean_per_year = round(n_july4 / max(n_july4_years, 1), 1)

    dow_pct = df["_dow"].value_counts(normalize=True).sort_index()
    month_pct = df["_month"].value_counts(normalize=True).sort_index()

    yearly = df.groupby("_year").size()
    daily_sorted = daily.sort_values(ascending=False)
    max_day_count = int(daily_sorted.iloc[0])
    july4_days = july4["_day"].unique()
    if len(july4_days) > 0:
        july4_total = daily.get(july4_days[0], 0)
        july4_rank = int((daily_sorted >= july4_total).sum())
    else:
        july4_rank = None

    top10_days = daily_sorted.head(10)

    return {
        "date_col_used": date_col,
        "n_days": n_days,
        "mean_per_day": round(mean_per_day, 1),
        "n_july4": n_july4,
        "july4_mean_per_year": july4_mean_per_year,
        "pct_saturday": round(dow_pct.get(5, 0) * 100, 1),
        "pct_monday": round(dow_pct.get(0, 0) * 100, 1),
        "pct_july": round(month_pct.get(7, 0) * 100, 1),
        "pct_february": round(month_pct.get(2, 0) * 100, 1),
        "max_day_count": max_day_count,
        "july4_rank": july4_rank,
        "yearly_volume": yearly.to_dict(),
        "top10_days": {str(k): int(v) for k, v in top10_days.items()},
    }


if __name__ == "__main__":
    import json
    import sys

    root = Path(__file__).resolve().parents[2]
    data = root / "data" / "releves_klaxo3.csv"
    if not data.exists():
        print("Exécutez d'abord : python scripts/download_data.py", file=sys.stderr)
        sys.exit(1)

    for col in ("datetime", "date_posted"):
        print(f"\n=== Phase 0 avec {col} ===")
        print(json.dumps(phase0_report(data, col), indent=2, ensure_ascii=False))
