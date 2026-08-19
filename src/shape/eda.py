"""Phase 0 — Analyse exploratoire du dossier du 4 juillet."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.load_data import load_csv_robust

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


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

    n_rows_raw = len(df)
    df = df.assign(_date=dates).dropna(subset=["_date"])
    n_rows_parsed = len(df)
    df = df[(df["_date"].dt.year >= year_min) & (df["_date"].dt.year <= year_max)]
    df["_day"] = df["_date"].dt.date
    df["_dow"] = df["_date"].dt.dayofweek  # 0=lundi
    df["_month"] = df["_date"].dt.month
    df["_year"] = df["_date"].dt.year

    daily = df.groupby("_day").size()
    n_days = len(daily)
    mean_per_day = round(len(df) / n_days, 1)

    july4 = df[(df["_date"].dt.month == 7) & (df["_date"].dt.day == 4)]
    n_july4 = len(july4)
    n_july4_years = july4["_date"].dt.year.nunique()
    july4_mean_per_year = round(n_july4 / max(n_july4_years, 1), 1)

    dow_pct = df["_dow"].value_counts(normalize=True).sort_index()
    month_pct = df["_month"].value_counts(normalize=True).sort_index()

    yearly = df.groupby("_year").size()
    daily_sorted = daily.sort_values(ascending=False)
    max_day = daily_sorted.index[0]
    max_day_count = int(daily_sorted.iloc[0])

    # Rang du 4 juillet : position de la moyenne annuelle parmi les journées
    july4_rank = int((daily_sorted > july4_mean_per_year).sum()) + 1

    top10_days = daily_sorted.head(10)

    return {
        "date_col_used": date_col,
        "n_rows_raw": n_rows_raw,
        "n_rows_parsed": n_rows_parsed,
        "n_rows_window": len(df),
        "n_days": n_days,
        "mean_per_day": mean_per_day,
        "n_july4": n_july4,
        "july4_mean_per_year": july4_mean_per_year,
        "july4_rank": july4_rank,
        "pct_saturday": round(dow_pct.get(5, 0) * 100, 1),
        "pct_monday": round(dow_pct.get(0, 0) * 100, 1),
        "pct_july": round(month_pct.get(7, 0) * 100, 1),
        "pct_february": round(month_pct.get(2, 0) * 100, 1),
        "max_day": str(max_day),
        "max_day_count": max_day_count,
        "yearly_volume": {int(k): int(v) for k, v in yearly.items()},
        "top10_days": {str(k): int(v) for k, v in top10_days.items()},
    }


def plot_volume_annuel(yearly_volume: dict, output_path: Path | None = None) -> Path:
    """Courbe du volume annuel de relevés."""
    output_path = output_path or FIGURES_DIR / "volume_annuel.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    years = sorted(yearly_volume.keys())
    counts = [yearly_volume[y] for y in years]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(years, counts, marker="o", linewidth=2, markersize=4)
    ax.set_xlabel("Année")
    ax.set_ylabel("Nombre de relevés")
    ax.set_title("Volume annuel de relevés (1990–2014)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_phase0(data_path: Path | None = None) -> dict:
    """Exécute Phase 0 complète : rapport datetime + figure."""
    root = Path(__file__).resolve().parents[2]
    data_path = data_path or root / "data" / "releves_klaxo3.csv"

    report_dt = phase0_report(data_path, "datetime")
    report_posted = phase0_report(data_path, "date_posted")

    fig_path = plot_volume_annuel(report_dt["yearly_volume"])

    return {
        "chosen": "datetime",
        "datetime": report_dt,
        "date_posted": report_posted,
        "figure": str(fig_path),
    }


if __name__ == "__main__":
    import json
    import sys

    root = Path(__file__).resolve().parents[2]
    data = root / "data" / "releves_klaxo3.csv"
    if not data.exists():
        print("Exécutez d'abord : python scripts/download_data.py", file=sys.stderr)
        sys.exit(1)

    result = run_phase0(data)
    print("=== Phase 0 — datetime (choix retenu) ===")
    print(json.dumps(result["datetime"], indent=2, ensure_ascii=False))
    print(f"\nFigure : {result['figure']}")
    print("\n=== Comparaison date_posted ===")
    posted = result["date_posted"]
    print(
        f"  jours={posted['n_days']}, moy/j={posted['mean_per_day']}, "
        f"4juil/an={posted['july4_mean_per_year']}, rang={posted['july4_rank']}"
    )
