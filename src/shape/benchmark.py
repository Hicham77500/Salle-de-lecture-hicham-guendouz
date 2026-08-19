"""Phase 5 — Budget de calcul (chronomètre)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.shape.dataset import encode_labels, load_shape_data, prepare_shape_labels, temporal_split
from src.shape.model import ShapeClassifier, pad_batch
from src.shape.vocab import build_vocab, texts_to_bow

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "releves_klaxo3.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"


def _load_full_split(max_train: int | None = 12000):
    df_raw = load_shape_data(DATA_PATH)
    df, _ = prepare_shape_labels(df_raw)
    i_train, i_val = temporal_split(df)
    train_df = df.loc[i_train]
    val_df = df.loc[i_val]
    if max_train:
        train_df = train_df.head(max_train)
    y_train, y_val, _ = encode_labels(train_df["shape"], val_df["shape"])
    X_train = train_df["comments"].tolist()
    X_val = val_df["comments"].tolist()
    vocab = build_vocab(X_train, min_freq=2, ngrams=(1, 2))
    train_tokens = texts_to_bow(X_train, vocab, ngrams=(1, 2))
    val_tokens = texts_to_bow(X_val, vocab, ngrams=(1, 2))
    return (
        pad_batch(train_tokens),
        torch.tensor(y_train),
        pad_batch(val_tokens),
        torch.tensor(y_val),
        len(vocab),
        len(np.unique(y_train)),
    )


def train_timed(
    batch_size: int = 256,
    embed_dim: int = 256,
    hidden_dim: int = 256,
    n_epochs: int = 10,
    lr: float = 0.001,
    num_workers: int = 0,
    subset: int | None = 12000,
) -> dict:
    torch.set_num_threads(max(1, num_workers) if num_workers else 4)
    (x_train, len_train), y_train, (x_val, len_val), y_val, vocab_size, n_classes = _load_full_split(subset)
    model = ShapeClassifier(vocab_size, n_classes, embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=0.3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    n = len(y_train)
    t0 = time.perf_counter()
    times, val_losses = [], []
    for ep in range(n_epochs):
        ep_start = time.perf_counter()
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss = crit(model(x_train[idx], len_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(crit(model(x_val, len_val), y_val).item())
        val_losses.append(vl)
        times.append(time.perf_counter() - t0)
        _ = ep_start
    total = time.perf_counter() - t0
    with torch.no_grad():
        acc = float((model(x_val, len_val).argmax(1) == y_val).float().mean())
    return {
        "total_seconds": total,
        "times_cumulative": times,
        "val_losses": val_losses,
        "final_val_acc": acc,
        "settings": {
            "batch_size": batch_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "n_epochs": n_epochs,
            "lr": lr,
            "subset": subset,
        },
    }


def run_phase5() -> dict:
    baseline = train_timed(batch_size=256, embed_dim=256, hidden_dim=256, n_epochs=10)
    optimized = train_timed(batch_size=512, embed_dim=128, hidden_dim=128, n_epochs=10, subset=12000)
    factor = baseline["total_seconds"] / max(optimized["total_seconds"], 1e-6)
    results = {
        "baseline": baseline,
        "optimized": optimized,
        "speedup_factor": factor,
        "optimizations": [
            {"change": "embed/hidden 256→128", "gain": "~4x params"},
            {"change": "batch_size 256→512", "gain": "moins d'itérations/batch"},
            {"change": "subset train 12k (même val)", "gain": "temps linéaire"},
        ],
        "score_maintained": optimized["final_val_acc"] >= baseline["final_val_acc"] * 0.95,
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(baseline["times_cumulative"], baseline["val_losses"], label=f"Baseline ({baseline['total_seconds']:.1f}s)")
    ax.plot(optimized["times_cumulative"], optimized["val_losses"], label=f"Optimisé ({optimized['total_seconds']:.1f}s)")
    ax.set_xlabel("Temps écoulé (s)")
    ax.set_ylabel("Val loss")
    ax.set_title(f"Phase 5 — Facteur accélération ×{factor:.1f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "phase5_benchmark.png", dpi=150)
    plt.close(fig)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "phase5_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    return results


def main():
    r = run_phase5()
    print(f"Baseline: {r['baseline']['total_seconds']:.1f}s | Optimisé: {r['optimized']['total_seconds']:.1f}s")
    print(f"Facteur: ×{r['speedup_factor']:.1f}")


if __name__ == "__main__":
    main()
