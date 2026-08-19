"""Phase 2 — Test d'acceptation : mémoriser 8 relevés."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from src.shape.dataset import load_shape_data, prepare_shape_labels
from src.shape.model import ShapeClassifier, pad_batch
from src.shape.vocab import build_vocab, texts_to_bow

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "releves_klaxo3.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

# 8 formes distinctes, commentaires descriptifs (sélection reproductible)
TARGET_SHAPES = [
    "triangle",
    "circle",
    "cigar",
    "fireball",
    "disk",
    "rectangle",
    "oval",
    "chevron",
]


def select_eight_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne 8 relevés avec formes distinctes et texte non vide."""
    rows = []
    used_shapes = set()
    candidates = df[df["comments"].str.len() > 20].copy()
    candidates = candidates.sort_values("comments", key=lambda s: s.str.len(), ascending=False)

    for shape in TARGET_SHAPES:
        pool = candidates[candidates["shape"] == shape]
        if pool.empty:
            pool = df[df["shape"] == shape]
        if pool.empty:
            raise ValueError(f"Aucun relevé trouvé pour la forme '{shape}'")
        row = pool.iloc[0]
        rows.append(row)
        used_shapes.add(shape)

    return pd.DataFrame(rows).reset_index(drop=True)


def train_overfit(
    max_epochs: int = 500,
    lr: float = 0.05,
    embed_dim: int = 64,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)

    df_raw = load_shape_data(DATA_PATH)
    df, _ = prepare_shape_labels(df_raw, drop_missing=True, merge_fourre_tout=False)
    sample_df = select_eight_samples(df)

    texts = sample_df["comments"].tolist()
    labels = sample_df["shape"].tolist()
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)

    vocab = build_vocab(texts)
    token_lists = texts_to_bow(texts, vocab)
    x_ids, lengths = pad_batch(token_lists)

    model = ShapeClassifier(
        vocab_size=len(vocab),
        n_classes=len(labels),
        embed_dim=embed_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    changes_log = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_ids, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()

        if acc == 1.0:
            break

    model.eval()
    with torch.no_grad():
        logits = model(x_ids, lengths)
        preds = logits.argmax(dim=1).tolist()
        pred_labels = [labels[i] for i in preds]

    results = {
        "epochs": epoch,
        "final_loss": losses[-1],
        "accuracy": float((torch.tensor(preds) == y).float().mean()),
        "samples": [
            {
                "index": int(sample_df.iloc[i].name),
                "datetime": sample_df.iloc[i]["datetime"],
                "true_shape": labels[i],
                "pred_shape": pred_labels[i],
                "comment": texts[i][:120],
            }
            for i in range(8)
        ],
        "hyperparams": {
            "lr": lr,
            "embed_dim": embed_dim,
            "max_epochs": max_epochs,
            "optimizer": "Adam",
            "architecture": f"Embedding({len(vocab)}, {embed_dim}) → mean pool → Linear(8)",
        },
        "changes_log": changes_log,
        "vocab_size": len(vocab),
    }

    # Courbe de perte
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, linewidth=2)
    ax.set_xlabel("Itération")
    ax.set_ylabel("Loss (cross-entropy)")
    ax.set_title("Phase 2 — Overfit sur 8 relevés")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase2_overfit_loss.png", dpi=150)
    plt.close(fig)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "phase2_overfit.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def main():
    results = train_overfit()
    print("=== Phase 2 — Test d'acceptation ===")
    print(f"Itérations : {results['epochs']}")
    print(f"Loss finale : {results['final_loss']:.6f}")
    print(f"Accuracy : {results['accuracy']:.0%}")
    print("\nPrédictions finales :")
    for s in results["samples"]:
        ok = "✓" if s["true_shape"] == s["pred_shape"] else "✗"
        print(f"  {ok} {s['true_shape']:12} → {s['pred_shape']:12} | {s['comment'][:60]}...")
    print(f"\nFigure : reports/figures/phase2_overfit_loss.png")
    if results["accuracy"] < 1.0:
        raise SystemExit("Échec : les 8 relevés ne sont pas tous corrects.")


if __name__ == "__main__":
    main()
