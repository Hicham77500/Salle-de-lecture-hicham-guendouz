"""Phase 3 — PyTorch vs baseline linéaire (TF-IDF + SGD) sur la même découpe."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.shape.dataset import (
    encode_labels,
    load_shape_data,
    prepare_shape_labels,
    temporal_split,
)
from src.shape.model import ShapeClassifier, pad_batch
from src.shape.vocab import build_vocab, texts_to_bow

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "releves_klaxo3.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"


def majority_baseline(y_train: np.ndarray, y_val: np.ndarray) -> dict:
    majority = int(np.bincount(y_train).argmax())
    preds = np.full_like(y_val, majority)
    return {
        "accuracy": float(accuracy_score(y_val, preds)),
        "macro_f1": float(f1_score(y_val, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_val, preds, average="weighted", zero_division=0)),
    }


def train_sklearn_linear(
    X_train: list[str],
    y_train: np.ndarray,
    X_val: list[str],
    y_val: np.ndarray,
    n_epochs: int = 30,
    seed: int = 42,
) -> tuple[dict, list[float], list[float]]:
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=5)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    sample_weight = weights[y_train]

    clf = SGDClassifier(
        loss="log_loss",
        max_iter=1,
        warm_start=True,
        random_state=seed,
        alpha=1e-5,
    )

    train_losses, val_losses = [], []

    for _ in range(n_epochs):
        clf.partial_fit(X_train_tfidf, y_train, classes=classes, sample_weight=sample_weight)
        train_proba = clf.predict_proba(X_train_tfidf)
        val_proba = clf.predict_proba(X_val_tfidf)
        train_losses.append(float(_log_loss(y_train, train_proba)))
        val_losses.append(float(_log_loss(y_val, val_proba)))

    preds = clf.predict(X_val_tfidf)
    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "macro_f1": float(f1_score(y_val, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_val, preds, average="weighted", zero_division=0)),
        "n_features": int(X_train_tfidf.shape[1]),
    }
    return metrics, train_losses, val_losses


def _log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    eps = 1e-12
    p = proba[np.arange(len(y_true)), y_true]
    return -np.log(np.clip(p, eps, 1.0)).mean()


def train_pytorch(
    X_train: list[str],
    y_train: np.ndarray,
    X_val: list[str],
    y_val: np.ndarray,
    n_epochs: int = 40,
    lr: float = 0.001,
    embed_dim: int = 256,
    hidden_dim: int = 256,
    min_freq: int = 2,
    batch_size: int = 256,
    seed: int = 42,
) -> tuple[dict, list[float], list[float], dict]:
    torch.manual_seed(seed)
    n_classes = len(np.unique(y_train))
    ngrams = (1, 2)

    vocab = build_vocab(X_train, min_freq=min_freq, ngrams=ngrams)
    train_tokens = texts_to_bow(X_train, vocab, ngrams=ngrams)
    val_tokens = texts_to_bow(X_val, vocab, ngrams=ngrams)

    x_train, len_train = pad_batch(train_tokens)
    x_val, len_val = pad_batch(val_tokens)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    model = ShapeClassifier(
        len(vocab),
        n_classes,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=0.3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float))

    n = len(y_train)
    train_losses, val_losses = [], []

    for _ in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            bx, bl = x_train[idx], len_train[idx]
            by = y_train_t[idx]
            optimizer.zero_grad()
            logits = model(bx, bl)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_losses.append(epoch_loss / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val, len_val), y_val_t)
            val_losses.append(float(val_loss.item()))

    model.eval()
    with torch.no_grad():
        preds = model(x_val, len_val).argmax(dim=1).numpy()

    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "macro_f1": float(f1_score(y_val, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_val, preds, average="weighted", zero_division=0)),
        "vocab_size": len(vocab),
        "embed_dim": embed_dim,
    }
    pipeline_info = {
        "texte_brut": "comments (html.unescape, min 5 car.)",
        "tokenisation": "regex [a-zA-Z']+ → indices vocab (train only, min_freq=3)",
        "pytorch_entree": (
            f"Tokenisation unigrams+bigrams (min_freq={min_freq}) → "
            f"Embedding({len(vocab)}, {embed_dim}) → mean pool → "
            f"MLP({hidden_dim}) → logits({n_classes})"
        ),
    }
    return metrics, train_losses, val_losses, pipeline_info


def plot_curves(
    sklearn_train: list[float],
    sklearn_val: list[float],
    torch_train: list[float],
    torch_val: list[float],
    path: Path,
) -> None:
    epochs = range(1, len(torch_train) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, sklearn_train, label="Train", linewidth=2)
    axes[0].plot(epochs, sklearn_val, label="Val", linewidth=2)
    axes[0].set_title("Linéaire — TF-IDF + SGD")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Log-loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, torch_train, label="Train", linewidth=2)
    axes[1].plot(epochs, torch_val, label="Val", linewidth=2)
    axes[1].set_title("PyTorch — Embedding + mean pool")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Cross-entropy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Phase 3 — Courbes train / validation (même découpe)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_phase3(
    n_epochs: int = 40,
    seed: int = 42,
) -> dict:
    if not DATA_PATH.exists():
        from scripts.download_data import download

        download(DATA_PATH)

    df_raw = load_shape_data(DATA_PATH)
    df, decisions = prepare_shape_labels(df_raw)

    i_train, i_val = temporal_split(df, train_ratio=0.75)
    train_df = df.loc[i_train]
    val_df = df.loc[i_val]

    y_train_enc, y_val_enc, label_to_idx = encode_labels(
        train_df["shape"], val_df["shape"]
    )
    X_train = train_df["comments"].tolist()
    X_val = val_df["comments"].tolist()

    majority = majority_baseline(y_train_enc, y_val_enc)
    sklearn_metrics, sk_train, sk_val = train_sklearn_linear(
        X_train, y_train_enc, X_val, y_val_enc, n_epochs=n_epochs, seed=seed
    )
    torch_metrics, pt_train, pt_val, pipeline_info = train_pytorch(
        X_train, y_train_enc, X_val, y_val_enc, n_epochs=n_epochs, seed=seed
    )

    plot_curves(sk_train, sk_val, pt_train, pt_val, FIGURES_DIR / "phase3_curves.png")

    results = {
        "split": {
            "strategy": "temporal 75/25 sur datetime (observation)",
            "n_train": len(train_df),
            "n_val": len(val_df),
        },
        "decisions": decisions,
        "n_classes": len(label_to_idx),
        "classes": list(label_to_idx.keys()),
        "metrics": {
            "majority": majority,
            "sklearn_linear": sklearn_metrics,
            "pytorch": torch_metrics,
        },
        "pytorch_beats_sklearn": {
            "accuracy": torch_metrics["accuracy"] > sklearn_metrics["accuracy"],
            "macro_f1": torch_metrics["macro_f1"] > sklearn_metrics["macro_f1"],
        },
        "pipeline_text_to_number": pipeline_info,
        "hyperparams": {"n_epochs": n_epochs, "seed": seed},
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "phase3_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def main():
    results = run_phase3()
    m = results["metrics"]
    print("=== Phase 3 — PyTorch vs linéaire ===")
    print(f"Classes : {results['n_classes']} | Train : {results['split']['n_train']:,} | Val : {results['split']['n_val']:,}")
    print(f"Split : {results['split']['strategy']}")
    print()
    print(f"{'Modèle':<22} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>12}")
    print("-" * 56)
    for name, key in [
        ("Classe majoritaire", "majority"),
        ("Linéaire TF-IDF", "sklearn_linear"),
        ("PyTorch (nôtre)", "pytorch"),
    ]:
        row = m[key]
        print(
            f"{name:<22} {row['accuracy']:>10.3f} {row['macro_f1']:>10.3f} {row['weighted_f1']:>12.3f}"
        )
    print(f"\nFigure : reports/figures/phase3_curves.png")
    beats = results["pytorch_beats_sklearn"]
    print(f"PyTorch bat le linéaire (accuracy) : {beats['accuracy']}")
    print(f"PyTorch bat le linéaire (macro-F1) : {beats['macro_f1']}")


if __name__ == "__main__":
    main()
