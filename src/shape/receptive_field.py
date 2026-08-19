"""Phase 6-7 — Champ de vision (réceptive field) et batch size = 4."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.shape.dataset import encode_labels, load_shape_data, prepare_shape_labels, temporal_split
from src.shape.model import pad_batch
from src.shape.vocab import build_vocab, texts_to_bow

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "releves_klaxo3.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"


class ConvShapeClassifier(nn.Module):
    """CNN 1D — toutes positions traitées en parallèle (phase 6)."""

    def __init__(self, vocab_size: int, n_classes: int, embed_dim: int = 64, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Linear(128, n_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids).transpose(1, 2)  # (B, E, L)
        x = torch.relu(self.conv1(x))
        x = self.bn(torch.relu(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.head(x)


def receptive_field_table() -> list[dict]:
    """Tableau couche par couche — kernel 3, padding 1."""
    layers = [
        ("Embedding", 1, 1),
        ("Conv1d k=3", 3, 3),
        ("Conv1d k=3", 3, 5),
    ]
    rows = []
    cumul = 0
    for name, extent, total in layers:
        rows.append({"couche": name, "etendue": extent, "cumul": total})
        cumul = total
    return rows


def perturbation_test(model, x_sample, len_sample) -> float:
    """Modifie le premier token et mesure l'écart de sortie."""
    model.eval()
    with torch.no_grad():
        out1 = model(x_sample, len_sample)
        x2 = x_sample.clone()
        x2[0, 0] = (x2[0, 0] + 1) % 100 + 2
        out2 = model(x2, len_sample)
        return float((out1 - out2).abs().mean())


def train_model(model, x_train, len_train, y_train, x_val, len_val, y_val, batch_size: int = 256, epochs: int = 8):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    n = len(y_train)
    train_losses, val_losses = [], []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        eloss, nb = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss = crit(model(x_train[idx], len_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            eloss += float(loss.item())
            nb += 1
        train_losses.append(eloss / max(nb, 1))
        model.eval()
        with torch.no_grad():
            val_losses.append(float(crit(model(x_val, len_val), y_val).item()))
    with torch.no_grad():
        acc = float((model(x_val, len_val).argmax(1) == y_val).float().mean())
    return train_losses, val_losses, acc


def run_phases_6_7() -> dict:
    df_raw = load_shape_data(DATA_PATH)
    df, _ = prepare_shape_labels(df_raw)
    i_train, i_val = temporal_split(df)
    train_df = df.loc[i_train].head(8000)
    val_df = df.loc[i_val].head(2000)
    y_train, y_val, _ = encode_labels(train_df["shape"], val_df["shape"])
    X_train = train_df["comments"].tolist()
    X_val = val_df["comments"].tolist()
    vocab = build_vocab(X_train, min_freq=2, ngrams=(1, 2))
    train_tokens = texts_to_bow(X_train, vocab, ngrams=(1, 2))
    val_tokens = texts_to_bow(X_val, vocab, ngrams=(1, 2))
    lens = [len(t) for t in train_tokens]
    max_len = max(lens)
    median_len = int(np.median(lens))
    x_train, len_train = pad_batch(train_tokens)
    x_val, len_val = pad_batch(val_tokens)
    y_train_t = torch.tensor(y_train)
    y_val_t = torch.tensor(y_val)
    n_classes = len(np.unique(y_train))

    rf_table = receptive_field_table()
    cumul_rf = rf_table[-1]["cumul"]

    model = ConvShapeClassifier(len(vocab), n_classes)
    perturb = perturbation_test(model, x_val[:1], len_val[:1])
    tl256, vl256, acc256 = train_model(model, x_train, len_train, y_train_t, x_val, len_val, y_val_t, batch_size=256)

    # Phase 7 : batch=4 sans BatchNorm fix → mauvais
    model_b4_bad = ConvShapeClassifier(len(vocab), n_classes)
    tl4b, vl4b, acc4b = train_model(model_b4_bad, x_train, len_train, y_train_t, x_val, len_val, y_val_t, batch_size=4)

    # Fix : remplacer BatchNorm par LayerNorm (indépendant du batch)
    class ConvShapeFixed(nn.Module):
        def __init__(self, vocab_size, n_classes, embed_dim=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.conv1 = nn.Conv1d(embed_dim, 128, 3, padding=1)
            self.conv2 = nn.Conv1d(128, 128, 3, padding=1)
            self.ln = nn.LayerNorm(128)
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.head = nn.Linear(128, n_classes)

        def forward(self, token_ids, lengths):
            x = self.embedding(token_ids).transpose(1, 2)
            x = torch.relu(self.conv1(x))
            x = x.transpose(1, 2)
            x = self.ln(x).transpose(1, 2)
            x = torch.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.head(x)

    model_b4_fix = ConvShapeFixed(len(vocab), n_classes)
    tl4f, vl4f, acc4f = train_model(model_b4_fix, x_train, len_train, y_train_t, x_val, len_val, y_val_t, batch_size=4)
    tl256f, vl256f, acc256f = train_model(model_b4_fix, x_train, len_train, y_train_t, x_val, len_val, y_val_t, batch_size=256)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(vl4b, label=f"Batch=4 avant fix (acc={acc4b:.3f})")
    ax.plot(vl4f, label=f"Batch=4 après LayerNorm (acc={acc4f:.3f})")
    ax.plot(vl256, label=f"Batch=256 ref (acc={acc256:.3f})")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Val loss")
    ax.set_title("Phase 7 — Batch size = 4")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "phase7_batch4.png", dpi=150)
    plt.close(fig)

    results = {
        "phase6": {
            "max_tokens": max_len,
            "median_tokens": median_len,
            "receptive_field_table": rf_table,
            "cumul_rf": cumul_rf,
            "covers_max": cumul_rf >= max_len,
            "perturbation_delta": perturb,
            "val_acc_cnn": acc256,
        },
        "phase7": {
            "batch4_before_acc": acc4b,
            "batch4_after_acc": acc4f,
            "batch256_after_fix_acc": acc256f,
            "fix": "BatchNorm → LayerNorm (stats indépendantes du lot)",
            "inter_batch_dependency": "BatchNorm calcule μ/σ sur le lot — avec batch=4, stats bruitées ; prédiction single-sample impossible en eval sans running stats",
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "phase6_7_receptive.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def main():
    r = run_phases_6_7()
    print("Phase 6 RF cumul:", r["phase6"]["cumul_rf"], "| covers max:", r["phase6"]["covers_max"])
    print("Phase 7 batch4 acc:", r["phase7"]["batch4_after_acc"])


if __name__ == "__main__":
    main()
