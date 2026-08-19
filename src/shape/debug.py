"""Phase 4 — Carnet de 3 pannes volontaires."""

from __future__ import annotations

import json
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


def _mini_data(seed: int = 0, n_train: int = 800, n_val: int = 200):
    df_raw = load_shape_data(DATA_PATH)
    df, _ = prepare_shape_labels(df_raw)
    i_train, i_val = temporal_split(df, train_ratio=0.75)
    train_df = df.loc[i_train].head(n_train)
    val_df = df.loc[i_val]
    val_df = val_df[val_df["shape"].isin(set(train_df["shape"]))].head(n_val)
    y_train, y_val, _ = encode_labels(train_df["shape"], val_df["shape"])
    X_train = train_df["comments"].tolist()
    X_val = val_df["comments"].tolist()
    vocab = build_vocab(X_train, min_freq=2, ngrams=(1, 2))
    train_tokens = texts_to_bow(X_train, vocab, ngrams=(1, 2))
    val_tokens = texts_to_bow(X_val, vocab, ngrams=(1, 2))
    x_train, len_train = pad_batch(train_tokens)
    x_val, len_val = pad_batch(val_tokens)
    return x_train, len_train, torch.tensor(y_train), x_val, len_val, torch.tensor(y_val), len(vocab), len(np.unique(y_train))


def _train_loop(model, x_train, len_train, y_train, x_val, len_val, y_val, lr=0.001, epochs=15, eval_train_as_val=False):
    """Retourne train_losses, val_losses, val_acc."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    n = len(y_train)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        eloss = 0.0
        nb = 0
        for start in range(0, n, 64):
            idx = perm[start : start + 64]
            opt.zero_grad()
            logits = model(x_train[idx], len_train[idx])
            loss = crit(logits, y_train[idx])
            loss.backward()
            opt.step()
            eloss += float(loss.item())
            nb += 1
        train_losses.append(eloss / max(nb, 1))
        model.eval()
        with torch.no_grad():
            xv, yv = (x_train, y_train) if eval_train_as_val else (x_val, y_val)
            lv = len_train if eval_train_as_val else len_val
            val_losses.append(float(crit(model(xv, lv), yv).item()))
    with torch.no_grad():
        preds = model(x_val, len_val).argmax(1)
        acc = float((preds == y_val).float().mean())
    return train_losses, val_losses, acc


def panne1_overfit_eval_mode(seed: int = 42) -> dict:
    """Panne 1 : excellent en train, bête en eval — model.train() pendant l'évaluation."""
    torch.manual_seed(seed)
    x_train, len_train, y_train, x_val, len_val, y_val, vocab_size, n_classes = _mini_data()
    model = ShapeClassifier(vocab_size, n_classes, embed_dim=64, hidden_dim=128, dropout=0.5)
    train_losses, val_losses, _ = _train_loop(model, x_train, len_train, y_train, x_val, len_val, y_val, epochs=20)
    # Évaluer en mode train (dropout actif) — simule mauvaise pratique
    model.train()
    with torch.no_grad():
        preds = model(x_val, len_val).argmax(1)
        acc_broken = float((preds == y_val).float().mean())
    model.eval()
    with torch.no_grad():
        preds_ok = model(x_val, len_val).argmax(1)
        acc_fixed = float((preds_ok == y_val).float().mean())
    return {
        "nom": "Overfitting / mauvais mode eval",
        "geste": "Laisser model.train() pendant l'évaluation (dropout actif)",
        "signature": "Train loss basse, val loss haute, prédictions instables",
        "train_losses": train_losses,
        "val_losses": val_losses,
        "acc_broken": acc_broken,
        "acc_fixed": acc_fixed,
        "test_1min": "Vérifier model.eval() avant predict — acc saute",
    }


def panne2_lr_trop_haut(seed: int = 42) -> dict:
    """Panne 2 : loss descend mais prédictions pires que hasard."""
    torch.manual_seed(seed)
    x_train, len_train, y_train, x_val, len_val, y_val, vocab_size, n_classes = _mini_data()
    model = ShapeClassifier(vocab_size, n_classes, embed_dim=64, hidden_dim=128)
    train_losses, val_losses, acc = _train_loop(
        model, x_train, len_train, y_train, x_val, len_val, y_val, lr=0.5, epochs=15
    )
    random_acc = 1.0 / n_classes
    return {
        "nom": "LR trop haut",
        "geste": "Adam lr=0.5 au lieu de 0.001",
        "signature": "Loss oscille/descend, accuracy << hasard (1/n_classes)",
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_acc": acc,
        "random_baseline": random_acc,
        "test_1min": "Réduire lr — courbe se stabilise, acc > hasard",
    }


def panne3_loss_figee(seed: int = 42) -> dict:
    """Panne 3 : loss figée — gradients non remis à zéro."""
    torch.manual_seed(seed)
    x_train, len_train, y_train, x_val, len_val, y_val, vocab_size, n_classes = _mini_data()
    model = ShapeClassifier(vocab_size, n_classes, embed_dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    losses = []
    n = len(y_train)
    for _ in range(15):
        model.train()
        perm = torch.randperm(n)
        eloss = 0.0
        nb = 0
        for start in range(0, n, 64):
            idx = perm[start : start + 64]
            # BUG : pas de zero_grad()
            logits = model(x_train[idx], len_train[idx])
            loss = crit(logits, y_train[idx])
            loss.backward()
            opt.step()
            eloss += float(loss.item())
            nb += 1
        losses.append(eloss / max(nb, 1))
    return {
        "nom": "Loss figée",
        "geste": "Oublier optimizer.zero_grad() entre les batches",
        "signature": "Loss plate ou diverge, pas de descente propre",
        "train_losses": losses,
        "test_1min": "Ajouter zero_grad() — loss redescend",
    }


def plot_pannes(results: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    titles = ["Panne 1 — eval en train()", "Panne 2 — LR=0.5", "Panne 3 — sans zero_grad"]
    for ax, res, title in zip(axes, results, titles):
        tl = res.get("train_losses", res.get("train_losses", []))
        vl = res.get("val_losses", [])
        if vl:
            ax.plot(tl, label="Train")
            ax.plot(vl, label="Val")
        else:
            ax.plot(tl, label="Train (figée)")
        ax.set_title(title)
        ax.set_xlabel("Époque")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Phase 4 — Carnet de pannes")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_phase4() -> dict:
    r1 = panne1_overfit_eval_mode()
    r2 = panne2_lr_trop_haut()
    r3 = panne3_loss_figee()
    results = {"pannes": [r1, r2, r3]}
    for p in results["pannes"]:
        for k in list(p.keys()):
            if k.endswith("_losses"):
                p[k] = [float(x) for x in p[k]]
    plot_pannes(results["pannes"], FIGURES_DIR / "phase4_pannes.png")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "phase4_pannes.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def main():
    r = run_phase4()
    for i, p in enumerate(r["pannes"], 1):
        print(f"Panne {i}: {p['nom']} — {p['geste']}")


if __name__ == "__main__":
    main()
