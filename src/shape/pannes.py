"""Phase 4 — Carnet de pannes : trois pannes volontaires sur le montage phase 3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.shape.dataset import encode_labels, load_shape_data, prepare_shape_labels, temporal_split
from src.shape.model import ShapeClassifier, pad_batch
from src.shape.vocab import build_vocab, texts_to_bow

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "releves_klaxo3.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"


@dataclass
class PanneConfig:
    name: str
    slug: str
    description: str
    eval_train_mode: bool = False
    swap_labels: bool = False
    lr: float = 0.001
    zero_grad: bool = True
    n_epochs: int = 25


def _prepare_data(max_train: int = 8000, max_val: int = 3000):
    df_raw = load_shape_data(DATA_PATH)
    df, _ = prepare_shape_labels(df_raw)
    i_train, i_val = temporal_split(df)
    train_df = df.loc[i_train].head(max_train)
    val_df = df.loc[i_val].head(max_val)
    y_train, y_val, label_to_idx = encode_labels(train_df["shape"], val_df["shape"])
    return (
        train_df["comments"].tolist(),
        val_df["comments"].tolist(),
        y_train,
        y_val,
        len(label_to_idx),
    )


def _train_loop(config: PanneConfig, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    X_train, X_val, y_train, y_val, n_classes = _prepare_data()

    if config.swap_labels:
        y_train = (y_train + 1) % n_classes

    ngrams = (1, 2)
    vocab = build_vocab(X_train, min_freq=2, ngrams=ngrams)
    train_tokens = texts_to_bow(X_train, vocab, ngrams=ngrams)
    val_tokens = texts_to_bow(X_val, vocab, ngrams=ngrams)
    x_train, len_train = pad_batch(train_tokens)
    x_val, len_val = pad_batch(val_tokens)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    model = ShapeClassifier(
        len(vocab), n_classes, embed_dim=128, hidden_dim=128, dropout=0.3
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float))

    batch_size = 256
    n = len(y_train)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for _ in range(config.n_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        all_preds, all_true = [], []
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            bx, bl = x_train[idx], len_train[idx]
            by = y_train_t[idx]
            if config.zero_grad:
                optimizer.zero_grad()
            logits = model(bx, bl)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
            all_preds.extend(logits.argmax(dim=1).detach().numpy())
            all_true.extend(by.numpy())
        train_losses.append(epoch_loss / max(n_batches, 1))
        train_accs.append(float(accuracy_score(all_true, all_preds)))

        if config.eval_train_mode:
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            val_logits = model(x_val, len_val)
            val_loss = criterion(val_logits, y_val_t)
            val_preds = val_logits.argmax(dim=1).numpy()
        val_losses.append(float(val_loss.item()))
        val_accs.append(float(accuracy_score(y_val, val_preds)))

    majority = float(np.bincount(y_val).max() / len(y_val))
    random_guess = 1.0 / n_classes

    return {
        "name": config.name,
        "slug": config.slug,
        "description": config.description,
        "geste": _geste(config),
        "test_rapide": _test_rapide(config),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "final_train_acc": train_accs[-1],
        "final_val_acc": val_accs[-1],
        "majority_baseline": majority,
        "random_baseline": random_guess,
        "val_worse_than_random": val_accs[-1] < random_guess * 1.2,
        "loss_plateau": _is_plateau(train_losses),
        "model_training_at_eval": config.eval_train_mode,
    }


def _geste(config: PanneConfig) -> str:
    if config.eval_train_mode:
        return "Laisser model.train() actif pendant l'évaluation (dropout + BatchNorm en mode entraînement)."
    if config.swap_labels:
        return "Permuter les étiquettes : y_train = (y_train + 1) % n_classes (décalage systématique)."
    if config.lr == 0.0:
        return "Fixer le learning rate à 0.0 — optimizer.step() ne modifie plus les poids."
    if not config.zero_grad:
        return "Oublier optimizer.zero_grad() — les gradients s'accumulent entre batches."
    return "Configuration saine."


def _test_rapide(config: PanneConfig) -> str:
    if config.eval_train_mode:
        return "Vérifier model.training : True pendant val → panne 1 (overfitting apparent)."
    if config.swap_labels:
        return "Val accuracy < 1/n_classes alors que train loss baisse → panne 2."
    if config.lr == 0.0 or not config.zero_grad:
        return "Écart-type des 5 dernières losses < 0.001 → panne 3 (perte figée)."
    return "Courbes train/val cohérentes, val acc > hasard."


def _is_plateau(losses: list[float], window: int = 5, tol: float = 0.002) -> bool:
    if len(losses) < window:
        return False
    tail = losses[-window:]
    return float(np.std(tail)) < tol


def _plot_panne(result: dict, path: Path) -> None:
    epochs = range(1, len(result["train_losses"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, result["train_losses"], label="Train loss", linewidth=2)
    axes[0].plot(epochs, result["val_losses"], label="Val loss", linewidth=2)
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].set_title(f"Panne — {result['name']}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, result["train_accs"], label="Train acc", linewidth=2)
    axes[1].plot(epochs, result["val_accs"], label="Val acc", linewidth=2)
    axes[1].axhline(
        result["random_baseline"],
        color="red",
        linestyle="--",
        label=f"Hasard ({result['random_baseline']:.3f})",
    )
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(result["description"], fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_phase4(seed: int = 42) -> dict:
    if not DATA_PATH.exists():
        from scripts.download_data import download

        download(DATA_PATH)

    configs = [
        PanneConfig(
            name="Overfitting apparent",
            slug="overfit_eval",
            description="Bon en train, bête en val — sans changer les données",
            eval_train_mode=True,
        ),
        PanneConfig(
            name="Labels permutés",
            slug="label_swap",
            description="Loss descend, prédictions pires que le hasard",
            swap_labels=True,
        ),
        PanneConfig(
            name="Perte figée",
            slug="loss_frozen",
            description="La loss n'en bouge plus",
            lr=0.0,
        ),
    ]

    results = []
    for cfg in configs:
        print(f"=== Panne : {cfg.name} ===")
        r = _train_loop(cfg, seed=seed)
        fig_path = FIGURES_DIR / f"phase4_panne_{cfg.slug}.png"
        _plot_panne(r, fig_path)
        r["figure"] = str(fig_path.relative_to(ROOT))
        results.append(r)
        print(f"  Train acc: {r['final_train_acc']:.3f} | Val acc: {r['final_val_acc']:.3f}")
        print(f"  Test rapide : {r['test_rapide']}")

    # Référence saine (sans panne)
    print("=== Référence saine ===")
    healthy = _train_loop(PanneConfig(name="Sain", slug="healthy", description="Montage phase 3"), seed=seed)
    _plot_panne(healthy, FIGURES_DIR / "phase4_panne_healthy.png")
    healthy["figure"] = "reports/figures/phase4_panne_healthy.png"

    output = {
        "pannes": results,
        "reference_saine": {
            "final_train_acc": healthy["final_train_acc"],
            "final_val_acc": healthy["final_val_acc"],
            "figure": healthy["figure"],
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "phase4_pannes.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output


def main():
    run_phase4()
    print("\nCarnet : reports/phase4_pannes.json")
    print("Figures : reports/figures/phase4_panne_*.png")


if __name__ == "__main__":
    main()
