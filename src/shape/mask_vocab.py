"""Phase 8 — Masque vocabulaire des formes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.shape.dataset import load_shape_data, prepare_shape_labels, temporal_split, encode_labels
from src.shape.train import train_pytorch, train_sklearn_linear
from src.shape.vocab import TOKEN_RE

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "releves_klaxo3.csv"
REPORTS_DIR = ROOT / "reports"

PLURALS = {
    "lights": "light", "circles": "circle", "disks": "disk", "disks": "disk",
    "triangles": "triangle", "rectangles": "rectangle", "cigars": "cigar",
    "fireballs": "fireball", "ovals": "oval", "chevrons": "chevron",
}


def build_forbidden_words(df: pd.DataFrame) -> set[str]:
    shapes = set(df["shape"].unique()) - {"", "rare", "other_merged"}
    forbidden = set()
    for s in shapes:
        forbidden.add(s.lower())
        forbidden.add(s.lower() + "s")
        if s in PLURALS:
            forbidden.add(PLURALS[s])
        # variantes courantes
        forbidden.add(s.replace("-", " "))
    # mots composés fréquents
    extras = {"ufo", "object", "objects", "saucer", "saucers"}
    forbidden |= extras
    return forbidden


def mask_forbidden(text: str, forbidden: set[str]) -> str:
    def repl(m):
        w = m.group(0).lower()
        return "" if w in forbidden else m.group(0)

    return TOKEN_RE.sub(repl, text)


def count_remaining_forbidden(texts: list[str], forbidden: set[str]) -> int:
    count = 0
    for text in texts:
        words = TOKEN_RE.findall(text.lower())
        if any(w in forbidden for w in words):
            count += 1
    return count


def run_phase8(n_epochs: int = 25) -> dict:
    df_raw = load_shape_data(DATA_PATH)
    df, decisions = prepare_shape_labels(df_raw)
    i_train, i_val = temporal_split(df)
    train_df = df.loc[i_train]
    val_df = df.loc[i_val]
    y_train, y_val, label_to_idx = encode_labels(train_df["shape"], val_df["shape"])

    forbidden = build_forbidden_words(df)
    X_train_raw = train_df["comments"].tolist()
    X_val_raw = val_df["comments"].tolist()

    before_sk = train_sklearn_linear(X_train_raw, y_train, X_val_raw, y_val, n_epochs=n_epochs)[0]
    before_pt, _, _, _ = train_pytorch(X_train_raw, y_train, X_val_raw, y_val, n_epochs=n_epochs)

    X_train_masked = [mask_forbidden(t, forbidden) for t in X_train_raw]
    X_val_masked = [mask_forbidden(t, forbidden) for t in X_val_raw]
    remaining = count_remaining_forbidden(X_train_masked + X_val_masked, forbidden)

    after_sk = train_sklearn_linear(X_train_masked, y_train, X_val_masked, y_val, n_epochs=n_epochs)[0]
    after_pt, _, _, _ = train_pytorch(X_train_masked, y_train, X_val_masked, y_val, n_epochs=n_epochs)

    per_class_before = {}
    per_class_after = {}
    for label in label_to_idx:
        idx = label_to_idx[label]
        mask = y_val == idx
        if mask.sum() == 0:
            continue
        per_class_before[label] = int(mask.sum())

    results = {
        "n_forbidden_words": len(forbidden),
        "forbidden_sample": sorted(list(forbidden))[:30],
        "remaining_with_forbidden": remaining,
        "before": {"sklearn": before_sk, "pytorch": before_pt},
        "after": {"sklearn": after_sk, "pytorch": after_pt},
        "drop": {
            "sklearn_acc": before_sk["accuracy"] - after_sk["accuracy"],
            "pytorch_macro_f1": before_pt["macro_f1"] - after_pt["macro_f1"],
        },
        "decisions": decisions,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "phase8_mask.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def main():
    r = run_phase8()
    print(f"Mots interdits: {r['n_forbidden_words']} | Restants: {r['remaining_with_forbidden']}")
    print(f"Avant acc: {r['before']['sklearn']['accuracy']:.3f} → Après: {r['after']['sklearn']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
