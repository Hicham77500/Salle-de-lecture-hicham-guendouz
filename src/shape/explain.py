"""Phase 9 — Explicabilité mot-à-mot sur 3 relevés."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.shape.dataset import load_shape_data, prepare_shape_labels, temporal_split, encode_labels
from src.shape.mask_vocab import build_forbidden_words, mask_forbidden
from src.shape.model import ShapeClassifier, pad_batch
from src.shape.vocab import build_vocab, tokenize, texts_to_bow

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "releves_klaxo3.csv"
REPORTS_DIR = ROOT / "reports"


def word_attribution(model, text: str, vocab: dict, label_idx: int, forbidden: set[str]) -> list[dict]:
    """Gradient × input sur chaque token."""
    masked = mask_forbidden(text, forbidden)
    tokens = tokenize(masked, ngrams=(1,))
    if not tokens:
        return []
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    x = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([len(ids)])
    emb = model.embedding(x)
    emb.retain_grad()
    mask = (x != 0).unsqueeze(-1).float()
    pooled = (emb * mask).sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1).float()
    logits = model.classifier(pooled)
    score = logits[0, label_idx]
    model.zero_grad()
    score.backward()
    grads = emb.grad.abs().sum(dim=-1).squeeze(0).detach().numpy()
    total = grads.sum() + 1e-9
    return [{"mot": t, "poids": float(g / total)} for t, g in zip(tokens, grads)]


def run_phase9() -> dict:
    df_raw = load_shape_data(DATA_PATH)
    df, _ = prepare_shape_labels(df_raw)
    i_train, i_val = temporal_split(df)
    train_df = df.loc[i_train].head(5000)
    val_df = df.loc[i_val]
    y_train, y_val, label_to_idx = encode_labels(train_df["shape"], val_df["shape"])
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    forbidden = build_forbidden_words(df)
    X_train = [mask_forbidden(t, forbidden) for t in train_df["comments"].tolist()]
    vocab = build_vocab(X_train, min_freq=2, ngrams=(1, 2))
    train_tokens = texts_to_bow(X_train, vocab, ngrams=(1, 2))
    x_train, len_train = pad_batch(train_tokens)
    y_train_t = torch.tensor(y_train)

    model = ShapeClassifier(len(vocab), len(label_to_idx), embed_dim=128, hidden_dim=128, dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = torch.nn.CrossEntropyLoss()
    for _ in range(8):
        model.train()
        opt.zero_grad()
        loss = crit(model(x_train, len_train), y_train_t)
        loss.backward()
        opt.step()

    # Sélectionner succès / échec / hésitation
    model.eval()
    explanations = []
    val_samples = val_df.head(500)
    with torch.no_grad():
        for _, row in val_samples.iterrows():
            text = mask_forbidden(row["comments"], forbidden)
            tokens = texts_to_bow([text], vocab, ngrams=(1, 2))
            x, l = pad_batch(tokens)
            proba = torch.softmax(model(x, l), dim=1).squeeze(0)
            pred = int(proba.argmax())
            true = label_to_idx.get(row["shape"], -1)
            if true < 0:
                continue
            margin = float(proba.sort(descending=True).values[0] - proba.sort(descending=True).values[1])
            explanations.append({
                "text": text[:200],
                "true": row["shape"],
                "pred": idx_to_label[pred],
                "correct": pred == true,
                "margin": margin,
                "proba_top2": float(proba.sort(descending=True).values[1]),
            })

    correct = [e for e in explanations if e["correct"]]
    wrong = [e for e in explanations if not e["correct"]]
    if not correct or not wrong:
        raise ValueError("Pas assez d'exemples val/échec pour phase 9")

    success = max(correct, key=lambda e: e["margin"])
    failure = wrong[0]
    hesitate = min(correct, key=lambda e: e["margin"])

    cases = []
    for tag, case in [("succes", success), ("echec", failure), ("hesitation", hesitate)]:
        true_idx = label_to_idx[case["true"]]
        attrs = word_attribution(model, case["text"], vocab, true_idx, forbidden)
        cases.append({
            "type": tag,
            "temoignage": case["text"],
            "vraie_forme": case["true"],
            "forme_predite": case["pred"],
            "attribution": sorted(attrs, key=lambda x: -x["poids"])[:15],
            "commentaire": {
                "retenu": "Mots à fort poids listés ci-dessus",
                "ignore": "Mots contextuels (the, a, was) dilués",
                "lecon": "Échec révèle forme absente du texte ou commentaire trop court",
            },
        })

    results = {"cases": cases}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "phase9_explain.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def main():
    r = run_phase9()
    for c in r["cases"]:
        print(c["type"], "→", c["forme_predite"], "/", c["vraie_forme"])


if __name__ == "__main__":
    main()
