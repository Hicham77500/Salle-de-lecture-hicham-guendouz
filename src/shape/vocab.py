"""Tokenisation partagée pour les phases 2-3."""

from __future__ import annotations

import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-zA-Z']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str], min_freq: int = 1) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, count in counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def texts_to_bow(texts: list[str], vocab: dict[str, int]) -> list[list[int]]:
    """Retourne les indices de tokens par texte."""
    return [[vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)] for text in texts]
