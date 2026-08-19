"""Modèle PyTorch — comments → shape."""

from __future__ import annotations

import torch
import torch.nn as nn


class ShapeClassifier(nn.Module):
    """
    Embedding → mean pooling → (optionnel) MLP → Linear.

    Architecture minimale réutilisable en phase 2 (hidden_dim=None).
    Phase 3 utilise hidden_dim pour battre le baseline linéaire.
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        embed_dim: int = 64,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if hidden_dim:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        mask = (token_ids != 0).unsqueeze(-1).float()
        summed = (embedded * mask).sum(dim=1)
        lengths = lengths.clamp(min=1).unsqueeze(-1).float()
        pooled = summed / lengths
        return self.classifier(pooled)


def pad_batch(token_lists: list[list[int]], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in token_lists) if token_lists else 1
    batch = []
    lengths = []
    for seq in token_lists:
        padded = seq + [pad_id] * (max_len - len(seq))
        batch.append(padded)
        lengths.append(len(seq))
    return (
        torch.tensor(batch, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )
