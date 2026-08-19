"""
Attention single-head — Phase 10.

Interdit : nn.MultiheadAttention, bibliothèques transformers pour cette phase.
Autorisé : torch.Tensor, produits matriciels, softmax, nn.Linear.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    """Une tête d'attention implémentée from scratch."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, d_model)

        Returns
        -------
        output, attention_weights : weights shape (batch, seq_len, seq_len)
        """
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)
        weights = F.softmax(scores, dim=-1)
        output = weights @ V
        return output, weights
