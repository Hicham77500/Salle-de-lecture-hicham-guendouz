"""Phase 10-13 — Attention manuelle."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shape.dataset import load_shape_data
from src.shape.vocab import tokenize

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "releves_klaxo3.csv"
FIGURES_DIR = ROOT / "reports/figures"
REPORTS_DIR = ROOT / "reports"


class SingleHeadAttentionManual(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)
        weights = F.softmax(scores, dim=-1)
        return weights @ V, weights


def sinusoidal_pos_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
    return pe


class MultiHeadManual(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 2):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttentionManual(d_model) for _ in range(n_heads)])
        self.proj = nn.Linear(d_model * n_heads, d_model)

    def forward(self, x):
        outs, weights = [], []
        for h in self.heads:
            o, w = h(x)
            outs.append(o)
            weights.append(w)
        return self.proj(torch.cat(outs, dim=-1)), weights


def find_pronoun_sentence(df) -> tuple[list[str], str]:
    for _, row in df.iterrows():
        text = str(row["comments"]).lower()
        if any(p in text for p in (" it ", " she ", " he ", " they ", " her ", " him ")):
            tokens = tokenize(text, ngrams=(1,))
            if len(tokens) >= 6:
                return tokens[:20], text[:120]
    tokens = tokenize(df.iloc[0]["comments"], ngrams=(1,))
    return tokens[:15], df.iloc[0]["comments"][:120]


def run_attention_phases() -> dict:
    df = load_shape_data(DATA_PATH)
    tokens, snippet = find_pronoun_sentence(df)
    d_model = 32
    seq_len = len(tokens)
    x = torch.randn(1, seq_len, d_model)

    attn = SingleHeadAttentionManual(d_model)
    out, weights = attn(x)
    row_sums = weights.sum(dim=-1).squeeze(0).tolist()

    # Phase 11 — permutation
    perm = torch.randperm(seq_len)
    x_perm = x[:, perm, :]
    _, w_perm = attn(x_perm)
    out_perm_unsorted, _ = attn(x_perm)
    diff_before = float((out - out_perm_unsorted[:, perm, :]).abs().mean())

    pe = sinusoidal_pos_encoding(seq_len, d_model)
    x_pos = x + pe.unsqueeze(0)
    x_perm_pos = x_pos[:, perm, :]
    out_pos, _ = attn(x_pos)
    out_perm_pos, _ = attn(x_perm_pos)
    diff_after = float((out_pos - out_perm_pos).abs().mean())

    # Phase 12 — benchmark O(n²)
    bench = []
    attn_b = SingleHeadAttentionManual(64)
    for L in [32, 64, 128, 256, 512]:
        xx = torch.randn(1, L, 64)
        t0 = time.perf_counter()
        for _ in range(5):
            attn_b(xx)
        elapsed = (time.perf_counter() - t0) / 5
        bench.append({"tokens": L, "seconds": elapsed, "matrix_cells": L * L})

    # Phase 13 — multi-head
    mh = MultiHeadManual(d_model, n_heads=2)
    _, w_heads = mh(x_pos)
    disagree = float((w_heads[0] - w_heads[1]).abs().mean())
    mh_identical = MultiHeadManual(d_model, n_heads=2)
    mh_identical.heads[1].load_state_dict(mh_identical.heads[0].state_dict())
    _, w_ident = mh_identical(x_pos)
    disagree_control = float((w_ident[0] - w_ident[1]).abs().mean())

    # Figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    w_np = weights.squeeze(0).detach().numpy()
    im = ax.imshow(w_np, cmap="Blues")
    ax.set_xticks(range(len(tokens)), tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tokens)), tokens, fontsize=8)
    ax.set_title("Phase 10 — Matrice attention")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase10_attention.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(w_heads[0].squeeze(0).detach().numpy(), cmap="Blues")
    axes[0].set_title("Tête 1")
    axes[1].imshow(w_heads[1].squeeze(0).detach().numpy(), cmap="Blues")
    axes[1].set_title("Tête 2")
    fig.suptitle(f"Phase 13 — Désaccord={disagree:.4f}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase13_multihead.png", dpi=150)
    plt.close(fig)

    lengths = [b["tokens"] for b in bench]
    times = [b["seconds"] for b in bench]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lengths, times, "o-")
    ax.set_xlabel("Jetons")
    ax.set_ylabel("Temps (s)")
    ax.set_title("Phase 12 — O(n²)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase12_benchmark.png", dpi=150)
    plt.close(fig)

    factor = bench[2]["seconds"] / max(bench[1]["seconds"], 1e-9)  # 128 vs 64 approx
    results = {
        "phase10": {
            "snippet": snippet,
            "tokens": tokens,
            "row_sums": row_sums,
            "output_shape": list(out.shape),
        },
        "phase11": {
            "diff_before_pos": diff_before,
            "diff_after_pos": diff_after,
        },
        "phase12": {"benchmark": bench, "doubling_factor_128_64": factor},
        "phase13": {"disagree": disagree, "disagree_identical_heads": disagree_control},
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "phase10_13_attention.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def main():
    r = run_attention_phases()
    print("Phase 10 tokens:", r["phase10"]["tokens"][:8])
    print("Phase 11 diff after pos:", r["phase11"]["diff_after_pos"])
    print("Phase 13 disagree:", r["phase13"]["disagree"])


if __name__ == "__main__":
    main()
