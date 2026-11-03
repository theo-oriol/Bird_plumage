from __future__ import annotations

import torch
import torch.nn as nn


# ── Gated attention pooling ──────────────────────────────────────────────────
# Ilse, Tomczak & Welling, ICML 2018
class GatedAttentionPool(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or feat_dim // 4
        self.V = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.Sigmoid())
        self.w = nn.Linear(hidden_dim, 1)

    # crop_features (B, N, d) -> pooled (B, d), attention (B, N)
    def forward(self, crop_features: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        gated = self.V(crop_features) * self.U(crop_features)
        scores = self.w(gated).squeeze(-1)  # (B, N)
        attention = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bn,bnd->bd", attention, crop_features)
        return pooled, attention
