from __future__ import annotations

import torch.nn as nn


# ── Heads ────────────────────────────────────────────────────────────────────
def _mlp_head(feat_dim: int, num_classes: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(feat_dim, feat_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(feat_dim // 2, num_classes),
    )


class Heads(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.cls_head = _mlp_head(feat_dim, num_classes, dropout)
        self.reg_head = _mlp_head(feat_dim, num_classes, dropout)

    def forward(self, pooled):
        return self.cls_head(pooled), self.reg_head(pooled)
