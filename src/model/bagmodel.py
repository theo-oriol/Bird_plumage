from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import build_backbone
from .heads import Heads
from .mil import GatedAttentionPool


# ── Bag model ────────────────────────────────────────────────────────────────
class BagModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.backbone = build_backbone(cfg)
        feat_dim = cfg["feat_dim"]
        self.pool = GatedAttentionPool(feat_dim)
        self.heads = Heads(feat_dim, cfg["num_habitats"], dropout=cfg["head_dropout"])

    # crops (B, N, 3, 224, 224) -> cls_logits, reg_logits, attention
    def forward(self, crops: torch.Tensor):
        b, n = crops.shape[:2]
        flat = crops.reshape(b * n, *crops.shape[2:])
        features = self.backbone(flat).reshape(b, n, -1)
        pooled, attention = self.pool(features)
        cls_logits, reg_logits = self.heads(pooled)
        return cls_logits, reg_logits, attention

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
