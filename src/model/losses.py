from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Weights ──────────────────────────────────────────────────────────────────
LAMBDA_KL = 0.5
LAMBDA_BCE = 0.5


# ── Combined loss ────────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, lambda_kl: float = LAMBDA_KL, lambda_bce: float = LAMBDA_BCE):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.lambda_kl = lambda_kl
        self.lambda_bce = lambda_bce

    def set_weights(self, lambda_kl: float, lambda_bce: float) -> None:
        self.lambda_kl = lambda_kl
        self.lambda_bce = lambda_bce

    def forward(self, cls_logits, reg_logits, support, habitat):
        log_pred = F.log_softmax(reg_logits, dim=-1)
        # KL(true || pred), keeping true's entropy term so values compare across runs
        safe_true = habitat.clamp_min(1e-12)
        kl = (habitat * (safe_true.log() - log_pred)).sum(dim=-1).mean()

        bce_per_class = F.binary_cross_entropy_with_logits(cls_logits, support, reduction="none")
        bce = (bce_per_class * self.class_weights).mean()

        return self.lambda_kl * kl + self.lambda_bce * bce, {"kl": kl.item(), "bce": bce.item()}
