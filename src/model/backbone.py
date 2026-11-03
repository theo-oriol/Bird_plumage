from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


# ── Backbone ─────────────────────────────────────────────────────────────────
def build_backbone(cfg: dict) -> nn.Module:
    backbone_name = cfg["backbone"]["name"]
    weights_path = cfg["dinov3_weights"][backbone_name]

    backbone = torch.hub.load(
        cfg["dinov3_repo"],
        backbone_name,
        source="local",
        weights=weights_path,
    )

    for param in backbone.parameters():
        param.requires_grad = False

    lora_cfg = cfg["backbone"]["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    backbone = get_peft_model(backbone, peft_config)
    return backbone
