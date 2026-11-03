from __future__ import annotations

from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import chain
from ..ablations.combinations import draw_condition
from ..ablations.common import Condition
from ..rng import derive_rng
from ..view_paths import resolve_view_dirs

# ── Normalisation ────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Training bag dataset ─────────────────────────────────────────────────────
class BagDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: dict,
        run_seed: int,
        sham: bool = False,
        enabled_codes: Optional[FrozenSet[str]] = None,
    ):
        # None = full protocol pool (always in training); a subset is for demos only
        self.enabled_codes = enabled_codes
        self.stems = df["stem"].to_numpy()
        self.views = df["view"].to_numpy()
        self.habitats = np.stack(df["habitat"].to_numpy())
        # One folder set per view; each item uses its own view's
        self.view_dirs = resolve_view_dirs(cfg)
        self.run_seed = run_seed
        self.sham = sham
        # Set by the training loop each epoch, so an index draws a new condition each pass
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.stems)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> Dict[str, object]:
        stem = self.stems[index]
        dirs = self.view_dirs[self.views[index]]
        # Crée un random generator
        rng = derive_rng(self.run_seed, self.epoch, index, stem)

        condition = Condition() if self.sham else draw_condition(rng, enabled_codes=self.enabled_codes)
        crops, meta = chain.build_bag(
            dirs["dataset_dir"], dirs["dataset_a1_dir"], dirs["dataset_a2_dir"], dirs["dataset_a7_dir"], stem, condition, rng,
            dataset_a6_class_dir=dirs["dataset_a6_class_dir"],
        )

        crops = crops.astype(np.float32) / 255.0
        crops = (crops - IMAGENET_MEAN) / IMAGENET_STD
        crops = torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous()  # (16, 3, 224, 224)

        habitat = torch.from_numpy(self.habitats[index].astype(np.float32))
        support = (habitat > 0).float()

        return {
            "crops": crops,
            "habitat": habitat,
            "support": support,
            "stem": stem,
            "condition": str(condition),
        }
