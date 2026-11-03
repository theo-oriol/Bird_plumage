#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src import chain, npy_io
from src.env import expand_config
from src.ablations.common import Condition
from src.datasets.bag_dataset import IMAGENET_MEAN, IMAGENET_STD
from src.rng import derive_rng
from src.training.seeding import worker_init_fn

BACKBONE_NAME = "dinov3_vits16"
N_CROPS = 16  # 8 x 128 px (resized to 224) + 8 x 224 px, in build_bag order
VIEWS = ("Back", "Belly", "Side")
VIEW_TEMPLATE = "NEW_Segmented-Aves-{view}-NPY"


# ── Helpers ──────────────────────────────────────────────────────────────────
def _check_back_leaf_and_get_parent(path: Path, config_key: str) -> Path:
    """Parent of a 'NEW_Segmented-Aves-Back-NPY' folder (where the Belly and
    Side folders are). Raises if `path` is not such a folder, so nothing is
    built in the wrong place."""
    expected = path.parent / VIEW_TEMPLATE.format(view="Back")
    if path != expected:
        raise ValueError(
            f"configs/paths.yaml {config_key} ({path}) isn't a "
            f"'{VIEW_TEMPLATE.format(view='Back')}' folder directly under its parent - "
            f"can't derive the Belly/Side sibling folders from it."
        )
    return path.parent


# ── Crops ────────────────────────────────────────────────────────────────────
class ReferenceCropsDataset(Dataset):
    """One item = one specimen's 16 reference crops (one view), normalised
    for the backbone. Built in DataLoader workers while the GPU runs."""

    def __init__(self, stems: List[str], view_dirs: Dict[str, Optional[Path]], run_id: Optional[int] = None):
        self.stems = stems
        self.dataset_dir = view_dirs["dataset_dir"]
        self.dataset_a1_dir = view_dirs["dataset_a1_dir"]
        self.dataset_a2_dir = view_dirs["dataset_a2_dir"]
        self.dataset_a7_dir = view_dirs["dataset_a7_dir"]
        self.dataset_a6_class_dir = view_dirs["dataset_a6_class_dir"]
        self.reference_condition = Condition()  # empty set = no ablation active
        self.run_id = run_id

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int):
        stem = self.stems[index]
        # Same seeds as run_eval.py's "reference" condition (same crops);
        # run_id, if set, gives a different but reproducible flip + crop draw
        if self.run_id is None:
            geometry_rng = derive_rng("eval-geometry", stem)
            condition_rng = derive_rng("eval-condition", stem, "reference")
        else:
            geometry_rng = derive_rng("eval-geometry", "run", self.run_id, stem)
            condition_rng = derive_rng("eval-condition", "run", self.run_id, stem, "reference")

        crops, _meta = chain.build_bag(
            self.dataset_dir,
            self.dataset_a1_dir,
            self.dataset_a2_dir,
            self.dataset_a7_dir,
            stem,
            self.reference_condition,
            condition_rng,
            positions_rng=geometry_rng,
            dataset_a6_class_dir=self.dataset_a6_class_dir,
        )

        crops = crops.astype(np.float32) / 255.0
        crops = (crops - IMAGENET_MEAN) / IMAGENET_STD
        crops = torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous()  # (16, 3, 224, 224)
        return {"crops": crops, "stem": stem}


# ── Backbone ─────────────────────────────────────────────────────────────────
def build_raw_backbone(paths_cfg: Dict[str, str], device: torch.device) -> torch.nn.Module:
    """Pretrained DINOv3-S without LoRA (unlike src/model/backbone.py's
    build_backbone(), used for training): all weights frozen, eval mode."""
    weights_path = paths_cfg["dinov3_weights"][BACKBONE_NAME]
    backbone = torch.hub.load(
        paths_cfg["dinov3_repo"],
        BACKBONE_NAME,
        source="local",
        weights=weights_path,
    )
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    return backbone.to(device)


# ── One view ─────────────────────────────────────────────────────────────────
def run_view(
    view: str,
    view_dirs: Dict[str, Optional[Path]],
    out_dir: Path,
    backbone: torch.nn.Module,
    device: torch.device,
    feat_dim: int,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = view_dirs["dataset_dir"]
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"[{view}] missing source view folder: {dataset_dir}")

    stems = sorted(p.name[: -len(".rgb.zst")] for p in dataset_dir.glob("*.rgb.zst"))
    if args.limit:
        stems = stems[: args.limit]
    already_done = {p.name[: -len(".dinov3s_features.zst")] for p in out_dir.glob("*.dinov3s_features.zst")}
    todo = [s for s in stems if s not in already_done]
    print(f"[{view}] {len(stems)} specimens total, {len(already_done)} already extracted, {len(todo)} to do")
    if not todo:
        return

    # What was extracted, written once per output folder
    meta_path = out_dir / "manifest.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "backbone": BACKBONE_NAME,
            "weights_path": args.paths_cfg["dinov3_weights"][BACKBONE_NAME],
            "feat_dim": feat_dim,
            "n_crops": N_CROPS,
            "view": view,
            "condition": "reference (no ablation, same crops as run_eval.py's baseline)",
            "run_id": args.run_id,
            "source_dataset_dir": str(dataset_dir),
        }, indent=2))

    dataset = ReferenceCropsDataset(todo, view_dirs, run_id=args.run_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    # Forward pass: B specimens x 16 crops at once, bf16 on GPU
    use_amp = args.use_amp and device.type == "cuda"
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{view} extract", unit="specimen", unit_scale=args.batch_size):
            crops = batch["crops"].to(device, non_blocking=True)  # (B, 16, 3, 224, 224)
            stems_batch = batch["stem"]
            b, n = crops.shape[:2]
            flat = crops.reshape(b * n, *crops.shape[2:])

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                feats = backbone(flat)  # (B*16, feat_dim)
            feats = feats.reshape(b, n, -1).float().cpu().numpy()

            for i, stem in enumerate(stems_batch):
                npy_io.write_array_zst(out_dir / f"{stem}.dinov3s_features.zst", feats[i])
            n_done += b

    elapsed = time.time() - t0
    print(f"[{view}] done: {n_done} specimens in {elapsed:.0f}s ({elapsed / max(n_done, 1):.3f}s/specimen)")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"))
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    parser.add_argument(
        "--run-id", type=int, default=None,
        help="Folds this into the crop-geometry seed so a repeated run draws DIFFERENT "
             "(but still reproducible) crops of the same images, for multi-run statistical "
             "testing - see module docstring. Omit for the original deterministic reference "
             "crops. Each run-id gets its own run{N}/ output subfolder, so repeats don't "
             "overwrite each other or the no-run-id extraction.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Defaults to $FEATURES_DIR/ (set in .env): "
             "Familly_split_dinov3s_features[/run{N}] - same sibling-folder convention as "
             "the other Familly_split_* caches.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Specimens per forward pass (x16 crops each).")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers building crops in parallel.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N specimens per view (smoke test).")
    parser.add_argument("--use-amp", action="store_true", default=True)
    args = parser.parse_args()

    # ── Paths: parent folders of the Back folders named in paths.yaml ────────
    with open(args.config) as f:
        paths_cfg = expand_config(yaml.safe_load(f))
    args.paths_cfg = paths_cfg  # passed to run_view() for the manifest

    source_root = _check_back_leaf_and_get_parent(Path(paths_cfg["dataset_dir"]), "dataset_dir")
    a1_root = _check_back_leaf_and_get_parent(Path(paths_cfg["dataset_a1_dir"]), "dataset_a1_dir")
    a2_root = _check_back_leaf_and_get_parent(Path(paths_cfg["dataset_a2_dir"]), "dataset_a2_dir")
    a7_root = _check_back_leaf_and_get_parent(Path(paths_cfg["dataset_a7_dir"]), "dataset_a7_dir")
    a6_key = paths_cfg.get("dataset_a6_classification_dir")
    a6_root = _check_back_leaf_and_get_parent(Path(a6_key), "dataset_a6_classification_dir") if a6_key else None

    # ── Output folder ────────────────────────────────────────────────────────
    if args.out_dir:
        out_root = Path(args.out_dir)
    else:
        features_root = Path(expand_config("${FEATURES_DIR}"))  # set in .env
        if not features_root.is_dir():
            raise FileNotFoundError(f"FEATURES_DIR not reachable at {features_root} - is the disk mounted?")
        # next to the Familly_split_A1/A2/A6/A7 caches
        out_root = features_root / "Familly_split_dinov3s_features"
    if args.run_id is not None:
        out_root = out_root / f"run{args.run_id}"
    print(f"output root: {out_root}  (run_id={args.run_id})")

    # ── Backbone ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    backbone = build_raw_backbone(paths_cfg, device)
    with torch.no_grad():
        probe = torch.zeros(1, 3, 224, 224, device=device)
        feat_dim = backbone(probe).shape[-1]
    print(f"backbone: {BACKBONE_NAME}  feat_dim: {feat_dim}  frozen params: {sum(p.numel() for p in backbone.parameters()):,}")

    # ── Extract, view by view ────────────────────────────────────────────────
    for view in args.views:
        view_dirs = {
            "dataset_dir": source_root / VIEW_TEMPLATE.format(view=view),
            "dataset_a1_dir": a1_root / VIEW_TEMPLATE.format(view=view),
            "dataset_a2_dir": a2_root / VIEW_TEMPLATE.format(view=view),
            "dataset_a7_dir": a7_root / VIEW_TEMPLATE.format(view=view),
            "dataset_a6_class_dir": (a6_root / VIEW_TEMPLATE.format(view=view)) if a6_root else None,
        }
        out_dir = out_root / VIEW_TEMPLATE.format(view=view)
        run_view(view, view_dirs, out_dir, backbone, device, feat_dim, args)


if __name__ == "__main__":
    main()
