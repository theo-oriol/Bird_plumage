import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from paths import PROJECT_ROOT

sys.path.insert(0, PROJECT_ROOT)
from src import chain  # noqa: E402
from src.configio import load_config  # noqa: E402
from src.datasets import folds  # noqa: E402
from src.datasets.eval_dataset import EvalDataset  # noqa: E402
from src.evaluation import metrics as metrics_mod  # noqa: E402
from src.model.bagmodel import BagModel  # noqa: E402

# ── Settings ─────────────────────────────────────────────────────────────────
# Clean images of one fold scored with other bag compositions than the trained one (8 x 128 + 8 x 224)
# Draw d uses the same seed as draw d of variance_crop.py (draws 0-2 of the 8 + 8 baseline are comparable)
N_DRAWS = 3
CONFIGS: Dict[str, Tuple[Tuple[int, ...], int]] = {  # name: (crop sizes, crops per size)
    "8_small": ((128,), 8),
    "8_large": ((224,), 8),
    "4_small_4_large": ((128, 224), 4),
    "16_small_16_large": ((128, 224), 16),
}
CROPS_PER_BATCH = 256  # bags per batch scaled so that every config sends ~256 crops per forward


# ── Checkpoint ───────────────────────────────────────────────────────────────
def load_checkpoint(model: BagModel, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.backbone.load_state_dict(state["backbone"])
    model.pool.load_state_dict(state["pool"])
    model.heads.load_state_dict(state["heads"])


# ── Collect predictions ──────────────────────────────────────────────────────
def collect_predictions(cfg: Dict[str, Any], model: BagModel, valid_df: pd.DataFrame,
                        sizes: Tuple[int, ...], per_size: int, name: str) -> pd.DataFrame:
    # build_bag reads these two module constants at each call; DataLoader workers are forked
    # after this point, so they see the new values too
    chain.CROP_SIZES, chain.N_CROPS_PER_SIZE = sizes, per_size
    batch_size = max(1, CROPS_PER_BATCH // (len(sizes) * per_size))

    device = next(model.parameters()).device
    family_by_stem = dict(zip(valid_df["stem"], valid_df["family"]))
    rows = []
    with torch.no_grad():
        for crops_number in range(N_DRAWS):
            dataset = EvalDataset(valid_df, cfg, draw=crops_number)
            dataset._items = [it for it in dataset._items if it[1].name == "reference"]
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=cfg["num_workers"])

            for batch in tqdm(loader, desc=f"{name} draw {crops_number}", unit="batch"):
                crops = batch["crops"].to(device)
                cls_logits, reg_logits, attention = model(crops)
                cls_prob = torch.sigmoid(cls_logits).cpu().numpy()
                reg_prob = F.softmax(reg_logits, dim=-1).cpu().numpy()
                habitat = batch["habitat"].numpy().copy()

                for i, stem in enumerate(batch["stem"]):
                    rows.append({
                        "stem": stem,
                        "family": family_by_stem[stem],
                        "condition_name": batch["condition_name"][i],
                        "habitat": habitat[i],
                        "cls_prob": cls_prob[i],
                        "reg_prob": reg_prob[i],
                        "crops_number": crops_number,
                        "meta": batch["positions"][i].numpy().copy(),
                        "attention": attention[i].cpu().numpy(),
                    })
    return pd.DataFrame(rows)


# ── Summarise ────────────────────────────────────────────────────────────────
def summarise(predictions: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for crops_number, group in predictions.groupby("crops_number"):
        support = np.stack([(h > 0).astype(int) for h in group["habitat"]])
        cls_prob = np.stack(group["cls_prob"])
        reg_prob = np.stack(group["reg_prob"])
        habitat = np.stack(group["habitat"])
        summaries.append({
            "crops_number": crops_number,
            "n_images": len(group),
            "macro_auc": metrics_mod.macro_auc(support, cls_prob),
            "macro_reg_auc": metrics_mod.macro_auc(support, reg_prob),
            "macro_ap": metrics_mod.macro_average_precision(support, cls_prob),
            "mean_kl": metrics_mod.mean_kl(habitat, reg_prob),
        })
    summaries = pd.DataFrame(summaries)
    print(f"AUC variance across draws: {summaries['macro_reg_auc'].var():.3g}")
    return summaries


# ── Main ─────────────────────────────────────────────────────────────────────
# Usage (from report/):
#   python variance_crop_alternative_number_of_crops.py --checkpoint ../runs/<run>/checkpoints/best.pt --fold 0
#   add --configs 8_small 8_large to run only some of them
# Writes one file pair per config in <run>/eval/n_crops/ (predictions_<config>.pkl, summary_<config>.csv)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "eval.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    out_dir = checkpoint_path.parents[1] / "eval" / "n_crops"  # .../<run>/checkpoints/best.pt -> .../<run>/eval/n_crops
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BagModel(cfg).to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()
    valid_df = folds.load_fold(cfg, args.fold, "valid")

    # Each config is written as soon as it is done, so a crash only loses the current one
    for name in args.configs:
        sizes, per_size = CONFIGS[name]
        predictions = collect_predictions(cfg, model, valid_df, sizes, per_size, name)
        predictions.to_pickle(out_dir / f"predictions_{name}.pkl")  # array columns, not CSV-friendly

        summary = summarise(predictions)
        summary.insert(0, "config", name)
        summary.to_csv(out_dir / f"summary_{name}.csv", index=False)
        print(summary.to_string(index=False))
        print(f"wrote {out_dir / f'summary_{name}.csv'} and {out_dir / f'predictions_{name}.pkl'}\n")


if __name__ == "__main__":
    main()
