import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from paths import PROJECT_ROOT

sys.path.insert(0, PROJECT_ROOT)
from src.configio import load_config  # noqa: E402
from src.datasets import folds  # noqa: E402
from src.datasets.eval_dataset import EvalDataset  # noqa: E402
from src.evaluation import metrics as metrics_mod  # noqa: E402
from src.model.bagmodel import BagModel  # noqa: E402

# Crop-sampling variance: the clean images of one fold scored with N_DRAWS independent crop draws
N_DRAWS = 10


# ── Checkpoint ───────────────────────────────────────────────────────────────
def load_checkpoint(model: BagModel, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.backbone.load_state_dict(state["backbone"])
    model.pool.load_state_dict(state["pool"])
    model.heads.load_state_dict(state["heads"])


# ── Collect predictions ──────────────────────────────────────────────────────
def collect_predictions(cfg: Dict[str, Any], checkpoint_path: Path, fold: int) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BagModel(cfg).to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    valid_df = folds.load_fold(cfg, fold, "valid")
    family_by_stem = dict(zip(valid_df["stem"], valid_df["family"]))
    rows = []
    with torch.no_grad():
        for crops_number in range(N_DRAWS):
            dataset = EvalDataset(valid_df, cfg, draw=crops_number)
            dataset._items = [it for it in dataset._items if it[1].name == "reference"]
            loader = DataLoader(dataset, batch_size=cfg.get("micro_batch_size", 16), shuffle=False,
                                num_workers=cfg["num_workers"])

            for batch in tqdm(loader, desc=f"fold {fold} draw {crops_number}", unit="bag"):
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
# Usage (from report/): python variance_crop.py --checkpoint ../runs/<run>/checkpoints/best.pt --fold 0
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "eval.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    out_dir = checkpoint_path.parents[1] / "eval"  # .../<run>/checkpoints/best.pt -> .../<run>/eval
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = collect_predictions(cfg, checkpoint_path, args.fold)
    predictions.to_pickle(out_dir / "predictions_variance.pkl")  # array columns, not CSV-friendly

    summary = summarise(predictions)
    summary.to_csv(out_dir / "summary_variance.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote {out_dir / 'summary_variance.csv'} and {out_dir / 'predictions_variance.pkl'}")


if __name__ == "__main__":
    main()
