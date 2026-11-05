#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # make `src` importable

from src.configio import load_config
from src.datasets import folds
from src.datasets.eval_dataset import EvalDataset
from src.evaluation.run_eval import load_checkpoint
from src.model.bagmodel import BagModel


# ── One fold ─────────────────────────────────────────────────────────────────
# Pooled MIL embedding (what the heads read) of each held-out image, reference crops only
def run_fold(model_name: str, fold: int, args: argparse.Namespace) -> None:
    run_dir = ROOT / "runs" / f"{model_name}_fold{fold}"
    out = run_dir / "eval" / ("embeddings_smoke.npz" if args.limit else "embeddings.npz")
    if out.exists():
        print(f"[{model_name} fold {fold}] {out} exists, skipped")
        return
    prefix = "train_sham_fold" if model_name == "sham" else "train_fold"
    cfg = load_config(ROOT / "configs" / f"{prefix}{fold}.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BagModel(cfg).to(device)
    load_checkpoint(model, run_dir / "checkpoints" / "best.pt")
    model.eval()

    pooled = {}
    model.pool.register_forward_hook(lambda _m, _i, o: pooled.__setitem__("x", o[0]))

    valid_df = folds.load_fold(cfg, fold, "valid")
    if args.limit:
        valid_df = valid_df.iloc[:args.limit]
    dataset = EvalDataset(valid_df, cfg)
    dataset._items = [it for it in dataset._items if it[1].name == "reference"]
    loader = DataLoader(dataset, batch_size=cfg["micro_batch_size"], shuffle=False, num_workers=args.workers)

    stems, emb, attn, reg = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{model_name} fold {fold}", unit="bag", unit_scale=cfg["micro_batch_size"]):
            _cls, reg_logits, attention = model(batch["crops"].to(device))
            stems += list(batch["stem"])
            emb.append(pooled["x"].float().cpu().numpy())
            attn.append(attention.float().cpu().numpy())
            reg.append(F.softmax(reg_logits, dim=-1).cpu().numpy())
    np.savez_compressed(out, stem=np.array(stems), embedding=np.concatenate(emb).astype(np.float32),
                        attention=np.concatenate(attn).astype(np.float32), reg_prob=np.concatenate(reg))
    print(f"[{model_name} fold {fold}] {len(stems)} images -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sham", choices=("sham", "main"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="First N images per fold (smoke test)")
    args = parser.parse_args()
    for k in args.folds:
        run_fold(args.model, k, args)


if __name__ == "__main__":
    main()
