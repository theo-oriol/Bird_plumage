#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.configio import load_config
from src.evaluation.run_eval import collect_predictions, summarise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="The fold's own train_*.yaml (has the right fold/sham/paths already).")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint (usually .../checkpoints/best.pt).")
    parser.add_argument("--fold", type=int, default=None, help="Defaults to the fold already set in --config.")
    parser.add_argument("--out-dir", default=None, help="Defaults to a sibling 'eval' folder next to the checkpoint's run directory.")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # ── Config, checkpoint and output folder ─────────────────────────────────
    cfg = load_config(args.config)
    fold = args.fold if args.fold is not None else cfg["fold"]
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    # <run>/checkpoints/<name>.pt -> <run>/eval/
    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parents[1] / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: collect predictions ──────────────────────────────────────────
    print(f"[eval] config={args.config}  fold={fold}  checkpoint={checkpoint_path}")
    predictions = collect_predictions(cfg, checkpoint_path, fold)
    predictions_path = out_dir / "predictions.parquet"
    predictions.to_parquet(predictions_path)
    print(f"[eval] {len(predictions)} (image, condition) predictions -> {predictions_path}")

    # ── Step 2: summarise ────────────────────────────────────────────────────
    summary = summarise(predictions, args.n_bootstrap, args.seed)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[eval] summary ({len(summary)} conditions) -> {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
