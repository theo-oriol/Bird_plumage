#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from src.configio import load_config
from src.evaluation.run_eval import collect_predictions, summarise


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parents[1]  # .../<run_name>/checkpoints/best.pt -> .../<run_name>
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = collect_predictions(cfg, checkpoint_path, args.fold)
    predictions.to_pickle(out_dir / "predictions.pkl")  # habitat/prob columns hold arrays, not CSV-friendly

    summary = summarise(predictions, n_bootstrap=cfg["n_bootstrap"], seed=cfg.get("seed", 0))
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote {out_dir / 'summary.csv'} and {out_dir / 'predictions.pkl'}")


if __name__ == "__main__":
    main()
