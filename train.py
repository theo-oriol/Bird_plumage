#!/usr/bin/env python3

from __future__ import annotations

import argparse

from src.configio import load_config
from src.training.run import Trainer


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a configs/train_*.yaml file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = Trainer(cfg, config_path=args.config)
    trainer.fit()


if __name__ == "__main__":
    main()
