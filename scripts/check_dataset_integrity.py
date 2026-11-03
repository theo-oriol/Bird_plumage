#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.datasets import folds
from src.env import expand_config


def main() -> None:
    # ── Load paths ───────────────────────────────────────────────────────────
    config_path = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
    with open(config_path) as f:
        paths_cfg = expand_config(yaml.safe_load(f))

    # ── Run the checks (AssertionError on the first violation) ───────────────
    report = folds.check_partition(paths_cfg)
    print("all partition/family-leak checks passed\n")
    print(f"valid sizes:    {report['valid_sizes']}  (protocole: [12630, 18200, 10157])")
    print(f"valid families: {report['valid_families']}  (protocole: [83, 77, 80])")
    print(f"train sizes:    {report['train_sizes']}")


if __name__ == "__main__":
    main()
