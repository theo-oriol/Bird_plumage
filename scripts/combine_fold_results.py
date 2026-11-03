#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs=3, required=True, help="the three folds' summary.csv, in any order")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # ── Load the three summaries ─────────────────────────────────────────────
    folds = [pd.read_csv(p) for p in args.summaries]
    for i, df in enumerate(folds):
        df["fold"] = i

    combined = pd.concat(folds, ignore_index=True)
    combined = combined[combined["condition_name"] != "reference"]

    # ── Combine per condition ────────────────────────────────────────────────
    rows = []
    for condition_name, group in combined.groupby("condition_name"):
        deltas = group["delta_auc"].to_numpy()
        variances = group["delta_auc_variance"].to_numpy()
        sizes = group["n_images"].to_numpy()

        # Inverse-variance mean and its standard error
        weights = 1.0 / variances
        inverse_variance_mean = float(np.sum(weights * deltas) / np.sum(weights))
        inverse_variance_std = float(np.sum(weights) ** -0.5)

        rows.append(
            {
                "condition_name": condition_name,
                "n_folds": len(group),
                "delta_auc_inverse_variance": inverse_variance_mean,
                "delta_auc_inverse_variance_std": inverse_variance_std,
                "delta_auc_simple_mean": float(deltas.mean()),
                "delta_auc_size_weighted_mean": float(np.sum(sizes * deltas) / np.sum(sizes)),
                "delta_auc_per_fold": list(deltas),
            }
        )

    # ── Write output ─────────────────────────────────────────────────────────
    out_df = pd.DataFrame(rows).sort_values("delta_auc_inverse_variance")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(out_df.drop(columns=["delta_auc_per_fold"]).to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
