#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src import npy_io, color
from src.env import expand_config
from src.ablations import a1_gray_lstar, a6_colour_removal, a7_chroma_only
from src.ablations.common import Condition
from src.chain import build_bag

RESULTS_DIR = Path(__file__).resolve().parents[1] / "runs" / "benchmarks"


# ── Timing ───────────────────────────────────────────────────────────────────
def sample_stems(dataset_dir: Path, n: int, seed: int = 0) -> list:
    """n specimens drawn at random (fixed seed)."""
    all_files = sorted(dataset_dir.glob("*.rgb.zst"))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(all_files), size=min(n, len(all_files)), replace=False)
    return [all_files[i].name[: -len(".rgb.zst")] for i in chosen]


def time_condition(
    dataset_dir, dataset_a1_dir, dataset_a2_dir, dataset_a7_dir, stems, codes, n_repeats=1,
    dataset_a6_class_dir=None, require_a6_cache_hit=False,
):
    """Seconds per build_bag() call under one condition, for each stem."""
    condition = Condition(active=frozenset(codes))
    timings = []
    for stem in stems:
        for rep in range(n_repeats):
            rng = np.random.default_rng(hash((stem, rep)) % (2**32))
            t0 = time.perf_counter()
            try:
                _, meta = build_bag(
                    dataset_dir, dataset_a1_dir, dataset_a2_dir, dataset_a7_dir, stem, condition, rng,
                    dataset_a6_class_dir=dataset_a6_class_dir,
                )
            except FileNotFoundError as e:
                print(f"  [skip] {stem}: {e}")
                continue
            elapsed = time.perf_counter() - t0
            if require_a6_cache_hit and meta.get("A6", {}).get("source") != "precomputed_classification_and_A1_cache":
                print(f"  [skip] {stem}: not yet in the A6 classification cache, would fall back to the online classifier")
                continue
            timings.append(elapsed)
    return timings


def report(name, timings, n_train_images=40987, n_workers=8):
    """Print summary statistics and the projected time per training epoch."""
    if not timings:
        print(f"{name:20s}  no successful runs")
        return None
    m = mean(timings)
    row = {
        "condition": name,
        "n": len(timings),
        "mean_s": m,
        "median_s": median(timings),
        "std_s": stdev(timings) if len(timings) > 1 else 0.0,
        "min_s": min(timings),
        "max_s": max(timings),
        "projected_epoch_hours": m * n_train_images / n_workers / 3600,
    }
    print(
        f"{name:20s}  mean {m:7.3f}s  median {row['median_s']:7.3f}s  "
        f"std {row['std_s']:6.3f}s  -> ~{row['projected_epoch_hours']:.2f} CPU-worker-hours/epoch "
        f"(projected, {n_workers} workers, {n_train_images} images)"
    )
    return row


def fine_grained_cielab_breakdown(dataset_dir, stems):
    """Time each CIELAB operation behind A1, A6 and A7 separately (they
    dominate the coarse timings)."""
    print("\n--- fine-grained breakdown of the CIELAB-exclusive group ---")
    rows = []
    for stem in stems:
        canvas = npy_io.read_canvas(dataset_dir, stem)
        mask = npy_io.foreground_mask(canvas)
        rgb01 = canvas.astype(np.float64) / 255.0

        t0 = time.perf_counter()
        color.achromatic_raw(rgb01, foreground=mask)
        t_a1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        labels = color.classify_delhey_rgb(rgb01, foreground=mask)
        t_classify = time.perf_counter() - t0

        t0 = time.perf_counter()
        present = a6_colour_removal.categories_present(canvas, mask)
        t_presence = time.perf_counter() - t0  # includes its own classify call

        category = present[0] if present else None
        t_ablate = None
        if category is not None:
            t0 = time.perf_counter()
            color.ablate_delhey_colour(rgb01, category=category, foreground=mask, strength=1.0)
            t_ablate = time.perf_counter() - t0

        # What chain.py actually runs: one classification shared by the
        # presence check and the removal (vs. two in t_presence + t_ablate)
        t0 = time.perf_counter()
        rng = np.random.default_rng(0)
        a6_colour_removal.draw_and_apply(canvas, mask, rng)
        t_fused = time.perf_counter() - t0

        t0 = time.perf_counter()
        color.chroma_isoluminant(rgb01, foreground=mask, luminance_l=50.0)
        t_a7 = time.perf_counter() - t0

        rows.append(
            {
                "stem": stem,
                "A1_achromatic_raw_s": t_a1,
                "classify_delhey_rgb_s": t_classify,
                "A6_categories_present_s": t_presence,
                "A6_ablate_delhey_colour_s": t_ablate,
                "A6_draw_and_apply_fused_s": t_fused,
                "A7_chroma_isoluminant_s": t_a7,
            }
        )
        print(
            f"  {stem[:40]:40s}  A1 {t_a1:.2f}s  classify {t_classify:.2f}s  "
            f"A6-presence+ablate(separate) {t_presence + (t_ablate or 0):.2f}s  "
            f"A6-fused {t_fused:.2f}s  A7 {t_a7:.2f}s"
        )
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--skip-slow", action="store_true", help="skip A2/A6/A7 (A2 needs the offline cache; A6/A7 are the slow CIELAB ones)")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"))
    args = parser.parse_args()

    # ── Paths ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        paths_cfg = expand_config(yaml.safe_load(f))
    dataset_dir = Path(paths_cfg["dataset_dir"])
    dataset_a1_dir = Path(paths_cfg["dataset_a1_dir"])
    dataset_a2_dir = Path(paths_cfg["dataset_a2_dir"])
    dataset_a7_dir = Path(paths_cfg["dataset_a7_dir"])
    dataset_a6_class_dir = paths_cfg.get("dataset_a6_classification_dir")

    stems = sample_stems(dataset_dir, args.n_images)
    print(f"benchmarking on {len(stems)} real specimens\n")

    # ── Coarse level ─────────────────────────────────────────────────────────
    # A1/A2/A7 read their caches ([skip] if missing). A6 is timed online here;
    # the A6_cached row below times the cache path on specimens already cached.
    conditions = [
        ("reference", frozenset()), ("A3", {"A3"}), ("A4", {"A4"}), ("A5", {"A5"}),
        ("A1", {"A1"}), ("A2", {"A2"}), ("A7", {"A7"}),
    ]
    if not args.skip_slow:
        conditions.append(("A6", {"A6"}))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    rows = []
    for name, codes in conditions:
        timings = time_condition(dataset_dir, dataset_a1_dir, dataset_a2_dir, dataset_a7_dir, stems, codes)
        row = report(name, timings)
        if row:
            rows.append(row)

    if not args.skip_slow and dataset_a6_class_dir:
        print()
        timings = time_condition(
            dataset_dir, dataset_a1_dir, dataset_a2_dir, dataset_a7_dir, stems, {"A6"},
            dataset_a6_class_dir=Path(dataset_a6_class_dir), require_a6_cache_hit=True,
        )
        row = report("A6_cached", timings)
        if row:
            rows.append(row)

    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out_csv}")

    # ── Fine level (5 specimens) ─────────────────────────────────────────────
    if not args.skip_slow:
        breakdown = fine_grained_cielab_breakdown(dataset_dir, stems[: min(5, len(stems))])
        breakdown_csv = RESULTS_DIR / f"benchmark_cielab_breakdown_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(breakdown_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(breakdown[0].keys()))
            writer.writeheader()
            writer.writerows(breakdown)
        print(f"wrote {breakdown_csv}")


if __name__ == "__main__":
    main()
