#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src import npy_io
from src.env import expand_config
from src.ablations import a1_gray_lstar

VIEWS = ("Back", "Belly", "Side")
VIEW_TEMPLATE = "NEW_Segmented-Aves-{view}-NPY"

# ── Worker state (set once per process) ──────────────────────────────────────
_DATASET_DIR = None
_DATASET_A1_DIR = None


def _init_worker(dataset_dir: str, dataset_a1_dir: str) -> None:
    global _DATASET_DIR, _DATASET_A1_DIR
    _DATASET_DIR = Path(dataset_dir)
    _DATASET_A1_DIR = Path(dataset_a1_dir)


def _process_one(stem: str) -> None:
    canvas = npy_io.read_canvas(_DATASET_DIR, stem)
    mask = npy_io.foreground_mask(canvas)
    out_uint8 = a1_gray_lstar.build(canvas, mask)
    a1_gray_lstar.write_cache_entry(_DATASET_A1_DIR, stem, out_uint8)


# ── One view ─────────────────────────────────────────────────────────────────
def build_view(view: str, source_root: Path, dest_root: Path, workers: int, limit: Optional[int]) -> None:
    dataset_dir = source_root / VIEW_TEMPLATE.format(view=view)
    dataset_a1_dir = dest_root / VIEW_TEMPLATE.format(view=view)
    dataset_a1_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.name[: -len(".rgb.zst")] for p in dataset_dir.glob("*.rgb.zst"))
    if limit:
        stems = stems[:limit]

    already_done = {p.name[: -len(".rgb.zst")] for p in dataset_a1_dir.glob("*.rgb.zst")}
    todo = [s for s in stems if s not in already_done]
    print(f"[{view}] {len(stems)} specimens total, {len(already_done)} already cached, {len(todo)} to do")
    if not todo:
        return

    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(str(dataset_dir), str(dataset_a1_dir))
    ) as pool:
        list(tqdm(pool.map(_process_one, todo, chunksize=8), total=len(todo), desc=view))

    elapsed = time.time() - t0
    print(f"[{view}] done: {len(todo)} specimens in {elapsed:.0f}s ({elapsed/len(todo):.2f}s/specimen, {workers} workers)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"))
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N specimens per view (smoke test)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # ── Paths: parent folders of the Back folders named in paths.yaml ────────
    with open(args.config) as f:
        paths_cfg = expand_config(yaml.safe_load(f))
    dataset_dir = Path(paths_cfg["dataset_dir"])
    dataset_a1_dir = Path(paths_cfg["dataset_a1_dir"])
    source_root = _check_back_leaf_and_get_parent(dataset_dir, "dataset_dir")
    dest_root = _check_back_leaf_and_get_parent(dataset_a1_dir, "dataset_a1_dir")

    # ── Check every source view exists before starting ───────────────────────
    for view in args.views:
        source_dir = source_root / VIEW_TEMPLATE.format(view=view)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"missing source view folder: {source_dir}")

    # ── Build ────────────────────────────────────────────────────────────────
    for view in args.views:
        build_view(view, source_root, dest_root, args.workers, args.limit)


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


if __name__ == "__main__":
    main()
