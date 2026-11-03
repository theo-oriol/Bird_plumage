#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src import npy_io
from src.env import expand_config
from src.ablations import a1_gray_lstar

VIEWS = ("Back", "Belly", "Side")
VIEW_DIR_TEMPLATE = "NEW_Segmented-Aves-{view}-NPY"

# ── Worker state (set once per process) ──────────────────────────────────────
_SOURCE_DIR = None
_DEST_DIR = None


def _init_worker(source_dir: str, dest_dir: str) -> None:
    global _SOURCE_DIR, _DEST_DIR
    _SOURCE_DIR = Path(source_dir)
    _DEST_DIR = Path(dest_dir)


def _process_one(stem: str) -> None:
    # Greyscale pixels
    canvas = npy_io.read_canvas(_SOURCE_DIR, stem)
    mask = npy_io.foreground_mask(canvas)
    gray = a1_gray_lstar.build(canvas, mask)
    a1_gray_lstar.write_cache_entry(_DEST_DIR, stem, gray)

    # Crop-position tables, unchanged
    for suffix in (".cand_128.zst", ".cand_224.zst"):
        shutil.copy2(_SOURCE_DIR / f"{stem}{suffix}", _DEST_DIR / f"{stem}{suffix}")


# ── One view ─────────────────────────────────────────────────────────────────
def build_view(view: str, source_root: Path, dest_root: Path, workers: int, limit: int | None) -> None:
    source_dir = source_root / VIEW_DIR_TEMPLATE.format(view=view)
    dest_dir = dest_root / VIEW_DIR_TEMPLATE.format(view=view)
    dest_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.name[: -len(".rgb.zst")] for p in source_dir.glob("*.rgb.zst"))
    if limit:
        stems = stems[:limit]

    already_done = {p.name[: -len(".rgb.zst")] for p in dest_dir.glob("*.rgb.zst")}
    todo = [s for s in stems if s not in already_done]
    print(f"[{view}] {len(stems)} specimens total, {len(already_done)} already done, {len(todo)} to do")
    if not todo:
        return

    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(str(source_dir), str(dest_dir))
    ) as pool:
        list(tqdm(pool.map(_process_one, todo, chunksize=8), total=len(todo), desc=view))

    elapsed = time.time() - t0
    print(f"[{view}] done: {len(todo)} specimens in {elapsed:.0f}s ({elapsed / len(todo):.2f}s/specimen, {workers} workers)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=expand_config("${DATASETS_DIR}/Familly_split_no_ablation"),
        help="parent folder containing NEW_Segmented-Aves-{Back,Belly,Side}-NPY",
    )
    parser.add_argument(
        "--dest",
        default=expand_config("${DATASETS_DIR}/Familly_split_grayscale"),
        help="parent folder to create/fill, same layout as --source",
    )
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N specimens per view (smoke test)")
    args = parser.parse_args()

    # ── Check every source view exists before starting ───────────────────────
    source_root = Path(args.source)
    dest_root = Path(args.dest)
    for view in args.views:
        if not (source_root / VIEW_DIR_TEMPLATE.format(view=view)).is_dir():
            raise FileNotFoundError(f"missing source view folder: {source_root / VIEW_DIR_TEMPLATE.format(view=view)}")

    # ── Build ────────────────────────────────────────────────────────────────
    for view in args.views:
        build_view(view, source_root, dest_root, args.workers, args.limit)


if __name__ == "__main__":
    main()
