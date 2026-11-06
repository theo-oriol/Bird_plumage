from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from . import npy_io
from .ablations import a1_gray_lstar, a2_gray_standardized, a3_block_permutation
from .ablations import a4_directional_blur, a5_spectral_slope, a6_colour_removal, a7_chroma_only
from .ablations.common import Condition
from .sampling import load_candidates, sample_positions

# ── Settings ─────────────────────────────────────────────────────────────────
MEDIAN_BLUR_K = 15
N_CROPS_PER_SIZE = 8
CROP_SIZES = (128, 224)


# ── Blur near crops ──────────────────────────────────────────────────────────
# Median blur (and A4 if given) only around each crop; exact, since both filters are local
def _blur_near_crops(
    canvas: np.ndarray, crop_specs, k: int, a4_params: Dict[str, int] = None
) -> Tuple[np.ndarray, float, float]:
    median_pad = k // 2
    if a4_params is not None:
        theta, length = a4_params["theta"], a4_params["length"]
        extra = length // 2
        y_pad = median_pad + (extra if theta == 90 else 0)
        x_pad = median_pad + (extra if theta == 0 else 0)
    else:
        y_pad = x_pad = median_pad

    out = canvas.copy()
    h, w = canvas.shape[:2]
    t_median = t_a4 = 0.0
    for size, y, x in crop_specs:
        y0, y1 = max(0, y - y_pad), min(h, y + size + y_pad)
        x0, x1 = max(0, x - x_pad), min(w, x + size + x_pad)

        t0 = time.perf_counter()
        block = cv2.medianBlur(canvas[y0:y1, x0:x1], k)
        t_median += time.perf_counter() - t0

        if a4_params is not None:
            t0 = time.perf_counter()
            block = a4_directional_blur.apply(block, theta=theta, length=length)
            t_a4 += time.perf_counter() - t0

        iy, ix = y - y0, x - x0
        out[y : y + size, x : x + size] = block[iy : iy + size, ix : ix + size]
    return out, t_median, t_a4


# ── Build one bag ────────────────────────────────────────────────────────────
# One bag: 16 crops -> (16, 224, 224, 3) uint8 + metadata; positions_rng and fixed_* serve evaluation
def build_bag(
    dataset_dir: Path,
    dataset_a1_dir: Path,
    dataset_a2_dir: Path,
    dataset_a7_dir: Path,
    stem: str,
    condition: Condition,
    rng: np.random.Generator,
    *,
    fixed_a3_side: int = None,
    fixed_a4_params: Dict[str, int] = None,
    fixed_a6_category: str = None,
    positions_rng: np.random.Generator = None,
    dataset_a6_class_dir: Path = None,
    allow_flip: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    geometry_rng = positions_rng if positions_rng is not None else rng
    meta: Dict[str, object] = {"stem": stem, "condition": str(condition)}
    # Seconds per stage; stages not drawn have no key
    timings: Dict[str, float] = {}
    meta["timings"] = timings
    t_bag_start = time.perf_counter()

    # ── Steps 0-1 ────────────────────────────────────────────────────────────
    canvas, mask = read_working_image(
        dataset_dir, dataset_a1_dir, dataset_a2_dir, dataset_a7_dir, stem, condition, rng, meta,
        fixed_a6_category=fixed_a6_category,
        dataset_a6_class_dir=dataset_a6_class_dir,
    )

    # ── Step 2: vertical flip (drawn here, applied after the blur) ───────────
    # Median blur commutes with the flip; blurring before flipping is ~10% faster
    # Always drawn, so turning the flip off leaves the crop positions unchanged
    flip = bool(geometry_rng.random() < 0.5) and allow_flip
    meta["flip"] = flip

    # ── Step 5: crop positions (drawn early) ─────────────────────────────────
    # Positions depend only on the candidate table; drawing them now tells step 3 where to blur
    t0 = time.perf_counter()
    crop_specs_unflipped = []  # (size, y, x), pre-flip coordinates
    for size in CROP_SIZES:
        candidates = load_candidates(dataset_dir, stem, size)
        positions = sample_positions(candidates, N_CROPS_PER_SIZE, geometry_rng)
        crop_specs_unflipped.extend((size, y, x) for y, x in positions)
    timings["sample_positions_s"] = time.perf_counter() - t0

    # ── Step 4: A4 parameters (drawn early) ──────────────────────────────────
    # A4 is fused with the step 3 blur, so its parameters are needed first
    a4_params = None
    if "A4" in condition:
        a4_params = fixed_a4_params if fixed_a4_params is not None else a4_directional_blur.draw_params(rng)
        meta["A4"] = a4_params

    # ── Steps 3-4 fused: median blur, then A4 if drawn ───────────────────────
    # Only around the crops: exact, and skips ~85% of the canvas
    canvas, t_median, t_a4 = _blur_near_crops(canvas, crop_specs_unflipped, MEDIAN_BLUR_K, a4_params)
    timings["median_blur_s"] = t_median
    if a4_params is not None:
        timings["A4_s"] = t_a4

    t0 = time.perf_counter()
    if flip:
        canvas = np.flipud(canvas)
        mask = np.flipud(mask)
    timings["flip_s"] = time.perf_counter() - t0

    # Remap the positions for the flip (y -> H - P - y)
    canvas_h = canvas.shape[0]
    if flip:
        crop_specs = [(size, canvas_h - size - y, x) for size, y, x in crop_specs_unflipped]
    else:
        crop_specs = crop_specs_unflipped
    meta["positions"] = crop_specs

    # ── Steps 6-8: crop, resize, A3, A5 ──────────────────────────────────────
    crops = []
    crop_meta = []
    t_crop_extract = t_a3 = t_a5 = 0.0
    for size, y, x in crop_specs:
        t0 = time.perf_counter()
        crop = canvas[y : y + size, x : x + size, :3].copy()
        if size == 128:
            crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        t_crop_extract += time.perf_counter() - t0

        this_crop_meta = {"source_size": size, "y": y, "x": x}

        if "A3" in condition:
            t0 = time.perf_counter()
            h = fixed_a3_side if fixed_a3_side is not None else a3_block_permutation.draw_side(rng)
            crop = a3_block_permutation.apply(crop, h, rng)
            this_crop_meta["A3_h"] = h
            t_a3 += time.perf_counter() - t0

        if "A5" in condition:
            t0 = time.perf_counter()
            target_alpha = a5_spectral_slope.draw_target_alpha(size, rng)
            crop, a5_meta = a5_spectral_slope.apply(crop, target_alpha)
            this_crop_meta["A5"] = a5_meta
            t_a5 += time.perf_counter() - t0

        crops.append(crop)
        crop_meta.append(this_crop_meta)

    meta["crops"] = crop_meta
    timings["crop_extract_resize_s"] = t_crop_extract  # all 16 crops combined, baseline cost (no ablation)
    if "A3" in condition:
        timings["A3_s"] = t_a3  # all 16 crops combined
    if "A5" in condition:
        timings["A5_s"] = t_a5  # all 16 crops combined
    timings["total_s"] = time.perf_counter() - t_bag_start
    return np.stack(crops, axis=0), meta


# ── Reading the working image ────────────────────────────────────────────────
# Steps 0-1: the (canvas, mask) to crop from; A1/A2/A7 from their caches, A6 from its caches if present
def read_working_image(
    dataset_dir: Path,
    dataset_a1_dir: Path,
    dataset_a2_dir: Path,
    dataset_a7_dir: Path,
    stem: str,
    condition: Condition,
    rng: np.random.Generator,
    meta: Dict[str, object],
    fixed_a6_category: str = None,
    dataset_a6_class_dir: Path = None,
) -> Tuple[np.ndarray, np.ndarray]:
    timings = meta["timings"]
    member = condition.cielab_member
    if member == "A1":
        t0 = time.perf_counter()
        canvas = a1_gray_lstar.read_canvas(dataset_a1_dir, stem)
        mask = npy_io.foreground_mask(canvas)
        timings["read_source_s"] = time.perf_counter() - t0
        return canvas, mask
    if member == "A2":
        t0 = time.perf_counter()
        canvas = a2_gray_standardized.read_canvas(dataset_a2_dir, stem)
        mask = npy_io.foreground_mask(canvas)
        timings["read_source_s"] = time.perf_counter() - t0
        return canvas, mask
    if member == "A7":
        t0 = time.perf_counter()
        canvas = a7_chroma_only.read_canvas(dataset_a7_dir, stem)
        mask = npy_io.foreground_mask(canvas)
        timings["read_source_s"] = time.perf_counter() - t0
        return canvas, mask

    t0 = time.perf_counter()
    canvas = npy_io.read_canvas(dataset_dir, stem)
    mask = npy_io.foreground_mask(canvas)
    timings["read_source_s"] = time.perf_counter() - t0

    if member == "A6":
        t0 = time.perf_counter()
        cached = _read_a6_caches(dataset_a1_dir, dataset_a6_class_dir, stem) if dataset_a6_class_dir else None

        if cached is not None:
            labels, grayscale = cached
            if fixed_a6_category is not None:
                canvas, _mask, a6_meta = a6_colour_removal.apply_from_cache(canvas, labels, grayscale, fixed_a6_category)
                meta["A6_category"] = fixed_a6_category
                meta["A6"] = a6_meta
            else:
                canvas, category, a6_meta = a6_colour_removal.draw_and_apply_from_cache(
                    canvas, mask, labels, grayscale, rng
                )
                meta["A6_category"] = category
                if category is not None:
                    meta["A6"] = a6_meta
        elif fixed_a6_category is not None:
            canvas, a6_meta = a6_colour_removal.apply(canvas, mask, fixed_a6_category)
            meta["A6_category"] = fixed_a6_category
            meta["A6"] = a6_meta
        else:
            # One classification shared by the draw and the removal
            canvas, category, a6_meta = a6_colour_removal.draw_and_apply(canvas, mask, rng)
            meta["A6_category"] = category
            if category is not None:
                meta["A6"] = a6_meta
        timings["A6_s"] = time.perf_counter() - t0
    return canvas, mask


# Both caches A6's fast path needs, or None (then the online classifier is used)
def _read_a6_caches(dataset_a1_dir: Path, dataset_a6_class_dir: Path, stem: str):
    try:
        labels = a6_colour_removal.read_classification_cache_entry(dataset_a6_class_dir, stem)
        grayscale = a1_gray_lstar.read_canvas(dataset_a1_dir, stem)
    except FileNotFoundError:
        return None
    return labels, grayscale
