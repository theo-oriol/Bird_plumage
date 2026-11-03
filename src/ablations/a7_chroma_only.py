from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import zstandard as zstd
from skimage import color


# ── Array I/O ────────────────────────────────────────────────────────────────
_DECOMPRESSOR = zstd.ZstdDecompressor()


def _read_array_zst(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        compressed = f.read()
    raw = _DECOMPRESSOR.decompress(compressed)
    return np.load(io.BytesIO(raw))


def _write_array_zst(path: Path, array: np.ndarray, level: int = 12) -> None:
    buf = io.BytesIO()
    np.save(buf, array)
    compressed = zstd.ZstdCompressor(level=level).compress(buf.getvalue())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(compressed)


# ── Input checks ─────────────────────────────────────────────────────────────
def _check_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected H x W x 3 RGB array, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("RGB image contains non-finite values")
    if arr.min() < -1e-8 or arr.max() > 1.0 + 1e-8:
        raise ValueError("RGB input must be scaled to [0, 1]")
    return np.clip(arr, 0.0, 1.0)


def _check_foreground(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"Foreground mask shape {mask.shape} != image shape {shape}")
    if not np.any(mask):
        raise ValueError("Foreground mask is empty")
    return mask


# ── CIELAB conversion ────────────────────────────────────────────────────────
# sRGB in [0, 1] -> CIELAB (D65)
def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return color.rgb2lab(_check_rgb(rgb), illuminant="D65")


# CIELAB (D65) -> sRGB in [0, 1], clipped
def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    if lab.ndim != 3 or lab.shape[-1] != 3:
        raise ValueError(f"Expected H x W x 3 Lab array, got {lab.shape}")
    with np.errstate(invalid="ignore"):
        rgb = color.lab2rgb(lab, illuminant="D65")
    return np.clip(rgb, 0.0, 1.0)


# ── A7 transform ─────────────────────────────────────────────────────────────
# Fix foreground L*; returns (output_rgb, input_lab, metadata), metadata = chroma lost to the gamut
def chroma_isoluminant(
    rgb: np.ndarray,
    foreground: Optional[np.ndarray] = None,
    luminance_l: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if not 0.0 <= luminance_l <= 100.0:
        raise ValueError("luminance_l must lie in [0, 100]")
    arr = _check_rgb(rgb)
    lab = rgb_to_lab(arr)
    fg = _check_foreground(foreground, lab.shape[:2])
    intended = lab.copy()
    intended[..., 0][fg] = luminance_l
    output = lab_to_rgb(intended)
    output[~fg] = arr[~fg]
    realised = rgb_to_lab(output)

    intended_chroma = np.hypot(intended[..., 1], intended[..., 2])
    realised_chroma = np.hypot(realised[..., 1], realised[..., 2])
    chromatic = fg & (intended_chroma > 1.0)
    retention = np.divide(
        realised_chroma,
        intended_chroma,
        out=np.ones_like(realised_chroma),
        where=intended_chroma > 1.0,
    )
    delta_e = np.linalg.norm(realised - intended, axis=-1)

    metadata: Dict[str, object] = {
        "mode": "constant_L_preserve_ab_before_sRGB_gamut_mapping",
        "constant_L": float(luminance_l),
        "n_foreground_pixels": int(fg.sum()),
        "n_chromatic_pixels": int(chromatic.sum()),
        "mean_intended_chroma": float(intended_chroma[fg].mean()),
        "mean_realized_chroma": float(realised_chroma[fg].mean()),
        "mean_chroma_retention_ratio": (
            float(retention[chromatic].mean()) if np.any(chromatic) else None
        ),
        "median_chroma_retention_ratio": (
            float(np.median(retention[chromatic])) if np.any(chromatic) else None
        ),
        "fraction_chromatic_pixels_losing_gt_5pct_chroma": (
            float(np.mean(retention[chromatic] < 0.95)) if np.any(chromatic) else 0.0
        ),
        "fraction_chromatic_pixels_losing_gt_20pct_chroma": (
            float(np.mean(retention[chromatic] < 0.80)) if np.any(chromatic) else 0.0
        ),
        "mean_delta_E_roundtrip": float(delta_e[fg].mean()),
        "fraction_pixels_delta_E_gt_1": float(np.mean(delta_e[fg] > 1.0)),
        "gamut_or_roundtrip_distortion_detected": bool(np.any(delta_e[fg] > 1.0)),
    }
    return output, lab, metadata


# ── A7 build and cache ───────────────────────────────────────────────────────
CODE = "A7"
FIXED_L = 50.0


# Returns (ablated_rgb_uint8, input_lab_float32, metadata)
def build(rgb_uint8: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    rgb01 = rgb_uint8.astype(np.float64) / 255.0
    out01, lab, meta = chroma_isoluminant(rgb01, foreground=mask, luminance_l=FIXED_L)
    out_uint8 = np.clip(out01 * 255.0, 0, 255).astype(np.uint8)
    return out_uint8, lab.astype(np.float32), meta


def write_cache_entry(dataset_a7_dir: Path, stem: str, out_uint8: np.ndarray) -> None:
    _write_array_zst(Path(dataset_a7_dir) / f"{stem}.rgb.zst", out_uint8)


def read_canvas(dataset_a7_dir: Path, stem: str) -> np.ndarray:
    path = Path(dataset_a7_dir) / f"{stem}.rgb.zst"
    if not path.exists():
        raise FileNotFoundError(
            f"A7 cache entry missing: {path}\n"
            "Run scripts/build_a7_cache.py first (A7 is deterministic "
            "given the specimen, so it's precomputed once instead of online)."
        )
    return _read_array_zst(path)


# Cache the input Lab (float32, HxWx3) next to the A7 image, so it is never recomputed
def write_lab_cache_entry(dataset_a7_dir: Path, stem: str, lab_float32: np.ndarray) -> None:
    _write_array_zst(Path(dataset_a7_dir) / f"{stem}.lab.zst", lab_float32)


def read_lab_cache_entry(dataset_a7_dir: Path, stem: str) -> np.ndarray:
    path = Path(dataset_a7_dir) / f"{stem}.lab.zst"
    if not path.exists():
        raise FileNotFoundError(
            f"A7 Lab cache entry missing: {path}\n"
            "Run scripts/build_a7_cache.py first (the Lab conversion is "
            "deterministic given the specimen, so it's precomputed once "
            "instead of being recomputed online)."
        )
    return _read_array_zst(path)


# ── Run as a script ──────────────────────────────────────────────────────────
# Synthetic specimen for the example below: colour patches on a black background
def _demo_canvas(size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[40 : size - 40, 40 : size - 40] = rng.integers(
        60, 200, size=(size - 80, size - 80, 3), dtype=np.uint8
    )
    canvas[60:100, 60:140] = (200, 40, 40)  # a red patch
    canvas[110:150, 60:140] = (40, 60, 200)  # a blue patch
    canvas[160:200, 60:140] = (220, 200, 40)  # a yellow patch
    mask = canvas.any(axis=-1)
    return canvas, mask


if __name__ == "__main__":
    import tempfile

    canvas, mask = _demo_canvas()
    out, lab, meta = build(canvas, mask)

    l_before = rgb_to_lab(canvas.astype(np.float64) / 255.0)[..., 0][mask]
    l_after = rgb_to_lab(out.astype(np.float64) / 255.0)[..., 0][mask]
    print(f"A7: input {canvas.shape} -> output {out.shape}, cached input Lab {lab.shape} ({lab.dtype})")
    print(f"L* std before: {l_before.std():.2f}  after: {l_after.std():.2f} (should be ~0, L* is now constant)")
    print("gamut/roundtrip: mean chroma retention", round(meta["mean_chroma_retention_ratio"], 4))
    print("cached Lab matches a fresh rgb_to_lab() of the input:", np.allclose(lab[..., 0][mask], l_before, atol=1e-4))

    with tempfile.TemporaryDirectory() as tmp:
        write_cache_entry(tmp, "demo_specimen", out)
        reloaded = read_canvas(tmp, "demo_specimen")
        print("RGB cache round-trip identical:", np.array_equal(out, reloaded))

        write_lab_cache_entry(tmp, "demo_specimen", lab)
        reloaded_lab = read_lab_cache_entry(tmp, "demo_specimen")
        print("Lab cache round-trip identical:", np.array_equal(lab, reloaded_lab))
