from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import zstandard as zstd

# ── Compressed arrays ────────────────────────────────────────────────────────
_DECOMPRESSOR = zstd.ZstdDecompressor()


def read_array_zst(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        compressed = f.read()
    raw = _DECOMPRESSOR.decompress(compressed)
    return np.load(io.BytesIO(raw))


def write_array_zst(path: Path, array: np.ndarray, level: int = 12) -> None:
    buf = io.BytesIO()
    np.save(buf, array)
    compressed = zstd.ZstdCompressor(level=level).compress(buf.getvalue())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(compressed)


# ── Canvas ───────────────────────────────────────────────────────────────────
# The (2024, 2024, 3) uint8 canvas of one specimen (pipeline step 0)
def read_canvas(dataset_dir: Path, stem: str) -> np.ndarray:
    return read_array_zst(Path(dataset_dir) / f"{stem}.rgb.zst")


# Non-zero pixels = plumage, black = background (a truly black plumage pixel counts as background)
def foreground_mask(canvas: np.ndarray) -> np.ndarray:
    return canvas.any(axis=-1)
