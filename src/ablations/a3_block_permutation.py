from __future__ import annotations

import numpy as np

# ── Settings ─────────────────────────────────────────────────────────────────
CODE = "A3"
CROP_SIDE = 224
SIDE_CHOICES = (1, 2, 4, 7, 8, 14, 16, 28, 32, 56)


# ── A3 block permutation ─────────────────────────────────────────────────────
def draw_side(rng: np.random.Generator) -> int:
    return int(SIDE_CHOICES[rng.integers(len(SIDE_CHOICES))])


def apply(crop_uint8: np.ndarray, h: int, rng: np.random.Generator) -> np.ndarray:
    n = CROP_SIDE // h
    blocks = crop_uint8.reshape(n, h, n, h, 3).transpose(0, 2, 1, 3, 4).reshape(n * n, h, h, 3)
    order = rng.permutation(n * n)
    permuted = blocks[order].reshape(n, n, h, h, 3).transpose(0, 2, 1, 3, 4).reshape(CROP_SIDE, CROP_SIDE, 3)
    return permuted


# ── Run as a script ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, size=(CROP_SIDE, CROP_SIDE, 3), dtype=np.uint8)

    h = draw_side(rng)
    permuted = apply(crop, h, rng)

    print(f"A3: side drawn = {h}px, crop {crop.shape} -> {permuted.shape}")
    print(
        "histogram preserved (same pixel values, just rearranged):",
        np.array_equal(np.sort(crop.ravel()), np.sort(permuted.ravel())),
    )
