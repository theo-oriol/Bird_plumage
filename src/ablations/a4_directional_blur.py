from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

# ── Settings ─────────────────────────────────────────────────────────────────
CODE = "A4"
THETA_CHOICES = (0, 90)
LENGTH_MIN = 17
LENGTH_MAX = 101


# ── A4 directional blur ──────────────────────────────────────────────────────
def draw_params(rng: np.random.Generator) -> Dict[str, int]:
    theta = int(THETA_CHOICES[rng.integers(2)])
    # Length log-uniform in [17, 101], rounded to the nearest odd integer
    log_l = rng.uniform(np.log(LENGTH_MIN), np.log(LENGTH_MAX))
    length = int(round(np.exp(log_l)))
    if length % 2 == 0:
        length += 1
    length = int(np.clip(length, LENGTH_MIN, LENGTH_MAX))
    return {"theta": theta, "length": length}


def apply(rgb_uint8: np.ndarray, theta: int, length: int) -> np.ndarray:
    if theta == 0:
        kernel = np.full((1, length), 1.0 / length, dtype=np.float32)
    else:
        kernel = np.full((length, 1), 1.0 / length, dtype=np.float32)
    blurred = cv2.filter2D(rgb_uint8, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)
    return blurred


# ── Run as a script ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

    params = draw_params(rng)
    blurred = apply(canvas, **params)

    print(f"A4: params drawn = {params}")
    print(f"pixel std before: {canvas.std():.1f}  after: {blurred.std():.1f} (blur should reduce it)")
