from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .npy_io import read_array_zst


# ── Candidate positions ──────────────────────────────────────────────────────
def load_candidates(dataset_dir: Path, stem: str, crop_size: int) -> dict:
    path = Path(dataset_dir) / f"{stem}.cand_{crop_size}.zst"
    payload = read_array_zst(path)
    return _parse_candidates(payload)


def _parse_candidates(payload: np.ndarray) -> dict:
    kind = int(payload[0])
    if kind == 0:
        return {"edge_case": True}
    if kind == 1:
        return {"fallback": (int(payload[1]), int(payload[2]))}

    n = int(payload[1])
    valid_ys = payload[2 : 2 + n]
    valid_xs = payload[2 + n : 2 + 2 * n]
    # The stored P/2 grid-cell ids (payload[2+2n:]) are ignored: strips along the body replace them
    return {
        "valid_ys": valid_ys,
        "valid_xs": valid_xs,
        "x_order": np.argsort(valid_xs, kind="stable"),
    }


# ── Sampling ─────────────────────────────────────────────────────────────────
# One top-left corner per strip along the body: the x range of admissible corners is cut into
# n_positions strips of equal length, so every bag covers head to tail and narrow parts weigh as much as wide ones
def sample_positions(candidates: dict, n_positions: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    if candidates.get("edge_case"):
        return [(0, 0)] * n_positions
    if "fallback" in candidates:
        return [candidates["fallback"]] * n_positions

    valid_ys = candidates["valid_ys"]
    valid_xs = candidates["valid_xs"]
    strips = equal_length_strips(candidates, n_positions)

    positions: List[Tuple[int, int]] = []
    for strip in strips:
        # Empty strip (gap in the admissible x range, or tiny specimen): filled below
        if len(strip) == 0:
            continue
        chosen = strip[rng.integers(len(strip))]
        positions.append((int(valid_ys[chosen]), int(valid_xs[chosen])))

    while len(positions) < n_positions:
        idx = rng.integers(len(valid_ys))
        positions.append((int(valid_ys[idx]), int(valid_xs[idx])))

    return positions


# Indices of admissible corners in each of n equal-length x intervals, head to tail
def equal_length_strips(candidates: dict, n: int) -> List[np.ndarray]:
    order = candidates["x_order"]
    xs_sorted = candidates["valid_xs"][order]
    edges = np.linspace(xs_sorted[0], xs_sorted[-1] + 1, n + 1)
    cuts = np.searchsorted(xs_sorted, edges[1:-1], side="left")
    return np.split(order, cuts)
