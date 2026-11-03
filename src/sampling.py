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
    cell_id = payload[2 + 2 * n : 2 + 3 * n]
    unique_cells = np.unique(cell_id)
    in_cell = {int(c): np.where(cell_id == c)[0] for c in unique_cells}
    return {
        "valid_ys": valid_ys,
        "valid_xs": valid_xs,
        "unique_cells": unique_cells,
        "in_cell": in_cell,
    }


# ── Sampling ─────────────────────────────────────────────────────────────────
# n_positions top-left corners, one grid cell at a time so draws stay >= P/2 apart
def sample_positions(candidates: dict, n_positions: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    if candidates.get("edge_case"):
        return [(0, 0)] * n_positions
    if "fallback" in candidates:
        return [candidates["fallback"]] * n_positions

    valid_ys = candidates["valid_ys"]
    valid_xs = candidates["valid_xs"]
    cells = candidates["unique_cells"].copy()
    rng.shuffle(cells)

    positions: List[Tuple[int, int]] = []
    for cell in cells:
        if len(positions) >= n_positions:
            break
        choices = candidates["in_cell"][int(cell)]
        chosen = choices[rng.integers(len(choices))]
        positions.append((int(valid_ys[chosen]), int(valid_xs[chosen])))

    # More crops than P/2 cells (small specimens): uniform draws for the rest
    while len(positions) < n_positions:
        idx = rng.integers(len(valid_ys))
        positions.append((int(valid_ys[idx]), int(valid_xs[idx])))

    return positions
