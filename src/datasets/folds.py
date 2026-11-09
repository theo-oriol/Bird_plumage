from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..view_paths import VIEWS, VIEW_TEMPLATE, check_back_leaf_and_get_parent, view_of_name

# ── Habitat columns ──────────────────────────────────────────────────────────
HABITAT_CODE_COLUMNS = ["1", "2", "3", "4", "8", "12", "14", "9_11", "5_13_15", "6_7"]
N_HABITATS = len(HABITAT_CODE_COLUMNS)


# ── Helpers ──────────────────────────────────────────────────────────────────
# Stems in a folder, listed once (a stat() per file is very slow on this disk)
@lru_cache(maxsize=4)
def _available_stems(dataset_dir: str) -> frozenset:
    suffix_len = len(".rgb.zst")
    return frozenset(p.name[:-suffix_len] for p in Path(dataset_dir).glob("*.rgb.zst"))


def load_habitat_names(paths_cfg: dict) -> List[str]:
    with open(paths_cfg["habitat_names_json"]) as f:
        return json.load(f)


def _stem(name_of_img: str) -> str:
    return Path(name_of_img).stem


# ── Fold loading ─────────────────────────────────────────────────────────────
# One fold split ('train'/'valid'): stem, family, view and habitat distribution (sums to 1), all views
def load_fold(paths_cfg: dict, fold: int, split: str) -> pd.DataFrame:
    assert split in ("train", "valid")
    csv_path = Path(paths_cfg["fold_dir"]) / f"{split}_fold_{fold}.csv"
    df = pd.read_csv(csv_path)

    # Stem = file name without extension or directory
    df["stem"] = df["name_of_img"].map(_stem)
    df["view"] = df["name_of_img"].map(view_of_name)

    # Keep only the CSV files present in their view's folder
    source_root = check_back_leaf_and_get_parent(Path(paths_cfg["dataset_dir"]), "dataset_dir")
    kept = []
    for view in VIEWS:
        available = _available_stems(str(source_root / VIEW_TEMPLATE.format(view=view)))
        kept.append(df[(df["view"] == view) & df["stem"].isin(available)])
    df = pd.concat(kept, ignore_index=True) if kept else df.iloc[0:0]

    # Normalise the habitat distribution
    y = df[HABITAT_CODE_COLUMNS].to_numpy(dtype=np.float64) / 100.0
    row_sums = y.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    y = y / row_sums
    df["habitat"] = list(y)

    return df[["stem", "family", "habitat", "view"]]


# ── Class weights ────────────────────────────────────────────────────────────
# f_c: share of images where habitat c is present; BCE weights w_c ~ 1/f_c
def class_frequencies(train_df: pd.DataFrame) -> np.ndarray:
    support = np.stack([(h > 0).astype(np.float64) for h in train_df["habitat"]])
    return support.mean(axis=0)


def bce_class_weights(train_df: pd.DataFrame) -> np.ndarray:
    freq = class_frequencies(train_df)
    freq = np.clip(freq, 1e-6, None)
    weights = 1.0 / freq
    return weights / weights.mean()


# ── Partition checks ─────────────────────────────────────────────────────────
# Fold checks: validation images and families partition exactly, no train/valid overlap
def check_partition(paths_cfg: dict) -> Dict[str, object]:
    valids = [load_fold(paths_cfg, k, "valid") for k in range(3)]
    trains = [load_fold(paths_cfg, k, "train") for k in range(3)]

    all_valid_stems = pd.concat([v["stem"] for v in valids])
    assert all_valid_stems.is_unique, "validation stems overlap across folds"

    for k in range(3):
        overlap = set(trains[k]["stem"]) & set(valids[k]["stem"])
        assert not overlap, f"fold {k}: {len(overlap)} images in both train and valid"
        fam_overlap = set(trains[k]["family"]) & set(valids[k]["family"])
        assert not fam_overlap, f"fold {k}: families leak across train/valid"

    return {
        "valid_sizes": [len(v) for v in valids],
        "valid_families": [v["family"].nunique() for v in valids],
        "train_sizes": [len(t) for t in trains],
    }
