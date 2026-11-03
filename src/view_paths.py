from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

# ── Views ────────────────────────────────────────────────────────────────────
VIEWS = ("Back", "Belly", "Side")
VIEW_TEMPLATE = "NEW_Segmented-Aves-{view}-NPY"


# ── Helpers ──────────────────────────────────────────────────────────────────
# View of a filename or stem (embedded in the name, e.g. '..._Back_...')
def view_of_name(name: str) -> str:
    for view in VIEWS:
        if f"_{view}_" in name:
            return view
    raise ValueError(f"couldn't find a view (Back/Belly/Side) in {name!r}")


# Parent of a 'NEW_Segmented-Aves-Back-NPY' folder (where Belly and Side are); raises otherwise
def check_back_leaf_and_get_parent(path: Path, config_key: str) -> Path:
    expected = path.parent / VIEW_TEMPLATE.format(view="Back")
    if path != expected:
        raise ValueError(
            f"configs/paths.yaml {config_key} ({path}) isn't a "
            f"'{VIEW_TEMPLATE.format(view='Back')}' folder directly under its parent - "
            f"can't derive the Belly/Side sibling folders from it."
        )
    return path.parent


# Dataset and cache folders for each view, derived from the Back folders in paths.yaml
def resolve_view_dirs(paths_cfg: dict) -> Dict[str, Dict[str, Optional[Path]]]:
    source_root = check_back_leaf_and_get_parent(Path(paths_cfg["dataset_dir"]), "dataset_dir")
    a1_root = check_back_leaf_and_get_parent(Path(paths_cfg["dataset_a1_dir"]), "dataset_a1_dir")
    a2_root = check_back_leaf_and_get_parent(Path(paths_cfg["dataset_a2_dir"]), "dataset_a2_dir")
    a7_root = check_back_leaf_and_get_parent(Path(paths_cfg["dataset_a7_dir"]), "dataset_a7_dir")
    a6_key = paths_cfg.get("dataset_a6_classification_dir")
    a6_root = check_back_leaf_and_get_parent(Path(a6_key), "dataset_a6_classification_dir") if a6_key else None

    dirs_by_view = {}
    for view in VIEWS:
        leaf = VIEW_TEMPLATE.format(view=view)
        dirs_by_view[view] = {
            "dataset_dir": source_root / leaf,
            "dataset_a1_dir": a1_root / leaf,
            "dataset_a2_dir": a2_root / leaf,
            "dataset_a7_dir": a7_root / leaf,
            "dataset_a6_class_dir": (a6_root / leaf) if a6_root else None,
        }
    return dirs_by_view
