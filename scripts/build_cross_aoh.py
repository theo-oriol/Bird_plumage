#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_COLS = ["1", "2", "3", "4", "8", "12", "14", "9_11", "5_13_15", "6_7"]
SPLITS = [f"{split}_fold_{k}.csv" for k in range(3) for split in ("train", "valid")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a fold folder with its habitat labels replaced by the AOH labels (same folds, same images)."
    )
    parser.add_argument("--cross", default=str(REPO_ROOT / "data/cross/cross_version_1_FAM"), help="source fold folder (PHA labels), read only")
    parser.add_argument("--aoh", default=str(REPO_ROOT / "data/datasets/dataset_version_AOH_1.csv"), help="image table with AOH labels")
    parser.add_argument("--out", default=str(REPO_ROOT / "data/cross/cross_version_1_FAM_AOH"), help="new fold folder (must not exist)")
    parser.add_argument("--pha-matched", default=None,
                        help="optional: also write the PHA folds restricted to the same images, in this folder")
    args = parser.parse_args()

    # ── Paths ────────────────────────────────────────────────────────────────
    src_dir, out_dir = Path(args.cross), Path(args.out)
    out_dirs = [out_dir] + ([Path(args.pha_matched)] if args.pha_matched else [])
    for d in out_dirs:
        if d.exists():
            raise FileExistsError(f"{d} already exists; remove it or choose another --out.")

    # ── AOH labels, one row per image ────────────────────────────────────────
    aoh = pd.read_csv(args.aoh).set_index("name_of_img")[LABEL_COLS]
    assert aoh.index.is_unique, "duplicate image names in the AOH table"

    # ── Relabel each fold file ───────────────────────────────────────────────
    folds_aoh, folds_pha, dropped = {}, {}, []
    for name in SPLITS:
        pha = pd.read_csv(src_dir / name)
        has_aoh = pha["name_of_img"].isin(aoh.index)
        dropped.append(pha.loc[~has_aoh, ["name_of_img", "species", "family"]].assign(file=name))

        # Same rows, same order, same metadata columns; only the labels change
        kept = pha[has_aoh].reset_index(drop=True)
        new = kept.copy()
        new[LABEL_COLS] = aoh.loc[kept["name_of_img"], LABEL_COLS].to_numpy()
        folds_aoh[name], folds_pha[name] = new, kept

    dropped = pd.concat(dropped, ignore_index=True)

    # ── Checks ───────────────────────────────────────────────────────────────
    for name in SPLITS:
        new, kept = folds_aoh[name], folds_pha[name]
        assert list(new.columns) == list(kept.columns)
        assert new["name_of_img"].equals(kept["name_of_img"])
        assert new[LABEL_COLS].notna().all().all()
        sums = new[LABEL_COLS].sum(axis=1)
        assert sums.between(99, 101).all(), f"{name}: label rows do not sum to 100"
    for k in range(3):
        tr, va = folds_aoh[f"train_fold_{k}.csv"], folds_aoh[f"valid_fold_{k}.csv"]
        assert not set(tr["family"]) & set(va["family"]), f"fold {k}: family in both train and valid"
    valid_all = pd.concat([folds_aoh[f"valid_fold_{k}.csv"]["name_of_img"] for k in range(3)])
    assert valid_all.is_unique, "an image is in more than one validation split"

    # ── Write ────────────────────────────────────────────────────────────────
    def write(folder: Path, folds: dict) -> None:
        folder.mkdir(parents=True)
        for name, df in folds.items():
            df.to_csv(folder / name, index=False)
        for extra in ("labelname.json", "labeltoname.json"):  # same 10 classes, same order
            shutil.copy2(src_dir / extra, folder / extra)
        dropped.to_csv(folder / "dropped_images.csv", index=False)

    write(out_dir, folds_aoh)
    if args.pha_matched:
        write(Path(args.pha_matched), folds_pha)

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"{'file':18s} {'source':>8s} {'kept':>8s} {'dropped':>8s}")
    for name in SPLITS:
        n_new = len(folds_aoh[name])
        n_drop = int((dropped["file"] == name).sum())
        print(f"{name:18s} {n_new + n_drop:8,d} {n_new:8,d} {n_drop:8,d}")
    print(f"dropped (no AOH label): {dropped['name_of_img'].nunique()} images, "
          f"{dropped['species'].nunique()} species: {', '.join(sorted(dropped['species'].unique()))}")
    for d in out_dirs:
        print(f"wrote {d}")


if __name__ == "__main__":
    main()
