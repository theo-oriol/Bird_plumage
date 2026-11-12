#!/usr/bin/env python3
# Methods figures: crop sampling (fig. crop system) and the seven ablations on one crop.
# Runs the real training chain (src/chain.build_bag) on segmented PNGs rebuilt into the NPY format.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import chain, npy_io, sampling  # noqa: E402
from src.ablations import combinations, a1_gray_lstar, a2_gray_standardized, a6_colour_removal, a7_chroma_only  # noqa: E402
from src.ablations.common import Condition  # noqa: E402

# ── Settings ─────────────────────────────────────────────────────────────────
CANVAS = 2024
CROP_SIZES = (128, 224)
COLOURS = {128: "#E8833A", 224: "#2F6DB5"}
LEAF = "NEW_Segmented-Aves-Back-NPY"


# ── Specimen in the training format ──────────────────────────────────────────
# Segmented RGBA PNG -> (2024, 2024, 3) uint8 canvas: bbox spans the full width, black background
def png_to_canvas(png: Path) -> np.ndarray:
    rgba = np.asarray(Image.open(png).convert("RGBA"))
    alpha = rgba[..., 3] > 127
    ys, xs = np.nonzero(alpha)
    rgba = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    alpha = alpha[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    scale = CANVAS / rgba.shape[1]
    h = int(round(rgba.shape[0] * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    rgb = cv2.resize(np.ascontiguousarray(rgba[..., :3]), (CANVAS, h), interpolation=interp)
    fg = cv2.resize(alpha.astype(np.uint8), (CANVAS, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    # The mask is "non-zero pixels": background to 0, plumage never exactly 0
    rgb[~fg] = 0
    rgb[fg & ~rgb.any(axis=-1)] = 1

    canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    y0 = (CANVAS - h) // 2
    canvas[y0 : y0 + h] = rgb
    return canvas


# Candidate table: every top-left corner whose PxP window lies 100% in the mask, 1 px step
# (the cell-id block of the format is unused by the sampler, written as zeros)
def candidate_payload(mask: np.ndarray, size: int) -> np.ndarray:
    ii = cv2.integral(mask.astype(np.uint8))
    window = ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size]
    ys, xs = np.nonzero(window == size * size)
    return np.concatenate([[2, len(ys)], ys, xs, np.zeros_like(ys)]).astype(np.int64)


# A2 population target (M, S, L* histogram) from the specimens at hand, one specimen = one vote
def a2_population(canvases: list) -> tuple:
    stats = [a2_gray_standardized.compute_stats(c, npy_io.foreground_mask(c)) for c in canvases]
    M = float(np.mean([s["mean"] for s in stats]))
    S = float(np.mean([s["std"] for s in stats]))
    hists = [a2_gray_standardized.compute_target_contribution(c, npy_io.foreground_mask(c), M, S)["hist"] for c in canvases]
    target = np.mean(hists, axis=0)
    return M, S, target / target.sum()


# Write source, A1/A2/A7 caches, A6 labels and candidate tables, as the training chain expects
def write_specimen(root: Path, stem: str, canvas: np.ndarray, a2_target: tuple) -> dict:
    dirs = {k: root / k / LEAF for k in ("src", "a1", "a2", "a7", "a6")}
    mask = npy_io.foreground_mask(canvas)
    npy_io.write_array_zst(dirs["src"] / f"{stem}.rgb.zst", canvas)
    for size in CROP_SIZES:
        npy_io.write_array_zst(dirs["src"] / f"{stem}.cand_{size}.zst", candidate_payload(mask, size))

    a1_gray_lstar.write_cache_entry(dirs["a1"], stem, a1_gray_lstar.build(canvas, mask))
    M, S, target = a2_target
    a2_out, _ = a2_gray_standardized.build(canvas, mask, M, S, target, rng=np.random.default_rng(0))
    a2_gray_standardized.write_cache_entry(dirs["a2"], stem, a2_out)
    a7_out, _, _ = a7_chroma_only.build(canvas, mask)
    a7_chroma_only.write_cache_entry(dirs["a7"], stem, a7_out)
    labels = a6_colour_removal.build_classification(canvas, mask)
    a6_colour_removal.write_classification_cache_entry(dirs["a6"], stem, labels)
    return {**dirs, "labels": labels}


def bag(dirs: dict, stem: str, codes=(), seed: int = 0, **fixed):
    return chain.build_bag(
        dirs["src"], dirs["a1"], dirs["a2"], dirs["a7"], stem, Condition(active=frozenset(codes)),
        np.random.default_rng(seed + 1),
        positions_rng=np.random.default_rng(seed), dataset_a6_class_dir=dirs["a6"], allow_flip=False, **fixed,
    )


# ── Figure: crop system ──────────────────────────────────────────────────────
# Admissible window centres (corner + P/2) for one crop size, coloured by strip, with the drawn windows
def draw_admissible(ax, dirs: dict, stem: str, mask: np.ndarray, meta: dict, size: int, y_lo: int, y_hi: int) -> None:
    half = size // 2
    cand = sampling.load_candidates(dirs["src"], stem, size)
    ys, xs = cand["valid_ys"] + half, cand["valid_xs"] + half
    strips = sampling.equal_length_strips(cand, chain.N_CROPS_PER_SIZE)
    tint = np.array(matplotlib.colors.to_rgb(COLOURS[size])) * 255
    shown = np.where(mask[..., None], 235, 255).astype(np.uint8).repeat(3, axis=-1)
    for i, strip in enumerate(strips):
        mix = 0.55 if i % 2 == 0 else 0.3
        shown[ys[strip], xs[strip]] = (mix * tint + (1 - mix) * 255).astype(np.uint8)
    ax.imshow(shown[y_lo:y_hi], interpolation="nearest")
    for strip in strips[1:]:
        if len(strip):
            ax.axvline(xs[strip[0]], color="0.35", lw=0.6, ls=":")
    drawn = [(y, x) for s, y, x in meta["positions"] if s == size]
    for y, x in drawn:
        ax.plot(x + half, y + half - y_lo, "o", ms=3.2, color=COLOURS[size], mec="white", mew=0.5)
    # One example window around its centre
    y, x = drawn[len(drawn) // 2]
    ax.add_patch(Rectangle((x, y - y_lo), size, size, fill=False, lw=1.0, ls="--", ec=COLOURS[size]))
    ax.set_xlim(0, CANVAS)
    ax.set_ylim(y_hi - y_lo, 0)
    ax.axis("off")


# Bags as drawn during training: conditions from the online sampler, one per colour ablation, together covering A3-A5
def online_bags(dirs: dict, stem: str) -> list:
    colour, spatial = {"A1", "A2", "A6", "A7"}, {"A3", "A4", "A5"}
    out, used = [], set()
    for seed in range(10_000):
        cond = combinations.draw_condition(np.random.default_rng(seed))
        c, sp = cond.active & colour, cond.active & spatial
        missing = spatial - used
        if not c or c & used or not sp or (missing and not sp & missing):
            continue
        used |= cond.active
        crops, meta = chain.build_bag(
            dirs["src"], dirs["a1"], dirs["a2"], dirs["a7"], stem, cond, np.random.default_rng(seed + 10_000),
            dataset_a6_class_dir=dirs["a6"],
        )
        out.append((cond, crops, meta))
        if len(out) == len(colour):
            break
    return out


def condition_label(cond, meta: dict) -> str:
    parts = []
    for code in sorted(cond.active):
        if code == "A4":
            parts.append(f"A4 (θ = {meta['A4']['theta']}°, ℓ = {meta['A4']['length']})")
        elif code == "A6":
            parts.append(f"A6 ({meta['A6_category'].replace('_', ' ')})")
        elif code == "A3":
            parts.append(f"A3 (h = {meta['A3_h']})")
        else:
            parts.append(code)
    return " + ".join(parts)


def fig_crop_system(dirs: dict, stem: str, canvas: np.ndarray, out: Path, seed: int) -> None:
    crops, meta = bag(dirs, stem, seed=seed)
    mask = npy_io.foreground_mask(canvas)
    blurred = cv2.medianBlur(canvas, chain.MEDIAN_BLUR_K)
    bags = online_bags(dirs, stem)

    rows = np.nonzero(mask.any(axis=1))[0]
    y_lo, y_hi = max(0, rows.min() - 60), min(CANVAS, rows.max() + 60)

    # Row heights in units of one (c) crop panel (1/8 of the width), so every image keeps its aspect
    band = 8 * (y_hi - y_lo) / CANVAS
    label, small = 0.3, 0.5
    ratios = [band, band, band, label, small, label] + [small] * len(bags)
    fig = plt.figure(figsize=(7.2, 0.9 * (sum(ratios) + 0.22 * len(ratios))))
    gs = fig.add_gridspec(len(ratios), 16, height_ratios=ratios, hspace=0.35, wspace=0.08)

    # (a) specimen with the 16 windows
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(blurred[y_lo:y_hi], interpolation="lanczos")
    for k, (size, y, x) in enumerate(meta["positions"]):
        ax.add_patch(Rectangle((x, y - y_lo), size, size, fill=False, lw=1.3, ec=COLOURS[size]))
        ax.text(x + 12, y - y_lo + 12, str(k + 1), color=COLOURS[size], fontsize=6.5, va="top", ha="left", weight="bold")
    ax.set_title("(a) Specimen canvas after the median filter, with the 16 sampled windows", fontsize=8, loc="left")
    ax.axis("off")

    # (b) how positions are drawn, for both crop sizes
    for r, size in ((1, 128), (2, 224)):
        ax = fig.add_subplot(gs[r, :])
        draw_admissible(ax, dirs, stem, mask, meta, size, y_lo, y_hi)
        tag = "(b) " if size == 128 else ""
        ax.set_title(f"{tag}P = {size}: admissible window centres in 8 strips of equal length, one window per strip (dots; one dashed)",
                     fontsize=8, loc="left")

    # (c) the 16 crops given to the backbone, reference condition
    for k in range(16):
        ax = fig.add_subplot(gs[4, k])
        ax.imshow(crops[k])
        size = meta["positions"][k][0]
        for spine in ax.spines.values():
            spine.set_edgecolor(COLOURS[size])
            spine.set_linewidth(1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(10, 10, str(k + 1), color="white", fontsize=5, va="top", weight="bold")
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    ax.text(0, 0.1, "(c) The bag given to the backbone (reference): 8 crops of 128 px upsampled to 224 px (orange), "
            "8 native 224 px crops (blue)", fontsize=8, transform=ax.transAxes)

    # (d) bags of the same specimen under conditions drawn online
    ax = fig.add_subplot(gs[5, :])
    ax.axis("off")
    ax.text(0, 0.1, "(d) Bags of the same specimen under conditions drawn online during training",
            fontsize=8, transform=ax.transAxes)
    for r, (cond, bcrops, bmeta) in enumerate(bags):
        for k in range(16):
            ax = fig.add_subplot(gs[6 + r, k])
            ax.imshow(bcrops[k])
            size = bmeta["crops"][k]["source_size"]
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOURS[size])
                spine.set_linewidth(1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.text(0, 1.12, condition_label(cond, bmeta), fontsize=6.5, transform=ax.transAxes, va="bottom")

    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Figure: the seven ablations on one crop ──────────────────────────────────
def fig_ablations(dirs: dict, stem: str, out: Path, seed: int) -> None:
    ref, meta = bag(dirs, stem, seed=seed)
    # The native 224 crop with the most lightness structure, so the ablations are visible
    lstar = [a1_gray_lstar.rgb_to_lab(c.astype(np.float64) / 255.0)[..., 0].std() for c in ref[8:]]
    k = 8 + int(np.argmax(lstar))

    labels = dirs["labels"]
    present = a6_colour_removal.categories_present_from_labels(labels)
    areas = {c: int((labels == a6_colour_removal.DELHEY_CATEGORIES.index(c)).sum()) for c in present}
    a6_cat = max(areas, key=areas.get)

    panels = [
        ("Reference", (), {}),
        ("A1  achromatic", ("A1",), {}),
        ("A2  achromatic,\nstandardised $L^*$", ("A2",), {}),
        (f"A6  remove\n'{a6_cat.replace('_', ' ')}'", ("A6",), {"fixed_a6_category": a6_cat}),
        ("A7  isoluminant\n$L^*=50$", ("A7",), {}),
        ("A3  blocks\nh = 28", ("A3",), {"fixed_a3_side": 28}),
        ("A3  blocks\nh = 4", ("A3",), {"fixed_a3_side": 4}),
        ("A4  blur θ = 0°\nℓ = 101", ("A4",), {"fixed_a4_params": {"theta": 0, "length": 101}}),
        ("A4  blur θ = 90°\nℓ = 101", ("A4",), {"fixed_a4_params": {"theta": 90, "length": 101}}),
        ("A5  spectral\nslope", ("A5",), {}),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.9))
    for ax, (title, codes, fixed) in zip(axes.ravel(), panels):
        crops, _ = bag(dirs, stem, codes, seed=seed, **fixed)
        ax.imshow(crops[k])
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("colour\n(image level)", fontsize=8)
    axes[1, 0].set_ylabel("spatial\n(image or crop)", fontsize=8)
    fig.tight_layout(h_pad=1.2, w_pad=0.3)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", type=Path, required=True, help="folder of segmented RGBA *_Back_*.png")
    parser.add_argument("--specimen", required=True, help="stem of the specimen to draw")
    parser.add_argument("--work-dir", type=Path, required=True, help="scratch folder for the NPY files")
    parser.add_argument("--out-dir", type=Path, default=REPO / "docs" / "figures")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(args.png_dir.glob("*_Back_*.png"))
    print(f"{len(pngs)} specimens for the A2 population target")
    canvases = {p.stem: png_to_canvas(p) for p in pngs}
    a2_target = a2_population(list(canvases.values()))

    canvas = canvases[args.specimen]
    dirs = write_specimen(args.work_dir, args.specimen, canvas, a2_target)
    fig_crop_system(dirs, args.specimen, canvas, args.out_dir / "crop_system", args.seed)
    fig_ablations(dirs, args.specimen, args.out_dir / "ablations", args.seed)
    print(f"figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
