import json
from svlib import SP, GN, PAIRS, load_hist_images, hist_pred, family_auc, metrics, compare

# ── Load ─────────────────────────────────────────────────────────────────────
H, F, _, net = load_hist_images(("main", "sham"), ("A1", "A7"))
_, fidx, wauc = family_auc(F)
hist = {"L": metrics(H, hist_pred("_L"), fidx, wauc, habitats=False),
        "ab": metrics(H, hist_pred("_ab"), fidx, wauc, habitats=False)}

# ── Network vs. histogram per channel ────────────────────────────────────────
# network with colour removed (A1) vs. lightness histogram; lightness removed (A7) vs. colour histogram
out = {"pairs": [[GN[a], GN[b]] for a, b in PAIRS]}
for model in ("main", "sham"):
    for c, b in (("A1", "L"), ("A7", "ab")):
        r = metrics(H, net[(model, c)], fidx, wauc, habitats=False)
        key = f"{model} {c} - hist {b}"
        out[key] = {"net": r[0].tolist(), "hist": hist[b][0].tolist(), **compare(r, hist[b]),
                    "mean_net": float(r[0].mean()), "mean_hist": float(hist[b][0].mean())}
        print(f"{key:22s} net {r[0].mean():.3f} vs hist {hist[b][0].mean():.3f}: {out[key]['mean']:+.3f}")

# ── Write ────────────────────────────────────────────────────────────────────
json.dump(out, open(f"{SP}/texture_check.json", "w"), indent=1)
