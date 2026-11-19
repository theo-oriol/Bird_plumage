import numpy as np
import pandas as pd

from svlib import BASE, SP, NAMES, NH, MIN_SIDE, family_auc, ci

# ── Settings ─────────────────────────────────────────────────────────────────
# Clean images of fold 0 (sham model) scored with other bag compositions than the trained one
# (8 x 128 px + 8 x 224 px); predictions from variance_crop.py (trained bag, 10 draws) and
# variance_crop_alternative_number_of_crops.py (other bags, 3 draws, same seeds as draws 0-2)
EVAL = f"{BASE}/sham_fold0/eval"
BASELINE = "8_small_8_large"
CONFIGS = {  # name: (crops per bag, prediction file)
    "8_small": (8, f"{EVAL}/n_crops/predictions_8_small.pkl"),
    "8_large": (8, f"{EVAL}/n_crops/predictions_8_large.pkl"),
    "4_small_4_large": (8, f"{EVAL}/n_crops/predictions_4_small_4_large.pkl"),
    BASELINE: (16, f"{EVAL}/predictions_variance.pkl"),
    "16_small_16_large": (32, f"{EVAL}/n_crops/predictions_16_small_16_large.pkl"),
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def draws(path):
    # one (stem-sorted) frame per draw: labels H [n, NH], outputs S [n, NH], families
    d = pd.read_pickle(path)[["stem", "family", "habitat", "reg_prob", "crops_number"]]
    return [g.sort_values("stem") for _, g in d.groupby("crops_number")]


def habitat_aucs(g, fidx, wauc):
    # per-habitat AUC (positive = the bird uses the habitat): point [NH] and bootstrap [NH, B]
    H, S = np.stack(g.habitat.to_numpy()), np.stack(g.reg_prob.to_numpy())
    out = [wauc(S[:, h], H[:, h] > 0, fidx) for h in range(NH)]
    keep = (H > 0).sum(0) >= MIN_SIDE
    return np.where(keep, [o[0] for o in out], np.nan), np.where(keep[:, None], [o[1] for o in out], np.nan)


# ── AUCs per config, averaged over the draws ─────────────────────────────────
# every config scores the same images (fold 0), so one set of family resamples serves them all
ref = draws(CONFIGS[BASELINE][1])[0]
stems = ref.stem.to_numpy()
_, fidx, wauc = family_auc(ref.family.to_numpy())
res = {}
for name, (n_crops, path) in CONFIGS.items():
    ds = draws(path)
    assert all((g.stem.to_numpy() == stems).all() for g in ds)
    per_draw = [habitat_aucs(g, fidx, wauc) for g in ds]
    res[name] = (n_crops, len(ds), np.stack([p[0] for p in per_draw]), np.mean([p[1] for p in per_draw], 0))
    print(f"{name:18s} macro AUC {np.nanmean(res[name][2].mean(0)):.3f}", flush=True)

# ── Table: per habitat and macro, difference to the trained bag ──────────────
# paired: the same family resamples for every bag
P0, BT0 = res[BASELINE][2].mean(0), res[BASELINE][3]
rows = []
for name, (n_crops, n_draws, P_draws, BT) in res.items():
    P = P_draws.mean(0)
    ok = ~np.isnan(P)
    # AUC of each draw on its own (columns auc_draw0.., empty for the draws a bag does not have)
    draw_aucs = {**{NAMES[h]: P_draws[:, h] for h in range(NH)}, "macro": P_draws[:, ok].mean(1)}
    for label, p, bt, p0, bt0, sd in (
            [(NAMES[h], P[h], BT[h], P0[h], BT0[h], P_draws[:, h].std(ddof=1)) for h in range(NH)]
            + [("macro", P[ok].mean(), BT[ok].mean(0), P0[ok].mean(), BT0[ok].mean(0), P_draws[:, ok].mean(1).std(ddof=1))]):
        rows.append({"config": name, "crops_per_bag": n_crops, "n_draws": n_draws, "habitat": label,
                     "auc": p, "auc_lo": ci(bt)[0], "auc_hi": ci(bt)[1], "sd_draws": sd,
                     "delta": p - p0, "delta_lo": ci(bt - bt0)[0], "delta_hi": ci(bt - bt0)[1],
                     **{f"auc_draw{k}": v for k, v in enumerate(draw_aucs[label])}})

out = pd.DataFrame(rows)
out.to_csv(f"{SP}/crop_sampling.csv", index=False, float_format="%.4f")
print(out[out.habitat == "macro"].drop(columns="habitat").to_string(index=False, float_format="%.3f"))
print("wrote crop_sampling.csv")
