import gc
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from svlib import SP, B, NH, NAMES, ABLATIONS, load_conditions, family_auc, per_condition, ablation_row, ci

ROWS = ["reference"] + ABLATIONS


# ── Helpers ──────────────────────────────────────────────────────────────────
def habitat_aucs(H, S, fidx):
    # per-habitat AUC (positive = the bird uses the habitat), point [NH] and bootstrap [NH, B]
    P, BT = np.full(NH, np.nan), np.full((NH, B), np.nan)
    for h in range(NH):
        y = H[:, h] > 0
        if y.any() and not y.all():
            P[h], BT[h] = wauc(S[:, h], y, fidx)
    return P, BT


def table(data):
    # one row per condition: AUC, delta vs. clean images (paired, same images), macro means, 95% CIs
    res, clean = per_condition(data, habitat_aucs)
    out = {}
    for row in ROWS:
        P, BT, Pr, BTr = ablation_row(res, clean, row)
        ok = ~np.isnan(P)
        mBT = np.nanmean(BT[ok], 0)
        out[row] = {"n": int(len(data[row if row in data else row + "_r1"]["stem"])),
                    "auc": P.tolist(), "auc_lo": ci(BT)[0].tolist(), "auc_hi": ci(BT)[1].tolist(),
                    "delta": (P - Pr).tolist(), "delta_lo": ci(BT - BTr)[0].tolist(), "delta_hi": ci(BT - BTr)[1].tolist(),
                    "macro": float(np.nanmean(P[ok])), "macro_lo": float(ci(mBT)[0]), "macro_hi": float(ci(mBT)[1]),
                    "macro_delta": float(np.nanmean((P - Pr)[ok])),
                    "macro_delta_lo": float(ci(np.nanmean((BT - BTr)[ok], 0))[0]),
                    "macro_delta_hi": float(ci(np.nanmean((BT - BTr)[ok], 0))[1]),
                    "_BT": BT, "_mBT": mBT}
        print(f"  {row:18s} macro {out[row]['macro']:.3f}  delta {out[row]['macro_delta']:+.3f}", flush=True)
    return out


# ── Main model ───────────────────────────────────────────────────────────────
data = load_conditions("main")
ref = data["reference"]
fam2i, fidx, wauc = family_auc(ref["F"])
assert abs(roc_auc_score(ref["H"][:, 0] > 0, ref["S"][:, 0]) - wauc(ref["S"][:, 0], ref["H"][:, 0] > 0, fidx)[0]) < 1e-9
y = ref["H"] > 0
prevalence = y.mean(0).tolist()
mean_share = [float(ref["H"][y[:, h], h].mean()) for h in range(NH)]
main_rows = table(data)
del data, ref
gc.collect()

# ── Sham model ───────────────────────────────────────────────────────────────
data = load_conditions("sham")
assert set(data["reference"]["F"]) <= set(fam2i)
sham_rows = table(data)
del data
gc.collect()

# ── Paired main - sham ───────────────────────────────────────────────────────
diff = {}
for row in ROWS:
    m, s = main_rows[row], sham_rows[row]
    lo, hi = ci(m["_BT"] - s["_BT"])
    mlo, mhi = ci(m["_mBT"] - s["_mBT"])
    diff[row] = {"diff": (np.array(m["auc"]) - np.array(s["auc"])).tolist(), "diff_lo": lo.tolist(), "diff_hi": hi.tolist(),
                 "macro": m["macro"] - s["macro"], "macro_lo": float(mlo), "macro_hi": float(mhi)}
for r in (main_rows, sham_rows):
    for v in r.values():
        del v["_BT"], v["_mBT"]

# ── Write ────────────────────────────────────────────────────────────────────
json.dump({"habitat_names": NAMES, "B": B, "n_families": len(fam2i), "prevalence": prevalence, "mean_share": mean_share,
           "rows": ROWS, "main": main_rows, "sham": sham_rows, "diff": diff},
          open(f"{SP}/regression_bootstrap.json", "w"))
print("wrote regression_bootstrap.json")
