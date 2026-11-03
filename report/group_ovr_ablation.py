"""Section 5 with the 5 one-vs-rest group AUCs instead of the 10 group pairs.

For every condition: AUC of "main group = G" vs. all other images, score = summed
output of G's habitats. Change vs. clean images on the same images (A6: same
birds only; A3 seeds averaged). Family bootstrap B=500, same draws everywhere.
One model in memory at a time.
"""
import gc
import json
import pickle
import time
import sys
import numpy as np
import pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "main"
from svlib import BASE, SP, B, GN, bootstrap_counts, make_wauc, group_ovr, ci

SEED_GROUPS = {f"A3_h{h}": [f"A3_h{h}_r{r}" for r in (1, 2, 3)] for h in (56, 16, 4, 1)}
SINGLE = ["A1", "A2", "A5", "A7",
          "A4_theta0_L17", "A4_theta0_L41", "A4_theta0_L101",
          "A4_theta90_L17", "A4_theta90_L41", "A4_theta90_L101",
          "A6_blue", "A6_purple", "A6_red", "A6_yellow", "A6_green", "A6_rufous", "A6_light_brown", "A6_dark_brown"]
ROWS = SINGLE + list(SEED_GROUPS)

out = {"groups": GN, "rows": ROWS, "B": B}
wauc = None
for model in ("main", "sham"):
    t0 = time.time()
    parts = []
    for k in range(3):
        df = pickle.load(open(f"{BASE}/{model}_fold{k}/eval/predictions.pkl", "rb"))
        parts.append(df[["stem", "family", "condition_name", "habitat", "reg_prob"]])
        del df
        gc.collect()
    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    if wauc is None:
        fams = sorted(df.family.unique())
        fam2i = {f: i for i, f in enumerate(fams)}
        wauc = make_wauc(bootstrap_counts(fams))
    data = {}
    for cond, sub in df.groupby("condition_name", sort=True):
        data[cond] = {"stem": sub.stem.to_numpy(), "fidx": sub.family.map(fam2i).to_numpy(np.int32),
                      "H": np.stack(sub.habitat.to_numpy()).astype(np.float32),
                      "S": np.stack(sub.reg_prob.to_numpy()).astype(np.float64)}
    del df
    gc.collect()
    print(f"{model}: loaded ({time.time() - t0:.0f}s)", flush=True)

    ref = data["reference"]
    ref_pos = {s: i for i, s in enumerate(ref["stem"])}
    res, ref_sub = {}, {}
    for cond, dd in data.items():
        res[cond] = group_ovr(dd["H"], dd["S"], dd["fidx"], wauc, MODE)
        if cond.startswith("A6"):
            ii = np.array([ref_pos[s] for s in dd["stem"]])
            ref_sub[cond] = group_ovr(ref["H"][ii], ref["S"][ii], ref["fidx"][ii], wauc, MODE)
    del data
    gc.collect()

    P_ref, BT_ref = res["reference"]
    lo, hi = ci(BT_ref)
    rows = {"reference": {"auc": P_ref.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "mean": float(P_ref.mean())}}
    for row in ROWS:
        members = SEED_GROUPS.get(row, [row])
        P = np.nanmean([res[m][0] for m in members], axis=0)
        BT = np.nanmean([res[m][1] for m in members], axis=0)
        Pr, BTr = ref_sub[row] if row.startswith("A6") else (P_ref, BT_ref)
        D = BT - BTr
        dlo, dhi = ci(D)
        rows[row] = {"auc": P.tolist(), "clean_same": Pr.tolist(), "delta": (P - Pr).tolist(), "dlo": dlo.tolist(), "dhi": dhi.tolist(),
                     "mean_delta": float(np.nanmean(P - Pr)), "mean_ci": ci(np.nanmean(D, 0)).tolist()}
    out[model] = rows
    del res, ref_sub
    gc.collect()
    print(f"{model} done ({time.time() - t0:.0f}s)", flush=True)

json.dump(out, open(f"{SP}/group_ovr_ablation_{MODE}.json", "w"))
for model in ("main", "sham"):
    r = out[model]
    print(f"\n== {model}: clean " + " ".join(f"{g} {a:.3f}" for g, a in zip(GN, r["reference"]["auc"])))
    for row in ["A1", "A2", "A7", "A3_h1", "A3_h4", "A3_h16", "A3_h56", "A6_dark_brown", "A6_green", "A4_theta90_L101", "A5"]:
        x = r[row]
        print(f"  {row:16s} mean {x['mean_delta']:+.3f}  " + " ".join(
            f"{d:+.3f}{'*' if l > 0 or h < 0 else ' '}" for d, l, h in zip(x["delta"], x["dlo"], x["dhi"])))
