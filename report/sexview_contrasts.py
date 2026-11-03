import json
import os

import numpy as np
import pandas as pd

# ── Load ─────────────────────────────────────────────────────────────────────
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
AG = json.load(open(f"{SP}/ablation_subsets_groups.json"))["rows"]
AB = dict(np.load(f"{SP}/ablation_subsets_groups.npz"))
AG.update(json.load(open(f"{SP}/ablation_subsets_groups_viewsex.json"))["rows"])
AB.update(dict(np.load(f"{SP}/ablation_subsets_groups_viewsex.npz")))
SPJ = json.load(open(f"{SP}/spatial_subsets.json"))["rows"]
_spb = dict(np.load(f"{SP}/spatial_subsets.npz"))
for s, r in SPJ.items():
    AG[s]["SPATIAL"] = {"mean_delta": r["cost_all"]}
    AB[f"{s}|SPATIAL"] = _spb[f"{s}|all"]

# ── Labels ───────────────────────────────────────────────────────────────────
LABEL = {"A1": "All colour removed (A1)", "A6_dark_brown": "Dark brown only (A6)",
         "A6_light_brown": "Light brown only (A6)", "A6_green": "Green only (A6)",
         "A7": "All light/dark removed, colour kept (A7)", "A2": "Greyscale, standardized lightness (A2)",
         "SPATIAL": "Spatial structure, upper bound (histogram model - network)",
         "A4_theta90_L101": "Blur across the body, 101 px (A4)", "A5": "Spectral slope fixed (A5)"}
ROWS = ["A1", "A6_dark_brown", "A6_light_brown", "A6_green", "A7", "A2", "SPATIAL", "A4_theta90_L101", "A5"]
VIEWS = ("Back", "Belly", "Side")


# ── Helpers ──────────────────────────────────────────────────────────────────
# Pseudo-subset: mean change over several subsets (point and bootstrap draws)
def avg(name, parts):
    rows = {"reference": {"mean": float(np.mean([AG[q]["reference"]["mean"] for q in parts]))}}
    for r in AG[parts[0]]:
        if r == "reference" or any(AG[q][r]["mean_delta"] is None for q in parts):
            continue
        rows[r] = {"mean_delta": float(np.mean([AG[q][r]["mean_delta"] for q in parts]))}
        AB[f"{name}|{r}"] = np.mean([AB[f"{q}|{r}"] for q in parts], 0)
    AG[name] = rows


for v in VIEWS:
    avg(f"V_{v}", [f"M_{v}", f"F_{v}"])
for x in ("M", "F"):
    avg(f"S_{x}", [f"{x}_{v}" for v in VIEWS])

ci = lambda x: np.nanpercentile(x, [2.5, 97.5])


def value(s, row):
    lo, hi = ci(AB[f"{s}|{row}"])
    return AG[s][row]["mean_delta"], lo, hi


def contrast(a, b, row):
    lo, hi = ci(AB[f"{a}|{row}"] - AB[f"{b}|{row}"])
    return AG[a][row]["mean_delta"] - AG[b][row]["mean_delta"], lo, hi


def table(subsets, contrasts, rows, extra=None):
    recs = []
    for row in rows:
        rec = dict(extra or {}, ablation=row, label=LABEL[row])
        for name, s in subsets:
            v, lo, hi = value(s, row)
            rec.update({name: v, f"{name}_lo": lo, f"{name}_hi": hi})
        for name, (a, b) in contrasts:
            v, lo, hi = contrast(a, b, row)
            rec.update({name: v, f"{name}_lo": lo, f"{name}_hi": hi, f"{name}_significant": bool(lo > 0 or hi < 0)})
        recs.append(rec)
    clean = dict(extra or {}, ablation="reference", label="Clean images: mean group AUC",
                 **{name: AG[s]["reference"]["mean"] for name, s in subsets})
    return [clean] + recs


# ── Section 6.4: view and sex ────────────────────────────────────────────────
# 6.4: sex x view
sub64 = [(f"{x}_{v}", f"{x}_{v}") for x in ("M", "F") for v in VIEWS]
con64 = [("belly_minus_back", ("V_Belly", "V_Back")), ("side_minus_back", ("V_Side", "V_Back")),
         ("female_minus_male", ("S_F", "S_M"))]
t64 = pd.DataFrame(table(sub64, con64, ROWS))
t64.to_csv(f"{SP}/sec6_4_view_sex.csv", index=False, float_format="%.4f")

# ── Section 6.5: dimorphism, table A ─────────────────────────────────────────
# 6.5 A: tertile x sex, all views
sub65 = [(s, s) for s in ("low_M", "low_F", "high_M", "high_F")]
con65 = [("male_high_minus_low", ("high_M", "low_M")), ("female_high_minus_low", ("high_F", "low_F")),
         ("high_female_minus_male", ("high_F", "high_M"))]
tA = pd.DataFrame(table(sub65, con65, ROWS))
tA.to_csv(f"{SP}/sec6_5_dimorphism_A.csv", index=False, float_format="%.4f")

# ── Section 6.5: dimorphism, table B ─────────────────────────────────────────
# 6.5 B: the same within each view (main cues)
recs = []
for v in VIEWS:
    sub = [(s, f"{s}_{v}") for s in ("low_M", "low_F", "high_M", "high_F")]
    con = [(n, (f"{a}_{v}", f"{b}_{v}")) for n, (a, b) in con65]
    recs += table(sub, con, ["A1", "A2", "A6_dark_brown", "A6_light_brown", "SPATIAL"], extra={"view": v})
tB = pd.DataFrame(recs)
tB.to_csv(f"{SP}/sec6_5_dimorphism_B.csv", index=False, float_format="%.4f")
n_tests = int(sum(tB[c].notna().sum() for c in tB.columns if c.endswith("_significant")))
print(f"wrote sec6_4_view_sex.csv, sec6_5_dimorphism_A.csv, sec6_5_dimorphism_B.csv ({n_tests} contrasts in table B)")
