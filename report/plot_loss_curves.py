"""Loss evolution per epoch, main vs sham, mean over the 3 folds with a min-max band."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import RUNS_DIR
BASE = RUNS_DIR
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SP, exist_ok=True)
COL = {"main": "#1F7A72", "sham": "#B4622F"}
INK, MUTED, GRID = "#1C2126", "#5B6470", "#E3E0D7"

runs = {}
for m in ("main", "sham"):
    dfs = []
    for k in range(3):
        df = pd.read_csv(f"{BASE}/{m}_fold{k}/metrics.csv").set_index("epoch")
        df["train_total"] = 0.5 * df.kl + 0.5 * df.bce   # fixed weights, like val_loss
        dfs.append(df)
    runs[m] = dfs

panels = [("Total (0.5 KL + 0.5 BCE)", "train_total", "val_loss"),
          ("KL (regression head)", "kl", "val_kl"),
          ("BCE (classification head)", "bce", "val_bce")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)
for ax, (title, tr, va) in zip(axes, panels):
    for m in ("main", "sham"):
        for col, ls, lab in ((tr, (0, (4, 2)), "training"), (va, "-", "validation")):
            arr = np.stack([d[col].to_numpy() for d in runs[m]])
            ep = runs[m][0].index.to_numpy()
            mu = arr.mean(0)
            ax.fill_between(ep, arr.min(0), arr.max(0), color=COL[m], alpha=0.10, lw=0)
            ax.plot(ep, mu, color=COL[m], ls=ls, lw=2, label=f"{m.capitalize()} - {lab}")
        best = [d.early_stop_smoothed.idxmin() for d in runs[m]]
    ax.set_title(title, fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.set_xlabel("epoch")
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(1, 100)
axes[0].set_ylabel("loss (lower is better)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(f"{SP}/loss_curves.png", facecolor="white")
for m in ("main", "sham"):
    print(m, "best epochs", [int(d.early_stop_smoothed.idxmin()) for d in runs[m]],
          "argmin val", [int(d.val_loss.idxmin()) for d in runs[m]])
