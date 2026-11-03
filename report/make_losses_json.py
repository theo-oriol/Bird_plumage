"""Loss table of the report: per model and fold, the evaluated checkpoint's train/validation losses.

Recovered from the one-off command that produced losses.json. Reads runs/{model}_fold{k}/metrics.csv
and manifest.json; the evaluated checkpoint is the epoch with the lowest smoothed early-stop criterion.
"""
import json
import os
import pandas as pd

from paths import RUNS_DIR
RUNS = RUNS_DIR
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SP, exist_ok=True)

out = {}
for m in ["main", "sham"]:
    rows = []
    for k in range(3):
        df = pd.read_csv(f"{RUNS}/{m}_fold{k}/metrics.csv")
        df["fixed"] = 0.5 * df.kl + 0.5 * df.bce
        b = df.loc[df.early_stop_smoothed.idxmin()]
        n = json.load(open(f"{RUNS}/{m}_fold{k}/manifest.json"))["valid_size"]
        rows.append(dict(fold=k, best_epoch=int(b.epoch), n_epochs=len(df), n_val=n, train_loss=b.fixed, train_kl=b.kl, train_bce=b.bce,
                         val_loss=b.val_loss, val_kl=b.val_kl, val_bce=b.val_bce, min_val=df.val_loss.min(),
                         argmin_val=int(df.loc[df.val_loss.idxmin(), "epoch"])))
    print(m)
    print(pd.DataFrame(rows).round(4).to_string())
    out[m] = rows
json.dump(out, open(f"{SP}/losses.json", "w"))
