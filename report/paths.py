import os

# ── Read .env ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))


def _load_env(path):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), os.path.expanduser(val))


_load_env(os.path.join(os.path.dirname(HERE), ".env"))


# ── Paths inside the repository ──────────────────────────────────────────────
def _get(key, default=None):
    val = os.environ.get(key, default)
    if val is None:
        raise SystemExit(f"{key} is not set: copy .env.example to .env at the repository root and fill it in.")
    return val


PROJECT_ROOT = _get("PROJECT_ROOT", os.path.dirname(HERE))          # contains src/ (training/evaluation code)
RUNS_DIR = _get("RUNS_DIR", os.path.join(PROJECT_ROOT, "runs"))      # runs/<model>_fold<k>/eval/predictions.pkl


# ── External paths (read only when used) ─────────────────────────────────────
# Paths to data outside the repository, resolved only when a script asks for them
def __getattr__(name):
    if name == "LABELS_JSON":
        # labeltoname.json: the 10 habitat names
        return os.environ.get("LABELS_JSON") or os.path.join(PROJECT_ROOT, "configs", "labeltoname.json")
    if name == "DIMORPHISM_CSV":
        # dimorphism_species_{back,belly}.csv
        return os.path.join(os.environ.get("DIMORPHISM_DIR") or os.path.join(PROJECT_ROOT, "data", "dimorphism"),
                            "dimorphism_species_{}.csv")
    if name == "DATASET_DIR":
        return _get("DATASETS_DIR")                                  # contains Familly_split_<condition>/
    raise AttributeError(name)
