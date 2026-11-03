"""Paths used by the report scripts, read from the .env file at the repository root
(see .env.example there).

Paths inside this repository default to locations relative to it; paths to data
outside the repository (habitat names, dimorphism tables, image dataset) must be
set in .env or as environment variables.
"""
import os

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


def _get(key, default=None):
    val = os.environ.get(key, default)
    if val is None:
        raise SystemExit(f"{key} is not set: copy .env.example to .env at the repository root and fill it in.")
    return val


PROJECT_ROOT = _get("PROJECT_ROOT", os.path.dirname(HERE))          # contains src/ (training/evaluation code)
RUNS_DIR = _get("RUNS_DIR", os.path.join(PROJECT_ROOT, "runs"))      # runs/<model>_fold<k>/eval/predictions.pkl


def __getattr__(name):
    """External paths: only required by the scripts that use them."""
    if name == "LABELS_JSON":
        # labeltoname.json: the 10 habitat names (default: inside BIRD_PROJECT_DIR)
        return os.environ.get("LABELS_JSON") or os.path.join(_get("BIRD_PROJECT_DIR"), "configs", "labeltoname.json")
    if name == "DIMORPHISM_CSV":
        # dimorphism_species_{back,belly}.csv (default: inside BIRD_PROJECT_DIR)
        return os.path.join(os.environ.get("DIMORPHISM_DIR") or _get("BIRD_PROJECT_DIR"), "dimorphism_species_{}.csv")
    if name == "DATASET_DIR":
        return _get("DATASETS_DIR")                                  # contains Familly_split_<condition>/
    raise AttributeError(name)
