from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from itertools import chain

load_dotenv()
# ── Paths from environment ───────────────────────────────────────────────────
SOURCE_IMG = os.environ["DATASETS_DIR"]   # Directory containing bird images (.png)
SOURCE_IMG_BACK = os.path.join(SOURCE_IMG, "Familly_224_version", "V0_NEW_Segmented-Aves-Back-224-RGBA")
SOURCE_IMG_SIDE = os.path.join(SOURCE_IMG, "Familly_224_version", "V0_NEW_Segmented-Aves-Side-224-RGBA")
SOURCE_IMG_BELLY = os.path.join(SOURCE_IMG,"Familly_224_version", "V0_NEW_Segmented-Aves-Belly-224-RGBA")
CONFIG_DIR = Path("configs")
DOCS_DIR = Path("docs")
DATASET_DIR = Path("data/datasets")
RAW_DIR = Path("data/raw")
EBIRD_CSV  = RAW_DIR / "AOH_Jung_lvl1_fraction.csv"  # eBird habitat label data
CROSS_CSV  = RAW_DIR / "crosswalk_image_habitat_v0.csv"   # Crosswalk: image species → eBird species + family

VERSION = "AOH_1"  # Output filename for the processed dataset


# ── Label columns in the eBird CSV and their renamed counterparts ────────────
LABEL_COLS  = ["1", "2", "3", "4", "8", "12", "14", "9_11", "5_13_15", "6_7"]

# Priority order for seasons: breeding/resident takes precedence over non-breeding
SEASON_ORDERS = {"BREEDING-RESIDENT": 0, "NONBREEDING": 1}

# ── Load CSVs ────────────────────────────────────────────────────────────────
df_cross = pd.read_csv(CROSS_CSV, sep=";")
df_ebird = pd.read_csv(EBIRD_CSV, sep=";")

# ── Build crosswalk lookups from image species name → eBird name / family ────
img_to_ebird = dict(zip(
    df_cross["sp_image"].str.strip().str.lower(),
    df_cross["sp_habitat"].str.strip().str.lower()
))

img_to_fam = dict(zip(
    df_cross["sp_image"].str.strip().str.lower(),
    df_cross["family"].str.strip().str.lower()
))

# ── Deduplicate eBird rows, keeping the highest-priority season per species ───
df_ebird["species_key"]      = df_ebird["Species_Name"].str.strip().str.lower()
df_ebird["season_priority"]  = df_ebird["Seasonal"].map(SEASON_ORDERS).fillna(99)
df_ebird = (df_ebird.sort_values("season_priority")
                    .drop_duplicates(subset="species_key", keep="first"))

# Index by species for O(1) label lookups
ebird_dict = df_ebird.set_index("species_key")[LABEL_COLS].to_dict("index")

# ── Build one record per image ───────────────────────────────────────────────
img_paths = [
    f
    for f in chain(
        os.listdir(SOURCE_IMG_BACK),
        os.listdir(SOURCE_IMG_SIDE),
        os.listdir(SOURCE_IMG_BELLY),
    )
    if f.endswith(".png")
]

records = []
missing_img = []
for name in img_paths:
    # Filename format: <genre>_<species>_<extra>_<sexe>.png
    parts   = name.replace(".png", "").split("_")
    genre   = parts[0] if len(parts) > 0 else ""
    species = (" ".join(parts[:2])).strip().lower() if len(parts) > 1 else ""
    sexe    = parts[3] if len(parts) > 3 else ""

    # Look up labels directly, or via the crosswalk if the image name differs from eBird name
    labels = None
    if species in ebird_dict:
        labels = ebird_dict[species]
    elif img_to_ebird[species] in ebird_dict:
        labels = ebird_dict[img_to_ebird[species]]
    else :
        missing_img.append(name)
        continue  # Skip this image if no labels are found

    fam = img_to_fam.get(species, "unknown")
    if fam == "unknown":
        raise ValueError(f"Family not found for species '{species}' in image '{name}'")

    r = {
        "name_of_img": name,
        "genre":       genre,
        "species":     species,
        "sexe":        sexe,
        "family":      fam,
    }
    # Normalize decimal separator (comma → dot) for all label columns
    for c in LABEL_COLS:
        r[c] = labels[c].replace(",", ".")
    records.append(r)

# ── Write output ─────────────────────────────────────────────────────────────
path_out_df = os.path.join(DATASET_DIR, f"dataset_version_{VERSION}.csv")
if os.path.exists(path_out_df):
    raise FileExistsError(f"Output file '{path_out_df}' already exists. Please remove it before running the script.")

df = pd.DataFrame(records)
df.to_csv(path_out_df, index=False)


path_out_json = os.path.join(CONFIG_DIR,"labelname.json")
if not os.path.exists(path_out_json):
    with open(path_out_json, "w") as f:
        json.dump(LABEL_COLS, f, indent=2)


# ── Generate documentation ───────────────────────────────────────────────────
DOCS_DIR.mkdir(parents=True, exist_ok=True)
doc_path = DOCS_DIR / f"dataset_v{VERSION}.md"

n_records  = len(df)
n_missing  = len(missing_img)
n_species  = df["species"].nunique()
n_families = df["family"].nunique()
n_genres   = df["genre"].nunique()

with open(doc_path, "w") as f:
    f.write(f"# Dataset V{VERSION}\n\n")

    f.write("## Sources\n")
    f.write(f"| File | Path |\n|---|---|\n")
    f.write(f"| Label source | `{EBIRD_CSV.name}` |\n")
    f.write(f"| Metadata | `{CROSS_CSV.name}` |\n\n")

    f.write("## Statistics\n")
    f.write(f"| Metric | Value |\n|---|---|\n")
    f.write(f"| Total records | {n_records:,} |\n")
    f.write(f"| Images not matched | {n_missing:,} |\n")
    f.write(f"| Unique species | {n_species:,} |\n")
    f.write(f"| Unique families | {n_families:,} |\n")
    f.write(f"| Unique genres | {n_genres:,} |\n\n")

    f.write("## Label Columns\n")
    f.write(", ".join(f"`{c}`" for c in LABEL_COLS) + "\n")

print(f"Dataset built: {n_records:,} records, {n_missing:,} skipped. Doc written to {doc_path}")