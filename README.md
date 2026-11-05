# Entraînement unique multi-ablations

Implémentation du protocole `protocole_ablations_uniques_v3.pdf` : un modèle
DINOv3-S + LoRA, agrégation MIL par attention *gated*, sept ablations tirées
en ligne sur les crops. Voir le PDF pour la spécification complète ; ce
document explique comment le code s'y branche, ce qui vient d'où, et ce qui
reste ouvert.

## Démarrer

```bash
pip install -r requirements.txt   # torch 2.5.1 + CUDA testés

# chemins propres à la machine (dataset, folds, DINOv3) : à renseigner une fois
cp .env.example .env              # puis éditer .env ; configs/paths.yaml lit ses ${VARIABLES}

python scripts/check_dataset_integrity.py      # vérifie folds/crosswalk (§7)

# A1, A2, A7 sont déterministes (aucun paramètre tiré) — précalculées une
# fois hors ligne au lieu d'être recalculées à chaque sac qui les tire, à
# chaque époque. Nécessaire avant tout entraînement (chain.py lève une
# erreur claire si une de ces ablations est tirée et son cache n'existe pas).
python scripts/build_a1_cache.py --workers 8
python scripts/build_a2_cache.py --workers 8
python scripts/build_a7_cache.py --workers 8

python train.py --config configs/train_fold0.yaml
```

Le notebook `notebooks/01_ablation_catalogue.ipynb` montre l'effet de
chaque ablation sur un vrai spécimen (à ouvrir avec le kernel `torch`).
`scripts/benchmark_ablations.py` mesure ce qui est lent.


## Structure

```
src/
  color.py                 fourni pour le projet — CIELAB / colour naming
  npy_io.py                 lecture/écriture des .rgb.zst
  sampling.py                positions de crop depuis les tables .cand_*.zst
  rng.py                      graines déterministes (image, condition, ...)
  chain.py                    LA chaîne canonique (protocole §2) — ordonne les 4 points d'insertion
  ablations/
    a1_gray_lstar.py ... a7_chroma_only.py    une ablation, un fichier,
      chacun 100% autonome (numpy/scipy/skimage/zstandard seulement — voir
      "Portabilité des fichiers d'ablation" ci-dessous) : copier un seul de
      ces fichiers ailleurs suffit à le réutiliser, rien d'autre du projet
      n'est nécessaire
    common.py                  Condition, groupe exclusif CIELAB
    combinations.py             tirage §4 (36 conditions)
  datasets/
    folds.py                    fold CSVs + crosswalk + poids de classe
    bag_dataset.py                Dataset d'entraînement (tirage en ligne, §4)
    eval_dataset.py                Dataset d'évaluation (conditions fixes, §7)
  model/
    backbone.py, mil.py, heads.py, bagmodel.py, losses.py
  training/
    run.py                      la boucle d'entraînement + traçabilité
  evaluation/
    conditions.py, a6_presence.py, metrics.py, run_eval.py

configs/            un fichier YAML par run ; paths.yaml/model.yaml/optim.yaml sont inclus par les autres
scripts/             build_a{1,2,7}_cache, build_a6_presence_manifest, benchmark_ablations, check_dataset_integrity, combine_fold_results,
                      build_grayscale_dataset, build_grayscale_norm_lum_dataset
notebooks/            catalogue visuel des ablations
train.py, evaluate.py  points d'entrée
```

## Traçabilité d'un run

`python train.py --config configs/train_fold0.yaml` crée
`runs/main_fold0/` contenant :

- `config.yaml` — la config résolue (tous les `includes` fusionnés), donc
  le run est reproductible à partir de ce seul fichier ;
- `manifest.json` — horodatage, commit git (si dépôt), versions
  python/torch, GPU, poids de classe BCE calculés ;
- `metrics.csv` — une ligne par époque (loss, KL, BCE, lr, durée) ;
- `checkpoints/{last,best}.pt` — `best` = plus faible loss d'entraînement
  lissée (le seul critère utilisé pour l'arrêt anticipé, §1 — aucune
  métrique de validation n'entre dans ce choix, conformément au protocole).

