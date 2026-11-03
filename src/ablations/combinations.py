from __future__ import annotations

from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .common import ALL_CODES, Condition, is_admissible

# ── Admissible combinations ──────────────────────────────────────────────────
K_PROBABILITIES = {0: 0.20, 1: 0.40, 2: 0.30, 3: 0.10}


# Admissible ablation pools by number of ablations (0, 1, 2, 3)
def admissible_by_size() -> Dict[int, List[frozenset]]:
    by_size: Dict[int, List[frozenset]] = {k: [] for k in K_PROBABILITIES}
    for k in K_PROBABILITIES:
        for combo in combinations(ALL_CODES, k):
            # c'est un set imuable 
            codes = frozenset(combo)
            if is_admissible(codes):
                by_size[k].append(codes)
    return by_size


_ADMISSIBLE_BY_SIZE = admissible_by_size()


# ── Drawing a condition ──────────────────────────────────────────────────────
# Draw a condition; enabled_codes restricts the draw (demos only: it changes the marginals)
def draw_condition(rng: np.random.Generator, enabled_codes: Optional[FrozenSet[str]] = None) -> Condition:
    sizes = list(K_PROBABILITIES.keys())
    probs = list(K_PROBABILITIES.values())
    # Choisi la pool d'ablation aléatoirement entre 0 et 3 (en utilisant K_PROBABILITIES)
    k = int(rng.choice(sizes, p=probs))
    pool = _ADMISSIBLE_BY_SIZE[k]
    # Choisi la combinaison d'ablation aléatoirement dans le pool choisi précedemment 
    codes = pool[int(rng.integers(len(pool)))]
    # Condition est un objet qui stock les conditions selectionné
    condition = Condition(active=codes)

    if enabled_codes is None or condition.active <= enabled_codes:
        return condition

    for _ in range(1000):
        condition = draw_condition(rng)
        if condition.active <= enabled_codes:
            return condition
    raise RuntimeError(
        f"couldn't draw an admissible condition using only {sorted(enabled_codes)} in 1000 tries"
    )


# ── All conditions and inclusion probabilities ───────────────────────────────
# All 36 admissible conditions
def all_admissible_conditions() -> List[Condition]:
    conditions = []
    for k in sorted(_ADMISSIBLE_BY_SIZE):
        for codes in _ADMISSIBLE_BY_SIZE[k]:
            conditions.append(Condition(active=codes))
    return conditions


# P(ablation active in a random bag) per code (0.140 for the CIELAB ones, 0.246 for A3/A4/A5)
def marginal_inclusion_probabilities() -> Dict[str, float]:
    totals = {code: 0.0 for code in ALL_CODES}
    for k, pool in _ADMISSIBLE_BY_SIZE.items():
        if not pool:
            continue
        p_each_combo = K_PROBABILITIES[k] / len(pool)
        for codes in pool:
            for code in codes:
                totals[code] += p_each_combo
    return totals
