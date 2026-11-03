from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..ablations.common import Condition

# ── Fixed evaluation values ──────────────────────────────────────────────────
A3_SIDES = (56, 16, 4, 1)  # coarsest -> pixel-by-pixel, roughly geometric
A3_REALIZATIONS = (1, 2, 3)
A4_ANGLES = (0, 90)
A4_LENGTHS = (17, 41, 101)  # the two bounds and the median of the log-uniform law


# ── Evaluation conditions ────────────────────────────────────────────────────
@dataclass(frozen=True)
class EvalCondition:
    name: str
    condition: Condition
    a3_side: Optional[int] = None
    a3_realization: Optional[int] = None
    a4_params: Optional[Dict[str, int]] = None


def fixed_conditions() -> List[EvalCondition]:
    conditions = [EvalCondition("reference", Condition())]
    for code in ("A1", "A2", "A5", "A7"):
        conditions.append(EvalCondition(code, Condition(active=frozenset({code}))))

    for side in A3_SIDES:
        for realization in A3_REALIZATIONS:
            conditions.append(
                EvalCondition(
                    f"A3_h{side}_r{realization}",
                    Condition(active=frozenset({"A3"})),
                    a3_side=side,
                    a3_realization=realization,
                )
            )

    for angle in A4_ANGLES:
        for length in A4_LENGTHS:
            conditions.append(
                EvalCondition(
                    f"A4_theta{angle}_L{length}",
                    Condition(active=frozenset({"A4"})),
                    a4_params={"theta": angle, "length": length},
                )
            )

    return conditions
