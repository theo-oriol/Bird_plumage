from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional

# ── Ablation codes ───────────────────────────────────────────────────────────
# Mutually exclusive CIELAB ablations: at most one per bag, replaces the preparation step
CIELAB_EXCLUSIVE_GROUP: FrozenSet[str] = frozenset({"A1", "A2", "A6", "A7"})

ALL_CODES = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")

# Where each ablation acts: 'image' = whole canvas before cropping, 'crop' = each 224 px crop
INSERTION_LEVEL = {
    "A1": "image",  # step 0 — deterministic given the specimen, cached, replaces the source read
    "A2": "image",  # step 0 — same reasoning, cached from the start
    "A3": "crop",   # step 7, last spatial operation
    "A4": "image",  # step 4, before cropping (needs real neighbourhood)
    "A5": "crop",   # step 8, last intensity operation
    "A6": "image",  # step 1 — the only CIELAB-exclusive ablation still computed online (category drawn per sac)
    "A7": "image",  # step 0 — deterministic given the specimen, cached, replaces the source read
}

# Parameters drawn once per bag (shared by the 16 crops) or once per crop
DRAWN_PER = {
    "A1": None,
    "A2": None,
    "A3": "crop",
    "A4": "sac",
    "A5": "crop",
    "A6": "sac",
    "A7": None,
}


# ── Condition ────────────────────────────────────────────────────────────────
# Set of active ablations for one bag; empty = reference. The chain applies them in a fixed order
@dataclass(frozen=True)
class Condition:
    active: FrozenSet[str] = field(default_factory=frozenset)

    def __contains__(self, code: str) -> bool:
        return code in self.active

    def __len__(self) -> int:
        return len(self.active)

    def __str__(self) -> str:
        return "reference" if not self.active else "+".join(sorted(self.active))

    # The active CIELAB-exclusive ablation, if any (@property : c.cielab_member sans parenthèses)
    @property
    def cielab_member(self) -> Optional[str]:
        hit = self.active & CIELAB_EXCLUSIVE_GROUP
        return next(iter(hit)) if hit else None


# Admissible iff at most one of the four mutually exclusive CIELAB ablations
def is_admissible(codes: FrozenSet[str]) -> bool:
    return len(codes & CIELAB_EXCLUSIVE_GROUP) <= 1
