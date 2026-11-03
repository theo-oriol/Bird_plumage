from __future__ import annotations

import hashlib

import numpy as np


# ── Derived RNG ──────────────────────────────────────────────────────────────
def derive_rng(*parts: object) -> np.random.Generator:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    seed = int.from_bytes(digest, "big")
    return np.random.default_rng(seed)
