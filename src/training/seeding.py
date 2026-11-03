from __future__ import annotations

import random

import numpy as np
import torch


# ── Seeding ──────────────────────────────────────────────────────────────────
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    # One thread per worker: torch's default thread pool only adds contention
    torch.set_num_threads(1)
