from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Optional, Tuple


# ── Adaptive loss balancer ───────────────────────────────────────────────────
class AdaptiveLossBalancer:
    def __init__(
        self,
        names: Tuple[str, str] = ("kl", "bce"),
        ema_decay: float = 0.9,
        lag_epochs: int = 5,
        temperature: float = 0.1,
        weight_band: Tuple[float, float] = (0.2, 0.8),
    ):
        if len(names) != 2:
            raise ValueError("AdaptiveLossBalancer only supports exactly two loss terms")
        if not 0.0 < ema_decay < 1.0:
            raise ValueError("ema_decay must lie in (0, 1)")
        if lag_epochs < 1:
            raise ValueError("lag_epochs must be >= 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        lo, hi = weight_band
        if not (0.0 <= lo < 0.5 < hi <= 1.0) or abs((lo + hi) - 1.0) > 1e-9:
            raise ValueError("weight_band must be a (low, high) pair symmetric around 0.5, e.g. (0.2, 0.8)")

        self.names = names
        self.ema_decay = ema_decay
        self.lag_epochs = lag_epochs
        self.temperature = temperature
        self.weight_band = weight_band

        self._ema: Dict[str, Optional[float]] = {name: None for name in names}
        # Last lag_epochs + 1 EMA values per term: [0] and [-1] are lag_epochs apart
        self._history: Dict[str, Deque[float]] = {name: deque(maxlen=lag_epochs + 1) for name in names}

    # For checkpoints: a resumed run keeps the same EMA and lag history
    def state_dict(self) -> Dict[str, object]:
        return {
            "ema": dict(self._ema),
            "history": {name: list(hist) for name, hist in self._history.items()},
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self._ema = dict(state["ema"])
        for name, values in state["history"].items():
            self._history[name] = deque(values, maxlen=self.lag_epochs + 1)

    # Once per epoch, with that epoch's mean loss per term; returns the next epoch's weights
    def update(self, raw_losses: Dict[str, float]) -> Dict[str, float]:
        for name in self.names:
            raw = raw_losses[name]
            prev = self._ema[name]
            ema = raw if prev is None else self.ema_decay * prev + (1.0 - self.ema_decay) * raw
            self._ema[name] = ema
            self._history[name].append(ema)

        a, b = self.names
        if len(self._history[a]) <= self.lag_epochs:
            return {a: 0.5, b: 0.5}

        r_a = self._history[a][-1] / self._history[a][0]
        r_b = self._history[b][-1] / self._history[b][0]

        # Two-term softmax(r / T) written as a sigmoid, for numerical stability
        w_a = 1.0 / (1.0 + math.exp(-(r_a - r_b) / self.temperature))

        lo, hi = self.weight_band
        w_a = min(max(w_a, lo), hi)
        return {a: w_a, b: 1.0 - w_a}
