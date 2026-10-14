"""Which sub-networks receive gradient at a training step (the selected widths ``A_tau`` of Equation 8).

Two families of rules are supported:

* ``schedule``: a deterministic per-width firing rule ``"M/N"`` (the width trains at global step
  ``t`` when ``t % N == M % N``; ``"1/1"`` trains every step and ``"0"`` never). Round-robin
  sampling of one sub-network per step over five widths is ``["1/5", "2/5", "3/5", "4/5", "5/5"]``.
* stochastic strategies drawn from a generator seeded by ``(seed, step)``, so the choice is identical
  on every rank and continues seamlessly after a resume:
  ``all`` (every trained width), ``max``, ``random`` (one width uniformly at random),
  ``sandwich`` (smallest, largest and one random width in between) and ``max_random``
  (largest and one random other width).
"""

from __future__ import annotations

import torch

STRATEGIES = ("all", "max", "random", "sandwich", "max_random", "schedule")


def parse_step_rule(entry) -> tuple[int, int] | None:
    """``"M/N"`` -> ``(M, N)``; ``"0"``, ``0``, ``""`` or ``None`` -> never; a bare ``k`` -> ``(0, k)``."""
    if entry is None:
        return None
    if isinstance(entry, bool):
        raise ValueError(f"invalid schedule entry {entry!r}")
    if isinstance(entry, (int, float)):
        k = int(entry)
        if k < 0:
            raise ValueError(f"schedule entries must be >= 0, got {entry!r}")
        return None if k == 0 else (0, k)
    text = str(entry).strip()
    if text in ("0", ""):
        return None
    if "/" in text:
        a, b = text.split("/", 1)
        m, n = int(a), int(b)
        if n < 1 or m < 0:
            raise ValueError(f"schedule entry 'M/N' needs N >= 1 and M >= 0, got {text!r}")
        return (m, n)
    k = int(text)
    return None if k == 0 else (0, k)


class WidthSelector:
    """Selects the trained widths that receive gradient at each (1-indexed) global step."""

    def __init__(self, trained: list[int], strategy: str = "all", schedule: dict[int, tuple[int, int] | None] | None = None,
                 seed: int = 0) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"selection strategy must be one of {STRATEGIES}, got {strategy!r}")
        if not trained:
            raise ValueError("at least one width must be trained")
        self.trained = sorted(int(s) for s in trained)
        self.strategy = strategy
        self.seed = int(seed)
        self.schedule = None
        if strategy == "schedule":
            if schedule is None:
                raise ValueError("strategy 'schedule' needs a per-width schedule")
            self.schedule = {int(s): schedule.get(int(s)) for s in self.trained}
            if all(rule is None for rule in self.schedule.values()):
                raise ValueError("the schedule never trains any width")
        if strategy in ("sandwich", "max_random") and len(self.trained) < 2:
            raise ValueError(f"strategy {strategy!r} needs at least two trained widths")

    def _generator(self, step: int) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(self.seed * 1_000_003 + int(step))

    def __call__(self, step: int) -> list[int]:
        pool = self.trained
        lo, hi = pool[0], pool[-1]
        if self.strategy == "all":
            return list(pool)
        if self.strategy == "max":
            return [hi]
        if self.strategy == "schedule":
            fired = [s for s in pool if self.schedule[s] is not None
                     and step % self.schedule[s][1] == self.schedule[s][0] % self.schedule[s][1]]
            return fired or [hi]
        gen = self._generator(step)
        if self.strategy == "random":
            return [pool[int(torch.randint(len(pool), (1,), generator=gen))]]
        if self.strategy == "sandwich":
            middle = [s for s in pool if s not in (lo, hi)]
            chosen = [lo, hi]
            if middle:
                chosen.append(middle[int(torch.randint(len(middle), (1,), generator=gen))])
            return sorted(chosen)
        others = [s for s in pool if s != hi]  # max_random
        return sorted([hi, others[int(torch.randint(len(others), (1,), generator=gen))]])

