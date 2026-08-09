"""Seeded reversible scenario oracle (N6 / Pillar C).

Because seeds fully determine execution (challenge Sec. 5.2), the scenario
generator is a reversible deterministic function of the seed — exactly what
amplitude estimation needs as an oracle:

    O : seed -> scenario -> rollout(policy) -> rho -> [rho < 0]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class OracleConfig:
    """Documented state size, branching, and reversibility per Sec. 5.3."""
    track: str = "AD"
    num_scenarios: int = 200
    seed_bits: int = 32
    reversible: bool = True


class ReversibleOracle:
    """Deterministic seed -> failure-indicator map, wrapped for IQAE."""

    def __init__(self, rollout: Callable[[int], float], config: OracleConfig):
        self._rollout = rollout          # seed -> robustness rho
        self._rng = np.random.default_rng(0)
        self.config = config

    def __call__(self, seed: int) -> int:
        """Return 1 if the seeded rollout violates the spec (rho < 0)."""
        return int(self._rollout(seed) < 0.0)

    def query_budget(self, queries: int) -> dict:
        """Per-benchmark oracle-call cost model (printed in the report)."""
        return {"oracle_queries": queries, "per_query": self.config}
