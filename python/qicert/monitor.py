"""Syndrome-shadow runtime monitor.

Layer 1: protocol-data hygiene on the Pauli manifold — syndrome state
         change between inference steps = syndromic interrupt (gray-list).
Layer 2: classical-shadows median-of-means statistics on continuous
         latents, theorem-budgeted false alarms (Huang-Kueng-Preskill).
Fused: associative-cleanup codes + conformal alarm gate + Lipschitz-safe
fallback.
"""
from __future__ import annotations

import numpy as np


class SyndromeShadowMonitor:
    """Two-layer runtime guard on the action stream."""

    def __init__(self, num_shadows: int = 64, sketch_size: int = 32,
                 conformal_fpr: float = 0.01):
        self.num_shadows = num_shadows
        self.sketch_size = sketch_size
        self.conformal_fpr = conformal_fpr
        self._projections: np.ndarray | None = None

    def syndrome(self, compiled_families) -> int:
        """PDU hygiene: the syndrome is the compiled Clifford table's readout."""
        raise NotImplementedError("syndrome readout lands with pauli_family (N7).")

    def shadow_statistics(self, x: np.ndarray) -> np.ndarray:
        """s_i(x) = <G_i, x> over M random projection directions."""
        if self._projections is None:
            rng = np.random.default_rng(0)
            self._projections = rng.standard_normal(
                (self.num_shadows, x.shape[-1]))
        return self._projections @ x
