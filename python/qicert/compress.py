"""Compression entry points (N2, N2', N3).

- N2  : {TT, QTT} x TT-cross sweep — 6 bond plans x 2 backbones
- N2' : bit-ordering sensitivity — 3 orderings, 1 layer, 1 seed (Q20)
- N3  : Layer-1 exact Lipschitz table
"""
from __future__ import annotations

import numpy as np


def tt_cross(matrix: np.ndarray, rank: int,
             mode_factors: tuple[int, ...] | None = None) -> list[np.ndarray]:
    """TT-cross + maxvol decomposition (T2): cores built from entry queries.

    Phase-0 stub — returns cores of the correct length once the CUDA-Q
    kernel lands (``include/qicert/tt_cross.hpp``).
    """
    raise NotImplementedError("tt_cross kernel lands with the Q19 env pin.")


def lipschitz_constant(cores: list[np.ndarray]) -> float:
    """Exact Layer-1 global Lipschitz constant: L = prod_i ||G_i||_2 (N3)."""
    L = 1.0
    for core in cores:
        L *= np.linalg.norm(core, 2)
    return L
