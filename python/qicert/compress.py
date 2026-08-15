"""Compression entry points (N2, N2', N3).

- N2  : {TT, QTT} x TT-cross sweep — 6 bond plans x 2 backbones
- N2' : bit-ordering sensitivity — 3 orderings, 1 layer, 1 seed (Q20)
- N3  : Layer-1 exact Lipschitz table

These thin wrappers delegate to the active kernel backend (``qicert.kernels``),
so the same function serves the Python reference, the C++/CUDA-Q port (Q19),
and any future backend — the math is the conformance spec.
"""
from __future__ import annotations

import math

import numpy as np

from .kernels import get_backend, active_backend_name


def _split_factors(n: int, d: int) -> tuple[int, ...]:
    """Split n into d factors as evenly as possible (product == n).

    Used to derive TT-mode shapes from a flat layer dimension. QTT bit-order
    sweeps (N2') override this with explicit mode factors per layer type.
    """
    if d <= 1:
        return (n,)
    a = int(round(n ** (1.0 / d)))
    a = max(1, a)
    # walk down from the d-th root until it divides n
    while n % a != 0 and a > 1:
        a -= 1
    if a <= 1:
        return _split_factors(n, d - 1) + (1,)
    return (a,) + _split_factors(n // a, d - 1)


def tt_cross(matrix: np.ndarray, rank: int,
             mode_factors: tuple[int, ...] | None = None) -> list[np.ndarray]:
    """TT-cross + maxvol decomposition (T2): cores built from entry queries.

    ``mode_factors``: optional row-mode factorization ``(m_1..m_d)``; the
    column modes are split with the same number of factors. Defaults to a
    2-mode split (the QTT bit-order sweep supplies explicit factors).
    """
    matrix = np.asarray(matrix, dtype=float)
    k = get_backend()
    m, n = matrix.shape
    if mode_factors is None:
        m_dims = _split_factors(m, 2)
        d = len(m_dims)
        n_dims = _split_factors(n, d)
    else:
        m_dims = tuple(int(x) for x in mode_factors)
        d = len(m_dims)
        n_dims = _split_factors(n, d)
    ranks = tuple(int(rank) for _ in range(max(0, d - 1)))
    cores = k.tt_cross(matrix, m_dims, n_dims, ranks)
    return cores.arrays


def lipschitz_constant(cores: list[np.ndarray]) -> float:
    """Exact Layer-1 global Lipschitz constant: L = prod_i ||G_i||_2 (N3)."""
    return float(get_backend().lipschitz_product(cores))


def operator_norm_tight(cores: list[np.ndarray], m_dims, n_dims,
                        iters: int = 60) -> float:
    """Tight per-layer operator norm via power iteration (tightness ratio)."""
    return float(get_backend().operator_norm_tight(cores, m_dims, n_dims, iters))


def backend() -> str:
    """Name of the active kernel backend (for the bench header / tagging)."""
    return active_backend_name()
