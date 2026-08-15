"""Kernel contracts shared by every backend (python reference, C++/CUDA-Q,
Julia SOS). Backends implement the same signatures; the python reference is
the conformance spec (parity tests in ``tests/test_kernels.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class KernelError(RuntimeError):
    """Raised by backends that are registered but not yet implemented."""


@dataclass
class Cores:
    """TT/QTT matrix cores.

    Each core is a 4-way array of shape ``(r_{k-1}, m_k, n_k, r_k)``; a
    single-mode matrix factor may be stored as a 2-way array ``(m, n)`` with
    implicit unit bonds. Contracting the bond indices reconstructs the weight
    matrix ``W`` of shape ``(prod(m_k), prod(n_k))``.
    """

    arrays: list[np.ndarray]
    mode_dims: tuple[int, ...]            # row-mode factors (m_1 .. m_d)
    ranks: tuple[int, ...]                # internal bond dimensions (r_1 .. r_{d-1})
    source: str = "python"                # backend that produced the cores

    @property
    def d(self) -> int:
        return len(self.arrays)

    @property
    def row_dims(self) -> tuple[int, ...]:
        return self.mode_dims

    @property
    def col_dims(self) -> tuple[int, ...]:
        # n_k is axis 2 of each core (or axis 1 for 2-way cores)
        out = []
        for g in self.arrays:
            if g.ndim == 4:
                out.append(g.shape[2])
            elif g.ndim == 3:
                out.append(g.shape[1])
            else:
                out.append(g.shape[1])
        return tuple(out)


@dataclass
class PauliGrouping:
    """Greedy commutation grouping + per-family pruning certificates."""

    families: list[list[tuple[float, str]]]  # list of families of (coef, pauli)
    pruning_cost: list[float]                # ||sum_{a in F_j} c_a||_1 per family
    n_families: int

    def __post_init__(self):
        self.n_families = len(self.families)


@dataclass
class Diagonalization:
    """Joint eigenbasis of a commuting Pauli family (the 'single Clifford
    frame' realized concretely as a unitary for small k)."""

    unitary: np.ndarray                     # (2^k, 2^k) joint eigenbasis
    eigenvalues: list[np.ndarray]           # per string: diagonal entries (+-1)


@dataclass
class IQAEInterval:
    """Bayesian IQAE result: posterior over p = sin^2(theta)."""

    p_map: float
    lo: float
    hi: float
    n_queries: int
    max_depth: int
    grid: np.ndarray = field(repr=False)      # theta grid
    posterior: np.ndarray = field(repr=False)  # posterior density over theta

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        return (self.lo, self.hi)


@dataclass
class ShadowVerdict:
    """Runtime monitor verdict for one inference step."""

    statistic: float
    threshold: float
    alarm: bool
    detail: str = ""


class KernelSet:
    """The interface every kernel backend implements.

    v1 (python reference) implements all methods; cpp/julia backends raise
    KernelError until their environment pins land (Q19 / Q22).
    """

    name: str = "base"

    # -- Pillar A: compression + Layer-1 certificates ---------------------
    def tt_svd(self, matrix: np.ndarray, m_dims: tuple[int, ...],
               n_dims: tuple[int, ...], ranks: tuple[int, ...]) -> Cores:
        raise NotImplementedError

    def tt_cross(self, matrix: np.ndarray, m_dims: tuple[int, ...],
                 n_dims: tuple[int, ...], ranks: tuple[int, ...]) -> Cores:
        raise NotImplementedError

    def contract_cores(self, cores: list[np.ndarray], m_dims, n_dims) -> np.ndarray:
        raise NotImplementedError

    def lipschitz_product(self, cores: list[np.ndarray]) -> float:
        raise NotImplementedError

    def operator_norm_tight(self, cores: list[np.ndarray], m_dims, n_dims,
                            iters: int = 60) -> float:
        raise NotImplementedError

    # -- Pillar B: commuting-Pauli compiler --------------------------------
    def pauli_grouping(self, table: list[tuple[float, str]], k: int) -> PauliGrouping:
        raise NotImplementedError

    def pauli_matrix(self, pauli: str, k: int) -> np.ndarray:
        raise NotImplementedError

    def pauli_diagonalize(self, family: list[tuple[float, str]], k: int) -> Diagonalization:
        raise NotImplementedError

    # -- Pillar C: Bayesian IQAE -------------------------------------------
    def iqae(self, measure, budget: int = 200, shots: int = 30,
             grid: int = 4096, seed: int = 0) -> IQAEInterval:
        raise NotImplementedError

    # -- Monitor: syndrome-shadow runtime guard ----------------------------
    def shadow_statistics(self, x: np.ndarray, projections: np.ndarray,
                          baseline_mean: float, baseline_std: float,
                          blocks: int = 8, fpr: float = 0.01) -> ShadowVerdict:
        raise NotImplementedError

    def shadow_syndrome(self, diag_patterns: list[np.ndarray]) -> int:
        raise NotImplementedError
