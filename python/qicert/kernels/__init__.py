"""Backend registry for the qicert kernel set.

Decision (2026-08-11): v1 ships a pure-Python reference implementation of the
four kernels (``python_backend``); C++/CUDA-Q (``cpp_backend``, Q19 pin) and
Julia SumOfSquares (``julia_backend``, Q22 bridge) plug in behind the same
contracts and must pass parity tests against the Python reference. The Python
implementation is the conformance spec.

Select a backend:

    from qicert.kernels import get_backend
    k = get_backend()            # active backend (default "python")
    k = get_backend("python")
    set_active_backend("cpp")    # once the Q19 pin lands

The CLI flag ``qicert.bench.all --backend {python,cpp}`` and
``--sos-backend {python,julia}`` set the active backends before bench modules
run (see ``bench/all.py``).
"""
from __future__ import annotations

from typing import Callable

from .base import KernelError

_BACKENDS: dict[str, type] = {}
_SOS_BACKENDS: dict[str, type] = {}
_active: str = "python"
_sos_active: str = "python"


class _SosPythonStub:
    """v1 placeholder for the Layer-2a SOS solver.

    The real SOS backend is the Julia SumOfSquares bridge (Q22). The stub
    keeps the registry/default semantics honest until that lands.
    """

    name = "python"

    def sos_boxes(self, *args, **kwargs):
        from .base import KernelError
        raise KernelError("Layer-2a SOS boxes land with the Julia SumOfSquares bridge (Q22).")


_SOS_BACKENDS["python"] = _SosPythonStub


def register_backend(name: str, cls: type) -> None:
    """Register a kernel backend implementation."""
    _BACKENDS[name] = cls


def register_sos_backend(name: str, cls: type) -> None:
    """Register a Layer-2a SOS solver backend."""
    _SOS_BACKENDS[name] = cls


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def set_active_backend(name: str) -> None:
    global _active
    if name not in _BACKENDS:
        raise KernelError(f"unknown kernel backend '{name}' (have: {available_backends()})")
    _active = name


def set_active_sos_backend(name: str) -> None:
    global _sos_active
    if name not in _SOS_BACKENDS:
        raise KernelError(f"unknown SOS backend '{name}' (have: {sorted(_SOS_BACKENDS)})")
    _sos_active = name


def get_backend(name: str | None = None):
    """Return the (active or named) kernel backend instance."""
    key = name or _active
    if key not in _BACKENDS:
        raise KernelError(f"unknown kernel backend '{key}' (have: {available_backends()})")
    return _BACKENDS[key]()


def get_sos_backend(name: str | None = None):
    """Return the (active or named) Layer-2a SOS solver backend instance."""
    key = name or _sos_active
    if key not in _SOS_BACKENDS:
        raise KernelError(f"unknown SOS backend '{key}' (have: {sorted(_SOS_BACKENDS)})")
    return _SOS_BACKENDS[key]()


def active_backend_name() -> str:
    return _active


def active_sos_backend_name() -> str:
    return _sos_active


# --- parity harness ---------------------------------------------------------
def parity_check(backend: str | None = None) -> dict[str, float]:
    """Run each kernel once and return a deterministic fingerprint.

    Every backend must reproduce these fingerprints within the documented
    tolerances (see tests/test_kernels.py::test_parity_harness). The python
    reference values are the conformance spec for the C++/Julia ports.
    """
    import numpy as np

    k = get_backend(backend)
    rng = np.random.default_rng(7)

    # 1. compression: exact-TT matrix 16x16, ranks (2,)
    cores0 = [rng.normal(size=(1, 4, 4, 2)), rng.normal(size=(2, 4, 4, 1))]
    W = k.contract_cores(cores0, (4, 4), (4, 4))
    cs = k.tt_svd(W, (4, 4), (4, 4), ranks=(2,))
    rec = k.contract_cores(cs.arrays, cs.mode_dims, (4, 4))
    rel_err = float(np.linalg.norm(rec - W) / np.linalg.norm(W))

    # 2. lipschitz: product bound vs tight per-layer norm vs dense norm
    prod = k.lipschitz_product(cs.arrays)
    tight = k.operator_norm_tight(cs.arrays, (4, 4), (4, 4))
    dense = float(np.linalg.norm(W, 2))

    # 3. compiler: a genuinely commuting 2-qubit family (exact identity)
    fam = [(1.0, "ZZ"), (0.5, "ZI"), (0.25, "IZ")]
    pg = k.pauli_grouping([(c, s) for c, s in fam], k=2)
    dg = k.pauli_diagonalize(pg.families[0], k=2)
    U = dg.unitary
    T = sum(c * k.pauli_matrix(s, 2) for c, s in fam)
    ident = float(np.linalg.norm(U @ np.diag(np.diag(U.conj().T @ T @ U)) @ U.conj().T - T))

    # 4. iqae: known p = 0.1, moderate budget
    p_true = 0.1
    def measure(depth: int, shots: int, _rng=np.random.default_rng(3)) -> int:
        th = float(np.arcsin(np.sqrt(p_true)))
        prob = np.clip(np.sin((2 * depth + 1) * th) ** 2, 1e-12, 1 - 1e-12)
        return int(_rng.binomial(shots, prob))
    iv = k.iqae(measure, budget=150, shots=30, seed=0)
    contains = 1.0 if (iv.lo <= p_true <= iv.hi) else 0.0

    return {
        "tt_svd_rel_err": rel_err,
        "lipschitz_tight_vs_dense": abs(tight - dense) / max(dense, 1e-12),
        "pauli_identity_err": ident,
        "iqae_interval_contains_p": contains,
    }


# Lazy-import the reference implementation so that importing ``qicert.kernels``
# stays dependency-light (numpy/scipy only; never torch).
from . import python_backend  # noqa: E402  (registers "python")

__all__ = [
    "KernelError",
    "register_backend",
    "register_sos_backend",
    "available_backends",
    "set_active_backend",
    "set_active_sos_backend",
    "get_backend",
    "get_sos_backend",
    "active_backend_name",
    "active_sos_backend_name",
    "parity_check",
]
