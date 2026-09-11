"""Property-based integrity tests for the certify suite (B14a/B14b/B14c).

These tests are INDEPENDENT re-implementations, not invocations of the module
under test's own helpers — that is the point: if the witness/verifier logic
disagrees with a from-scratch contraction or a known analytic answer, the
certificate pipeline is broken and must fail loudly.

Covers:
  1. Gauge invariance: bond rescaling (L on the left core, L^{-1} on the
     right core) leaves the contracted operator tensor exactly invariant.
  2. Witness round-trip + tamper detection: a witness built from random cores
     verifies; the same witness with a perturbed shipped bound must FAIL.
  3. Clopper-Pearson interval: containment, boundary cases, monotonicity.
  4. mpmath cross-check: 50-digit spectral norm of a known matrix (diag(3,1))
     matches the analytic value to ~1e-20 and the float pipeline agrees.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from qicert.certify.verify import (
    Witness,
    verify_witness,
    witness_from_dict,
    _core_norm,
)
from qicert.certify.robustness import clopper_pearson_interval


# --------------------------------------------------------------------------
# Independent helpers (deliberately NOT imported from qicert.certify)
# --------------------------------------------------------------------------

def _random_chain(m_dims, n_dims, ranks, seed: int):
    """Random TT/MPO-style chain with cores shaped (r_prev, m, n, r)."""
    rng = np.random.default_rng(seed)
    cores = []
    r_prev = 1
    for i, (mk, nk) in enumerate(zip(m_dims, n_dims)):
        rk = ranks[i]
        cores.append(rng.standard_normal((r_prev, mk, nk, rk)))
        r_prev = rk
    return cores


def _dense_operator(cores):
    """Dense matrix of the TT chain via one einsum over all cores (independent)."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    subs = []
    for k, g in enumerate(cores):
        subs.append(f"{letters[k]}{letters[10 + k]}{letters[20 + k]}{letters[k + 1]}")
    lhs = ",".join(subs)
    rhs = "".join(f"{letters[10 + k]}{letters[20 + k]}" for k in range(len(cores)))
    dense = np.einsum(lhs + "->" + rhs, *cores)
    M = int(np.prod([g.shape[1] for g in cores]))
    N = int(np.prod([g.shape[2] for g in cores]))
    return dense.reshape(M, N)


def _contract_chain(cores, m_dims, n_dims, v):
    """T @ v via the dense operator — independent of verify._contract."""
    dense = _dense_operator(cores)
    assert dense.shape == (int(np.prod(m_dims)), int(np.prod(n_dims)))
    return dense @ np.asarray(v, dtype=np.float64).ravel()


def _gauge_pair(cores, rng):
    """Apply L on core i's right bond and L^{-1} on core i+1's left bond.

    The contracted operator is invariant because the bond sum between core i
    and i+1 telescopes: sum_r G_i[.., r] L[r, s] Linv[s, a] G_{i+1}[a, ..] =
    sum_r G_i[.., r] G_{i+1}[r, ..].
    """
    i = int(rng.integers(0, len(cores) - 1))
    r = cores[i].shape[3]
    while True:
        L = rng.standard_normal((r, r))
        if abs(np.linalg.det(L)) > 1e-3:
            break
    Linv = np.linalg.inv(L)
    out = [g.copy() for g in cores]
    out[i] = np.einsum("apmn,ns->apms", out[i], L)
    out[i + 1] = np.einsum("sa,apmn->spmn", Linv, out[i + 1])
    return out


# --------------------------------------------------------------------------
# 1. Gauge invariance
# --------------------------------------------------------------------------

@given(st.lists(st.integers(min_value=2, max_value=4), min_size=2, max_size=3))
@settings(max_examples=25, deadline=None)
def test_gauge_invariance_exact(m_dims):
    n_dims = m_dims  # square chain keeps the operator map well-defined
    ranks = [max(2, d) for d in m_dims[:-1]] + [1]
    cores = _random_chain(m_dims, n_dims, ranks, seed=42)
    rng = np.random.default_rng(7)
    n = int(np.prod(n_dims))
    v = rng.standard_normal(n)

    ref = _contract_chain(cores, m_dims, n_dims, v)
    gauged = _gauge_pair(cores, rng)
    out = _contract_chain(gauged, m_dims, n_dims, v)
    assert np.allclose(ref, out, atol=1e-9, rtol=1e-9), (
        "contracted operator changed under gauge transformation")


# --------------------------------------------------------------------------
# 2. Witness round-trip + tamper detection
# --------------------------------------------------------------------------

def _make_witness(m_dims, n_dims, ranks, seed: int) -> Witness:
    cores = _random_chain(m_dims, n_dims, ranks, seed=seed)
    L = 1.0
    for g in cores:
        L *= _core_norm(g)
    return witness_from_dict({
        "layer": f"test.layer.seed{seed}",
        "pareto_ratio": 0.5,
        "lipschitz_float": float(L),
        "cores": [[[float(x) for x in row] for row in
                   g.reshape(g.shape[0] * g.shape[1], -1)] for g in cores],
        "m_dims": list(m_dims),
        "n_dims": list(n_dims),
        "bound_kind": "core_product",
    })


@given(st.lists(st.integers(min_value=2, max_value=4), min_size=2, max_size=3))
@settings(max_examples=20, deadline=None)
def test_witness_roundtrip_and_tamper(m_dims):
    n_dims = m_dims
    ranks = [max(2, d) for d in m_dims[:-1]] + [1]
    w = _make_witness(m_dims, n_dims, ranks, seed=int(np.prod(m_dims)))

    # honest witness must pass
    res = verify_witness(w)
    assert res["verdict"] == "pass", res
    assert res["match"] is True

    # tampered witness must FAIL (soundness under perturbation)
    import dataclasses
    w_bad = dataclasses.replace(w, lipschitz_float=w.lipschitz_float * 1.05)
    res_bad = verify_witness(w_bad)
    assert res_bad["verdict"] == "FAIL", res_bad


# --------------------------------------------------------------------------
# 3. Clopper-Pearson interval properties
# --------------------------------------------------------------------------

@given(st.integers(min_value=0, max_value=40), st.integers(min_value=1, max_value=40))
@settings(max_examples=60, deadline=None)
def test_cp_interval_properties(x, n_extra):
    n = x + n_extra
    lo, hi = clopper_pearson_interval(x, n)
    p = x / n
    # containment of the point estimate
    assert lo <= p + 1e-12 and p <= hi + 1e-12
    # range
    assert 0.0 <= lo <= hi <= 1.0
    # boundary behaviour
    if x == 0:
        assert lo == 0.0
    if x == n:
        assert hi == 1.0


def test_cp_interval_monotone_in_x():
    n = 100
    his = [clopper_pearson_interval(x, n)[1] for x in range(n + 1)]
    assert all(b >= a for a, b in zip(his, his[1:]))
    los = [clopper_pearson_interval(x, n)[0] for x in range(n + 1)]
    assert all(b >= a for a, b in zip(los, los[1:]))


def test_cp_zero_trials_honest():
    assert clopper_pearson_interval(0, 0) == (0.0, 1.0)


# --------------------------------------------------------------------------
# 4. mpmath cross-check against a known matrix
# --------------------------------------------------------------------------

def test_mp50_known_spectral_norm():
    import mpmath as mp
    from qicert.certify.cross_check import _core_norm_mp50
    # diag(3, 1): spectral norm exactly 3
    g = [[3.0, 0.0], [0.0, 1.0]]
    mpi = mp.mp
    val = _core_norm_mp50(g, mpi)
    assert abs(val - 3.0) < 1e-20, val
    # float pipeline agrees
    assert abs(np.linalg.norm(np.asarray(g), 2) - val) < 1e-12


def test_mp50_known_rectangular():
    import mpmath as mp
    from qicert.certify.cross_check import _core_norm_mp50
    # [[1,2],[3,4],[5,6]]: A^T A = [[35,44],[44,56]], trace 91, det 24,
    # so lambda_max = (91 + sqrt(91^2 - 4*24))/2 = (91 + sqrt(8185))/2
    # and sigma_max = sqrt(lambda_max).
    g = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    expected = float(np.sqrt((91 + np.sqrt(91**2 - 4 * 24)) / 2))
    val = _core_norm_mp50(g, mp.mp)
    assert abs(val - expected) < 1e-12, (val, expected)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-q", __file__]))
