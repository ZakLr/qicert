"""Tests for the residual-compensation repair arm (N2-double-prime)."""
from __future__ import annotations

import numpy as np
import pytest

from qicert.compress_residual import (
    apply,
    compressed_params,
    layer_report,
    lipschitz_bound,
    svd_residual,
    to_dense,
)
from qicert.kernels import get_backend


def _random_cores(m_dims, n_dims, ranks, seed=0):
    rng = np.random.default_rng(seed)
    d = len(m_dims)
    cores = []
    for k in range(d):
        r_prev = 1 if k == 0 else ranks[k - 1]
        r_next = 1 if k == d - 1 else ranks[k]
        cores.append(rng.standard_normal(
            (r_prev, m_dims[k], n_dims[k], r_next)) / np.sqrt(r_prev * r_next))
    return cores


@pytest.fixture()
def backend():
    return get_backend()


def test_residual_recovers_truncation_loss(backend):
    """The repair mechanism's two provable properties + the adversarial case.

    1. MONOTONE: adding the best rank-r' approximation of R = W - What to
       What can only reduce the Frobenius error (Eckart-Young), for ANY W.
    2. EXACT when R is exactly low-rank: if W = What + (rank-r' perturbation),
       the repair reconstructs W to machine precision.
    3. ADVERSARIAL white-noise W: strict improvement (>=10%), but not
       near-exact — the honest expectation for flat-spectra real weights,
       which is why the bench arm pairs the residual with per-channel
       activation scales (output-error minimization), not Frobenius alone.
    """
    m_dims, n_dims = (4, 4), (4, 4)

    # --- (1) monotone improvement on a generic matrix ---------------------
    rng = np.random.default_rng(1)
    W = rng.standard_normal((16, 16))
    cs = backend.tt_svd(W, m_dims, n_dims, (2,))
    What = backend.contract_cores(cs.arrays, m_dims, n_dims)
    err_before = np.linalg.norm(What - W) / np.linalg.norm(W)
    comp = svd_residual(cs.arrays, m_dims, n_dims, W, residual_rank=4)
    Wc = to_dense(comp, m_dims, n_dims)
    err_after = np.linalg.norm(Wc - W) / np.linalg.norm(W)
    assert err_after < err_before, "repair must never hurt (Eckart-Young)"

    # --- (2) exact recovery when the residual is exactly low-rank ---------
    # Pipeline semantics: the TT cores are FIXED (what we keep/ship); the
    # residual is fitted against THAT reconstruction.  If W = What + P with
    # rank(P) <= r', the repaired operator reproduces W exactly.
    rng = np.random.default_rng(7)
    P = rng.standard_normal((16, 2)) @ rng.standard_normal((2, 16)) / 2.0
    W2 = What + P                      # residual vs the SAME cores is rank-2
    comp2 = svd_residual(cs.arrays, m_dims, n_dims, W2, residual_rank=2)
    W2c = to_dense(comp2, m_dims, n_dims)
    err2 = np.linalg.norm(W2c - W2) / np.linalg.norm(W2)
    # threshold = float32 storage precision of the residual factors (~1e-7),
    # not float64: deployment stores u/v as float32 by design.
    assert err2 < 1e-6, f"exact-rank residual must reconstruct: {err2:.2e}"

    # --- (3) adversarial white noise: strict improvement is the bar -------
    W_noise = rng.standard_normal((16, 16))
    cs_n = backend.tt_svd(W_noise, m_dims, n_dims, (2,))
    What_n = backend.contract_cores(cs_n.arrays, m_dims, n_dims)
    err_n_before = np.linalg.norm(What_n - W_noise) / np.linalg.norm(W_noise)
    comp_n = svd_residual(cs_n.arrays, m_dims, n_dims, W_noise, residual_rank=4)
    Wc_n = to_dense(comp_n, m_dims, n_dims)
    err_n_after = np.linalg.norm(Wc_n - W_noise) / np.linalg.norm(W_noise)
    assert err_n_after < err_n_before * 0.9, \
        f"noise case must improve: {err_n_before:.4f} -> {err_n_after:.4f}"


def test_to_dense_equals_apply(backend):
    """apply() and to_dense() must agree: same linear operator."""
    rng = np.random.default_rng(2)
    m_dims, n_dims = (4, 4), (4, 4)
    W = rng.standard_normal((16, 16))
    cs = backend.tt_svd(W, m_dims, n_dims, (2,))
    comp = svd_residual(cs.arrays, m_dims, n_dims, W, residual_rank=3)
    Wc = to_dense(comp, m_dims, n_dims)
    x = rng.standard_normal(16)
    y_apply = apply(comp, x, m_dims, n_dims)
    y_dense = Wc @ x
    np.testing.assert_allclose(y_apply, y_dense, rtol=1e-4, atol=1e-4)


def test_lipschitz_bound_is_sound(backend):
    """The analytic bound must upper-bound the true operator norm."""
    rng = np.random.default_rng(3)
    m_dims, n_dims = (4, 4), (4, 4)
    W = rng.standard_normal((16, 16))
    cs = backend.tt_svd(W, m_dims, n_dims, (2,))
    comp = svd_residual(cs.arrays, m_dims, n_dims, W, residual_rank=4)
    L = lipschitz_bound(comp, m_dims, n_dims)
    Wc = to_dense(comp, m_dims, n_dims)
    tight = np.linalg.norm(Wc, 2)  # true operator norm
    assert L >= tight - 1e-6, f"bound {L} < tight {tight}"


def test_param_accounting_is_honest(backend):
    """Reported params must include TT cores + residual factors + scales."""
    m_dims, n_dims = (4, 4), (4, 4)
    rng = np.random.default_rng(4)
    W = rng.standard_normal((16, 16))
    cs = backend.tt_svd(W, m_dims, n_dims, (2,))
    comp = svd_residual(cs.arrays, m_dims, n_dims, W, residual_rank=4)
    n = compressed_params(comp)
    tt = sum(int(np.prod(g.shape)) for g in cs.arrays)
    assert n == tt + 16 * 4 + 4 * 16 + 16 + 16


def test_layer_report_finite_and_sound(backend):
    m_dims, n_dims = (4, 4), (4, 4)
    rng = np.random.default_rng(5)
    W = rng.standard_normal((16, 16))
    cs = backend.tt_svd(W, m_dims, n_dims, (2,))
    comp = svd_residual(cs.arrays, m_dims, n_dims, W, residual_rank=2)
    rep = layer_report(comp, W, m_dims, n_dims)
    assert np.isfinite(rep["recon_rel_err"])
    assert np.isfinite(rep["lipschitz_bound"])
    assert rep["sound"] is True
    assert rep["params"] < rep["dense_params"]


# ---------------------------------------------------------------------------
# N2R-v2: activation-weighted residual fit (2026-09-12)
# ---------------------------------------------------------------------------

def test_weighted_fit_minimizes_weighted_error(backend):
    """Eckart-Young in the sqrt(w) space: the weighted fit must beat the
    plain Frobenius fit ON THE WEIGHTED OBJECTIVE, for any weight vector."""
    rng = np.random.default_rng(11)
    m_dims, n_dims = (4, 4), (4, 4)
    W = rng.standard_normal((16, 16))
    cs = backend.tt_svd(W, m_dims, n_dims, (2,))
    What = backend.contract_cores(cs.arrays, m_dims, n_dims)
    w = (1.0 + rng.random(16)) ** 3          # strongly non-uniform weights
    sw = np.sqrt(w)
    for rrank in (1, 3, 8):
        comp_plain = svd_residual(cs.arrays, m_dims, n_dims, W, rrank)
        comp_w = svd_residual(cs.arrays, m_dims, n_dims, W, rrank,
                              activation_weight=w)
        A = W - What
        def werr(c):
            Wc = to_dense(c, m_dims, n_dims) - What
            return np.linalg.norm((A - Wc) * sw[None, :])
        assert werr(comp_w) <= werr(comp_plain) + 1e-9, (
            f"weighted fit must minimize the weighted objective at r'={rrank}")


def test_weighted_apply_matches_to_dense(backend):
    """`apply` and `to_dense` must implement the same weighted operator."""
    rng = np.random.default_rng(12)
    m_dims, n_dims = (4, 4), (4, 4)
    W = rng.standard_normal((16, 16))
    cs = backend.tt_svd(W, m_dims, n_dims, (2,))
    w = rng.random(16) + 0.1
    comp = svd_residual(cs.arrays, m_dims, n_dims, W, 4,
                        activation_weight=w)
    x = rng.standard_normal(16)
    y_apply = apply(comp, x, m_dims, n_dims)
    y_dense = to_dense(comp, m_dims, n_dims) @ x
    np.testing.assert_allclose(y_apply, y_dense, rtol=1e-4, atol=1e-4)


def test_weighted_lipschitz_bound_is_sound(backend):
    """The analytic bound must upper-bound the true operator norm of the
    weighted-compensated layer (max_j|s_in_j| is the exact diagonal norm)."""
    rng = np.random.default_rng(13)
    m_dims, n_dims = (4, 4), (4, 4)
    W = rng.standard_normal((16, 16))
    cs = backend.tt_svd(W, m_dims, n_dims, (2,))
    w = (rng.random(16) + 0.05) ** 2
    comp = svd_residual(cs.arrays, m_dims, n_dims, W, 4,
                        activation_weight=w)
    L = lipschitz_bound(comp, m_dims, n_dims)
    Wc = to_dense(comp, m_dims, n_dims)
    tight = np.linalg.norm(Wc, 2)
    assert L >= tight - 1e-6, f"bound {L} < tight {tight}"


def test_weighted_scales_are_identity(backend):
    """The weighted arm keeps s_in/s_out at identity (the weighting picks
    WHICH rank-r' correction to add, it does not rescale the operator), so
    the certificate is exactly ||TT|| + ||U|| ||V|| with no diagonal factors."""
    rng = np.random.default_rng(14)
    m_dims, n_dims = (4, 4), (4, 4)
    W = rng.standard_normal((16, 16))
    cs = backend.tt_svd(W, m_dims, n_dims, (2,))
    w = (rng.random(16)) ** 6 + 1e-9
    comp = svd_residual(cs.arrays, m_dims, n_dims, W, 2,
                        activation_weight=w)
    s_in = comp["s_in"]
    assert np.all(s_in == 1.0)
    assert np.all(comp["s_out"] == 1.0)
    assert comp["activation_weighted"] is True
    assert comp["residual_rank"] == 2


def test_weighted_rejects_bad_weights(backend):
    m_dims, n_dims = (4, 4), (4, 4)
    W = np.eye(16)
    cs = backend.tt_svd(W, m_dims, n_dims, (2,))
    with pytest.raises(ValueError):
        svd_residual(cs.arrays, m_dims, n_dims, W, 2,
                     activation_weight=np.ones(15))   # wrong channel count
    with pytest.raises(ValueError):
        svd_residual(cs.arrays, m_dims, n_dims, W, 2,
                     activation_weight=-np.ones(16))  # negative weight
