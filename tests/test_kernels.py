"""Property tests for the four Python reference kernels (Phase-0.5 floor).

These tests ARE the conformance spec for the C++/CUDA-Q and Julia ports
(Q19/Q22): the parity harness below must be reproduced by every backend
within the documented tolerances.
"""
import numpy as np
import pytest

from qicert.kernels import (
    KernelError,
    available_backends,
    get_backend,
    parity_check,
    set_active_backend,
    set_active_sos_backend,
)
from qicert.kernels.base import Cores

K = get_backend("python")


def _random_cores(m_dims, n_dims, ranks, seed=0):
    """Build a random exact-TT matrix (cores with the qicert convention)."""
    rng = np.random.default_rng(seed)
    d = len(m_dims)
    cores = []
    for k in range(d):
        r_prev = 1 if k == 0 else ranks[k - 1]
        r_next = 1 if k == d - 1 else ranks[k]
        cores.append(rng.normal(size=(r_prev, m_dims[k], n_dims[k], r_next)))
    W = K.contract_cores(cores, m_dims, n_dims)
    return cores, W


# ---------------------------------------------------------------------------
# Pillar A — TT-SVD / TT-cross / Lipschitz
# ---------------------------------------------------------------------------

def test_tt_svd_exact_reconstruction():
    cores, W = _random_cores((4, 4), (4, 4), (2,), seed=1)
    cs = K.tt_svd(W, (4, 4), (4, 4), ranks=(2,))
    rec = K.contract_cores(cs.arrays, (4, 4), (4, 4))
    err = np.linalg.norm(rec - W) / np.linalg.norm(W)
    assert err < 1e-10, f"TT-SVD reconstruction err {err:.2e}"


def test_tt_svd_small_kv_proj_deep_split():
    """Regression: GQA k/v projections (128 x 896) at d=4 deep splits.

    The requested rank (e.g. 36 at 50% params) exceeds the SVD's available
    right dimension at the last split, so the achieved rank is smaller.
    tt_svd must track achieved ranks instead of assuming the requested rank
    fits; previously this raised "cannot reshape array of size X into shape
    (r,4,4,16)" (ValueError).
    """
    rng = np.random.default_rng(7)
    W = rng.standard_normal((128, 896))
    m_dims = (4, 4, 4, 2)
    n_dims = (8, 7, 4, 4)
    for frac_rank in (20, 36):  # 16% and 50%-plan rank for this shape
        cs = K.tt_svd(W, m_dims, n_dims, ranks=(frac_rank,) * 3)
        rec = K.contract_cores(cs.arrays, m_dims, n_dims)
        assert rec.shape == W.shape
        # achieved ranks: bond dims must telescope (r_{k-1}, m_k, n_k, r_k)
        for k in range(1, len(cs.arrays)):
            assert cs.arrays[k].shape[0] == cs.arrays[k - 1].shape[-1]
        err = np.linalg.norm(rec - W) / np.linalg.norm(W)
        assert 0.0 <= err <= 1.1, f"reconstruction err {err:.3f}"


def test_tt_svd_exact_reconstruction_deep_split():
    """TT-SVD must reconstruct an exact-TT matrix for d=4 (QTT deep split)."""
    cores, W = _random_cores((4, 2, 2, 2), (4, 2, 2, 2), (3, 4, 2), seed=7)
    cs = K.tt_svd(W, (4, 2, 2, 2), (4, 2, 2, 2), ranks=(3, 4, 2))
    rec = K.contract_cores(cs.arrays, (4, 2, 2, 2), (4, 2, 2, 2))
    err = np.linalg.norm(rec - W) / np.linalg.norm(W)
    assert err < 1e-8, f"TT-SVD d=4 reconstruction err {err:.2e}"
    assert cs.ranks == (3, 4, 2)


def test_tt_cross_exact_reconstruction():
    """TT-cross must reconstruct an exact-TT matrix (rank-2, 16x16)."""
    cores, W = _random_cores((4, 4), (4, 4), (2,), seed=2)
    cs = K.tt_cross(W, (4, 4), (4, 4), ranks=(2,))
    rec = K.contract_cores(cs.arrays, (4, 4), (4, 4))
    err = np.linalg.norm(rec - W) / np.linalg.norm(W)
    assert err < 1e-8, f"TT-cross reconstruction err {err:.2e}"


def test_tt_cross_tracks_svd_truncation():
    """On a full-rank matrix, TT-cross error must be within a small factor of
    the optimal rank-r SVD truncation error (never catastrophically worse)."""
    rng = np.random.default_rng(3)
    W = rng.normal(size=(16, 16))
    cs = K.tt_cross(W, (4, 4), (4, 4), ranks=(2,))
    rec = K.contract_cores(cs.arrays, (4, 4), (4, 4))
    err = np.linalg.norm(rec - W) / np.linalg.norm(W)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    opt = np.sqrt(np.sum(S[2:] ** 2)) / np.linalg.norm(W)
    assert err < 5.0 * opt + 1e-6, f"cross err {err:.3f} vs optimal {opt:.3f}"


def test_tt_ranks_padding():
    """The ranks tuple is normalized to exactly d-1 internal bonds."""
    cores, W = _random_cores((4, 4), (4, 4), (2,), seed=8)
    # pass too many ranks (should truncate) and too few (should pad)
    cs = K.tt_svd(W, (4, 4), (4, 4), ranks=(2, 2, 2))
    assert cs.ranks == (2,)
    # rank-1 cross of a rank-2 matrix: completes, finite, and strictly lossy
    cs1 = K.tt_cross(W, (4, 4), (4, 4), ranks=())
    assert cs1.ranks == (1,)
    rec1 = K.contract_cores(cs1.arrays, (4, 4), (4, 4))
    err1 = np.linalg.norm(rec1 - W) / np.linalg.norm(W)
    assert np.isfinite(err1) and err1 > 1e-6, f"rank-1 cross should be lossy, got {err1:.2e}"


def test_tt_cross_deep_split_survives_rank_deficient_pivots():
    """Regression (E3 triage): deep mode splits make random-skeleton
    collisions likely (small fused modes), so maxvol's pivot block goes
    rank-deficient and the exact solve raised LinAlgError. The kernel must
    degrade gracefully: complete, finite, and within the same small-factor
    of the optimal truncation as the shallow case."""
    rng = np.random.default_rng(3)
    W = rng.normal(size=(128, 128)) * np.logspace(0, -3, 128)[None, :]
    m_dims = n_dims = (8, 4, 2, 2)          # d=4 deep split, products 128
    ranks = (4, 4, 4)
    cs = K.tt_cross(W, m_dims, n_dims, ranks)
    assert all(np.isfinite(g).all() for g in cs.arrays), "non-finite cores"
    Wr = K.contract_cores(cs.arrays, m_dims, n_dims)
    err = float(np.linalg.norm(Wr - W) / np.linalg.norm(W))
    S = np.linalg.svd(W, compute_uv=False)
    opt8 = float(np.sqrt(np.sum(S[8:] ** 2)) / np.linalg.norm(W))
    assert err < 5.0 * opt8 + 1e-6, f"cross err {err:.3f} vs optimal {opt8:.3f}"


def test_lipschitz_product_bounds_tight_norm():
    """Layer-1: product bound >= tight norm, and tight norm tracks dense norm."""
    cores, W = _random_cores((4, 4), (4, 4), (2,), seed=4)
    L_prod = K.lipschitz_product(cores)
    L_tight = K.operator_norm_tight(cores, (4, 4), (4, 4))
    L_dense = float(np.linalg.norm(W, 2))
    assert L_prod >= L_tight - 1e-9, "product bound must dominate tight norm"
    assert abs(L_tight - L_dense) / L_dense < 1e-6, "power iteration should track dense norm"
    assert 0 < L_prod < 1e6


def test_lipschitz_constant_wrapper():
    """qicert.compress wrapper keeps the pre-registered formula."""
    import qicert.compress as C
    cores = [np.eye(4) * 0.5 for _ in range(3)]
    assert abs(C.lipschitz_constant(cores) - 0.5 ** 3) < 1e-12


# ---------------------------------------------------------------------------
# Gauge-minimized Layer-1 certificates (E1)
# ---------------------------------------------------------------------------

def _gauge_scaled(cores, logs):
    """Apply matched per-bond uniform scalings (a valid gauge move)."""
    out = []
    for k, g in enumerate(cores):
        h = np.array(g, dtype=float)
        if k > 0:
            h = h / (10.0 ** logs[k - 1])
        if k < len(cores) - 1:
            h = np.moveaxis(h, 3, -1) * (10.0 ** logs[k])
            h = np.moveaxis(h, -1, 3)
        out.append(h)
    return out


@pytest.mark.parametrize("dims,ranks", [
    ((4, 4), (2,)),
    ((2, 2, 2), (3, 2)),
    ((2, 2, 2, 2), (3, 4, 2)),
])
def test_min_gauge_product_invariance_and_tightening(dims, ranks):
    """Gauge descent must not move reconstruction and must not increase L."""
    cores, W = _random_cores(dims, dims, ranks, seed=7)
    rng = np.random.default_rng(11)
    logs = rng.uniform(-4, 4, size=len(cores) - 1)
    scaled = _gauge_scaled(cores, logs)
    W2 = K.contract_cores(scaled, dims, dims)
    rel_gauge = np.linalg.norm(W2 - W) / np.linalg.norm(W)
    assert rel_gauge < 1e-12, f"gauge move itself broke reconstruction {rel_gauge:.2e}"
    L_orig = K.lipschitz_product(cores)
    L_scaled = K.lipschitz_product(scaled)
    L_min, stats = K.min_gauge_product(scaled, sweeps=50)
    assert L_min <= L_orig * (1 + 1e-9), \
        f"L_min {L_min:.6e} exceeds original product {L_orig:.6e}"
    assert L_min <= L_scaled * (1 + 1e-12)
    Wr = K.contract_cores(stats["cores"], dims, dims)
    rel = np.linalg.norm(Wr - W) / np.linalg.norm(W)
    assert rel < 1e-10, f"minimization moved reconstruction by {rel:.2e}"
    assert stats["converged"], "descent did not converge within 50 sweeps"


def test_min_gauge_product_soundness_report():
    """L_min >= tight norm must hold on random TTs (soundness of the
    minimized certificate). A violation is a REAL finding: print the report
    and xfail rather than assert-crash (E1 brief)."""
    violations = []
    for seed in range(10):
        dims = (4, 4)
        cores, W = _random_cores(dims, dims, (2,), seed=100 + seed)
        L_min, _ = K.min_gauge_product(cores, sweeps=20)
        tight = K.operator_norm_tight(cores, dims, dims, iters=100)
        dense = float(np.linalg.norm(W, 2))
        if L_min < min(tight, dense) * (1 - 1e-9):
            violations.append((seed, L_min, tight, dense))
    if violations:
        for s, l, t, dn in violations:
            print(f"[gauge-soundness] seed={s}: L_min={l:.6e} tight={t:.6e} "
                  f"dense={dn:.6e}")
        pytest.xfail(f"{len(violations)}/10 random TTs violate L_min >= bound "
                     "(real finding — see printed report)")


# ---------------------------------------------------------------------------
# Pillar B — commuting-Pauli compiler
# ---------------------------------------------------------------------------

def test_pauli_grouping_families_commute():
    """Greedy grouping must only ever place commuting strings together."""
    table = [(1.0, "ZX"), (0.5, "ZZ"), (0.25, "XI"), (-0.3, "YY"),
             (0.2, "XX"), (0.1, "II"), (-0.4, "XZ")]
    g = K.pauli_grouping(table, k=2)
    for fam in g.families:
        for i in range(len(fam)):
            for j in range(i + 1, len(fam)):
                assert K._commute(fam[i][1], fam[j][1]), f"{fam[i][1]} vs {fam[j][1]}"


def test_pauli_diagonalize_identity():
    """The joint eigenbasis must exactly diagonalize a commuting family."""
    fam = [(1.0, "ZZ"), (0.5, "ZI"), (0.25, "IZ")]
    dg = K.pauli_diagonalize(fam, k=2)
    U = dg.unitary
    T = sum(c * K.pauli_matrix(p, 2) for c, p in fam)
    rebuilt = U @ np.diag(np.diag(U.conj().T @ T @ U)) @ U.conj().T
    err = np.linalg.norm(rebuilt - T)
    assert err < 1e-9, f"diagonalization identity err {err:.2e}"
    for d in dg.eigenvalues:
        assert np.allclose(np.abs(d), 1.0)


def test_pauli_matrix_basis():
    """pauli_matrix must reproduce the Pauli algebra on 1 qubit."""
    X = K.pauli_matrix("X", 1)
    Z = K.pauli_matrix("Z", 1)
    Y = K.pauli_matrix("Y", 1)
    assert np.allclose(X @ X, np.eye(2))
    assert np.allclose(Z @ Z, np.eye(2))
    assert np.allclose(Y @ Y, np.eye(2))
    assert np.allclose(X @ Z, 1j * Y) or np.allclose(X @ Z, -1j * Y)


# ---------------------------------------------------------------------------
# Pillar C — Bayesian IQAE
# ---------------------------------------------------------------------------

def test_iqae_covers_known_p():
    """IQAE posterior credible interval must contain a known p."""
    p_true = 0.1
    rng = np.random.default_rng(5)

    def measure(depth, shots):
        th = float(np.arcsin(np.sqrt(p_true)))
        prob = np.clip(np.sin((2 * depth + 1) * th) ** 2, 1e-12, 1 - 1e-12)
        return int(rng.binomial(shots, prob))

    iv = K.iqae(measure, budget=150, shots=30, grid=8192, seed=5)
    assert iv.lo <= p_true <= iv.hi, f"p={p_true} not in [{iv.lo:.3f}, {iv.hi:.3f}]"
    assert 0.0 <= iv.p_map <= 1.0
    # more budget => tighter interval
    iv2 = K.iqae(measure, budget=400, shots=30, grid=8192, seed=5)
    assert (iv2.hi - iv2.lo) < (iv.hi - iv.lo) + 1e-9


# ---------------------------------------------------------------------------
# Monitor — syndrome-shadow runtime guard
# ---------------------------------------------------------------------------

def _shadow_baseline_stats(rng, projections, d, n=4000):
    """Reference mean/std of one projection under in-distribution input."""
    base = rng.normal(size=(n, d))
    s = base @ projections.T          # (n, M)
    return float(s.mean()), float(s.std())


def test_shadow_statistics_baseline_no_alarm():
    """At fpr=0.01 the monitor must not alarm on in-distribution input."""
    rng = np.random.default_rng(6)
    d = 32
    proj = rng.normal(size=(128, d))
    mean0, std0 = _shadow_baseline_stats(rng, proj, d)
    x = rng.normal(size=d)
    v = K.shadow_statistics(x, proj, baseline_mean=mean0, baseline_std=std0)
    assert v.alarm is False, "false alarm on baseline input"


def test_shadow_statistics_anomaly_alarms():
    """A large injected shift must trip the median-of-means gate."""
    rng = np.random.default_rng(7)
    d = 32
    proj = rng.normal(size=(256, d))
    mean0, std0 = _shadow_baseline_stats(rng, proj, d)
    # shift the input along the mean projection direction (coherent shift)
    u = proj.sum(axis=0)
    u = u / np.linalg.norm(u)
    x = rng.normal(size=d) + 25.0 * u
    v = K.shadow_statistics(x, proj, baseline_mean=mean0, baseline_std=std0)
    assert v.alarm is True, "missed injected anomaly"


# ---------------------------------------------------------------------------
# Backend registry + parity harness (the conformance contract)
# ---------------------------------------------------------------------------

def test_registry_defaults_and_errors():
    assert "python" in available_backends()
    assert get_backend().name == "python"
    with pytest.raises(KernelError):
        get_backend("cpp")  # not registered yet (Q19)
    with pytest.raises(KernelError):
        set_active_backend("cpp")
    with pytest.raises(KernelError):
        set_active_sos_backend("julia")  # not registered yet (Q22)
    set_active_backend("python")
    set_active_sos_backend("python")


def test_parity_harness_fingerprint():
    """The parity harness returns the numbers every backend must match."""
    fp = parity_check()
    assert fp["tt_svd_rel_err"] < 1e-10
    assert fp["lipschitz_tight_vs_dense"] < 1e-4
    assert fp["pauli_identity_err"] < 1e-9
    assert fp["iqae_interval_contains_p"] == 1.0
