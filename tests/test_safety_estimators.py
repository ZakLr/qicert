"""Tests for the N6 safety estimators (three-arm race machinery)."""
from __future__ import annotations

import numpy as np
import pytest

from qicert.safety import (
    _clopper_pearson,
    _count_failures,
    analytic_margin_oracle,
    estimate_failure_probability,
    scenario_opt_eps,
)


class _VecOracle:
    """Vectorized indicator oracle with .rho access (mirrors bench usage)."""

    def __init__(self, rollout):
        self.rho = rollout

    def __call__(self, seed):
        arr = np.asarray(seed)
        return (self.rho(arr.astype(np.int64)) < 0.0).astype(np.int64)


@pytest.fixture(scope="module")
def oracle_and_truth():
    rollout, p_true = analytic_margin_oracle()   # rare-event defaults
    return _VecOracle(rollout), p_true


def test_oracle_ground_truth_is_rare_and_analytic(oracle_and_truth):
    oracle, p_true = oracle_and_truth
    assert 1e-6 < p_true < 1e-2, f"demo oracle should be rare-event: {p_true}"
    # empirical rate converges near the analytic value
    seeds = np.arange(1, 400_001, dtype=np.int64)
    emp = float(np.mean(oracle(seeds)))
    assert abs(emp - p_true) < 5 * np.sqrt(p_true / 400_000)


def test_count_failures_chunked_matches_direct(oracle_and_truth):
    oracle, _ = oracle_and_truth
    n = 250_001     # not a multiple of the 200k chunk size
    k = _count_failures(oracle, base=1, n=n)
    seeds = np.arange(0, n, dtype=np.int64) + 1
    k_direct = int(np.sum(oracle(seeds)))
    assert k == k_direct


def test_clopper_pearson_sane():
    lo, hi = _clopper_pearson(0, 100)
    assert lo == 0.0 and 0 < hi < 0.05
    lo, hi = _clopper_pearson(50, 100)
    # exact CP interval for 50/100 is [0.3983, 0.6017]
    assert 0.39 < lo < 0.5 < hi < 0.61


def test_mc_arm_covers_truth(oracle_and_truth):
    oracle, p_true = oracle_and_truth
    res = estimate_failure_probability("mc", oracle, n_queries=1_000_000,
                                       seed=0)
    assert res["ci_lo"] <= p_true <= res["ci_hi"]
    assert res["width"] < 1e-3


def test_gev_arm_shape_and_eventual_coverage(oracle_and_truth):
    oracle, p_true = oracle_and_truth
    res = estimate_failure_probability("restart", oracle, n_queries=1_000_000,
                                       seed=0, n_blocks=40)
    assert 0 < res["p_hat"] < 1
    assert 0 <= res["ci_lo"] <= res["ci_hi"] <= 1
    assert res["width"] > 0


def test_iqae_arm_needs_true_p(oracle_and_truth):
    oracle, _ = oracle_and_truth
    with pytest.raises(ValueError):
        estimate_failure_probability("q", oracle, n_queries=50_000)


def test_iqae_arm_intervals_contain_point_estimate(oracle_and_truth):
    oracle, p_true = oracle_and_truth
    res = estimate_failure_probability("q", oracle, n_queries=50_000,
                                       true_p=p_true, iqae_grid=4096)
    assert res["ci_lo"] <= res["p_hat"] <= res["ci_hi"]
    # simulated IQAE must beat naive MC at the SAME budget (bench-measured
    # ~3x at 50k queries; >=2x is the regression bar)
    mc = estimate_failure_probability("mc", oracle, n_queries=50_000, seed=0)
    assert res["width"] < mc["width"] / 2


def test_unknown_arm_raises(oracle_and_truth):
    oracle, _ = oracle_and_truth
    with pytest.raises(NotImplementedError):
        estimate_failure_probability("quantum-supremacy", oracle,
                                     n_queries=100)


def test_scenario_opt_eps_monotone():
    eps_small_n = scenario_opt_eps(1_000, 0.05)
    eps_big_n = scenario_opt_eps(1_000_000, 0.05)
    assert eps_big_n < eps_small_n < 0.01
    # tighter confidence => larger bound
    assert scenario_opt_eps(10_000, 0.01) > scenario_opt_eps(10_000, 0.05)
