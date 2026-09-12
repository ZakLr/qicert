"""Safety evaluation engine (N6): STL specs + three-arm estimator race.

Arms: Q (simulated Bayesian IQAE) vs C-strong (block-minima GEV extreme-
tail fit; the RESTART splitting component requires rollout trajectory
access, which a seed->rho oracle does not expose -- documented scoping)
vs C-naive (Monte-Carlo counting + Clopper-Pearson). Closed by
Campi-Garatti scenario-optimization bounds and empirical interval
calibration (split-conformal form).

Race design (2026-09-12): the demo oracle has an ANALYTIC failure
probability in the RARE-EVENT regime (p ~ 1e-4, heavy Student-t tail),
so all three arms are scored against ground truth -- coverage and
interval width at matched query budget are the honest race metrics.

IQAE simulation note: the amplification simulator uses the TRUE amplitude
(passed as true_p, knowledge of the simulated quantum device) to produce
shot-noised measurement counts; the ESTIMATOR (kernel iqae) never sees
true_p -- it only receives measurement statistics, exactly as on hardware.
This is the standard way to score an estimator's statistics without a QPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class STLSpec:
    """A formal safety requirement with quantitative robustness rho."""
    name: str
    formula: str          # e.g. "box[0,T](clearance > eps_coll)"
    track: str            # "AD" | "robotics"
    predicates: list[str]


AD_SPECS = [
    STLSpec("phi_1", "box[0,T](clearance(ego, agents) > eps_coll)", "AD",
            ["clearance", "eps_coll"]),
    STLSpec("phi_2", "box[0,T](|lateral_offset| < lane_halfwidth - eps_lane)", "AD",
            ["lateral_offset", "lane_halfwidth", "eps_lane"]),
    STLSpec("phi_3", "diamond[0,T_stop](speed < v_safe)", "AD", ["speed", "v_safe"]),
    STLSpec("phi_4", "box[0,T] not(off_road and moving)", "AD", ["off_road", "moving"]),
]

ROBOTICS_SPECS = [
    STLSpec("psi_1", "box(joint_velocities in [-v_j, +v_j])", "robotics",
            ["joint_velocities", "v_j"]),
    STLSpec("psi_2", "box not(end_effector in forbidden_region)", "robotics",
            ["end_effector", "forbidden_region"]),
    STLSpec("psi_3", "diamond(gripper settled and task_pose within delta)", "robotics",
            ["gripper", "task_pose", "delta"]),
]


def analytic_margin_oracle(m0: float = 1.0, scale: float = 0.05, df: float = 3.0):
    """Seeded heavy-tailed margin oracle with a KNOWN failure probability.

    rho(seed) = m0 - scale * T_df(rng(seed))  where T_df is standard
    Student-t with `df` dof (heavy tail => rare large violations, the
    regime where naive MC dies).  Failure: rho < 0, i.e.
    p_true = Pr[T_df > m0/scale] -- computable in closed form, so the
    three-arm race can be scored against ground truth.
    Defaults give p_true ~ 1.1e-4 (rare-event regime).

    Returns (rollout seed-array->rho vectorized, p_true).
    """
    from scipy import stats

    p_true = float(stats.t.sf(m0 / scale, df))

    def rollout(seed):
        # PER-SEED determinism is the oracle's contract (challenge Sec 5.2:
        # seeds fully determine execution).  numpy's default_rng treats a
        # seed ARRAY as one seed, which would make rho(seed_i) depend on its
        # neighbors -- so each seed gets its own scalar-seeded generator.
        seed = np.asarray(seed, dtype=np.int64)
        return np.array([m0 - scale * np.random.default_rng(int(s)).standard_t(df)
                         for s in seed], dtype=np.float64)

    return rollout, p_true


def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI (the C-naive arm's interval)."""
    from scipy import stats
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _count_failures(oracle, base: int, n: int) -> int:
    """Vectorized failure count over seed range [base, base+n), chunked
    (handles chunk sizes that do not divide evenly)."""
    total = 0
    got = 0
    chunk = 200_000
    while got < n:
        m = min(chunk, n - got)
        seeds = np.arange(got, got + m, dtype=np.int64) + base
        total += int(np.sum(oracle(seeds)))
        got += m
    return total


def estimate_failure_probability(arm: str, oracle, n_queries: int,
                                 seed: int = 0, shots: int = 30,
                                 n_blocks: int = 200,
                                 true_p: float | None = None,
                                 iqae_grid: int = 16384) -> dict:
    """Three-arm race on p = Pr[rho < 0] for a seed->indicator oracle.

    oracle   : callable, seed (int or int-array) -> {0,1} failure indicator;
               may expose .rho(seed_array) -> rho values (C-strong arm)
    arm="q"      simulated Bayesian IQAE (kernel backend).  true_p is the
                 SIMULATOR'S ground-truth amplitude (ideal amplitude-encoded
                 oracle); the estimator sees only measurement counts.
    arm="restart"  C-strong: block-minima GEV extreme-tail fit with block
                 bootstrap CI (RESTART splitting needs trajectory access;
                 documented scoping).
    arm="mc"     naive counting + Clopper-Pearson (the strawman)

    Returns {p_hat, ci_lo, ci_hi, width, n_queries}.
    """
    from qicert.kernels import get_backend

    n_queries = int(n_queries)
    if n_queries <= 0:
        raise ValueError("n_queries must be positive")
    base = 1 + seed * 1_000_003

    if arm == "mc":
        k = _count_failures(oracle, base, n_queries)
        lo, hi = _clopper_pearson(k, n_queries)
        return {"arm": "mc", "p_hat": k / n_queries, "ci_lo": lo,
                "ci_hi": hi, "width": hi - lo, "n_queries": n_queries}

    if arm == "restart":
        from scipy import stats
        # rho values on the same seed budget (oracle calls = n_queries)
        chunk = 200_000
        rhos = []
        got = 0
        while got < n_queries:
            m = min(chunk, n_queries - got)
            seeds = np.arange(got, got + m, dtype=np.int64) + base
            rhos.append(np.asarray(oracle.rho(seeds), dtype=np.float64))
            got += m
        rho = np.concatenate(rhos)
        # Block-minima GEV: split into b blocks, take per-block minima, fit
        # GEV to the block minima, p_block = P(min < 0) = cdf(0); pointwise
        # p_hat = p_block / b (small-p approximation, honest for rare p).
        b = max(100, n_queries // 1000)          # block size (~1e3 blocks)
        n_blocks_bm = n_queries // b
        if n_blocks_bm < 50:
            b = max(20, n_queries // 100)
            n_blocks_bm = n_queries // b
        minima = rho[: n_blocks_bm * b].reshape(n_blocks_bm, b).min(axis=1)
        c, loc, scl = stats.genextreme.fit(minima)
        p_block = float(stats.genextreme.cdf(0.0, c, loc=loc, scale=scl))
        p_block = min(max(p_block, 1e-12), 1.0)
        p_hat = p_block / b
        # block bootstrap over the block minima (fixed-size, tractable MLE)
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_blocks):
            sample = minima[rng.integers(0, n_blocks_bm, n_blocks_bm)]
            try:
                cb, lb, sb = stats.genextreme.fit(sample)
                pb = float(stats.genextreme.cdf(0.0, cb, loc=lb, scale=sb)) / b
                boots.append(min(max(pb, 0.0), 1.0))
            except Exception:
                continue
        lo, hi = (float(np.quantile(boots, 0.025)),
                  float(np.quantile(boots, 0.975))) if boots else (0.0, 1.0)
        return {"arm": "restart+gev", "p_hat": p_hat, "ci_lo": lo,
                "ci_hi": hi, "width": hi - lo, "n_queries": n_queries,
                "block_size": b, "n_block_minima": int(n_blocks_bm)}

    if arm == "q":
        if true_p is None:
            raise ValueError("arm 'q' needs true_p (the simulated device's "
                             "ground-truth amplitude) to run the "
                             "amplification simulator")
        theta_true = float(np.arcsin(np.sqrt(min(max(true_p, 1e-12), 1 - 1e-12))))

        def measure(depth: int, s: int) -> int:
            # ideal Grover amplification of the indicator, shot noise kept
            p_d = float(np.sin((2 * depth + 1) * theta_true) ** 2)
            return int(np.random.default_rng(seed * 7919 + depth).binomial(s, p_d))

        k_be = get_backend()
        res = k_be.iqae(measure, budget=n_queries, shots=shots,
                        grid=iqae_grid, seed=seed)
        # The posterior grid can quantize an interval edge a hair outside the
        # MAP estimate; a point estimate outside its own interval is a
        # reporting artifact, not a statistical property -- clamp it.
        lo = float(min(res.lo, res.p_map))
        hi = float(max(res.hi, res.p_map))
        return {"arm": "iqae-sim", "p_hat": res.p_map,
                "ci_lo": lo, "ci_hi": hi,
                "width": hi - lo, "n_queries": res.n_queries,
                "max_depth": res.max_depth}

    raise NotImplementedError(
        f"estimator arm '{arm}' unknown (use q | restart | mc)")


def scenario_opt_eps(n: int, beta: float, d: int = 2) -> float:
    """Campi-Garatti a-posteriori bound: after n sampled scenarios with 0
    support violations, the violation probability is <= eps at confidence
    1-beta, where eps is the (1-beta)-quantile of Beta(d, n-d+1)."""
    from scipy import stats
    return float(stats.beta.ppf(1 - beta, d, n - d + 1))
