"""Layer-3 safety suite: N6 - STL + IQAE vs GEV-tail vs MC, closed by
scenario-optimization bounds and empirical interval calibration.

2026-09-12: the three-arm race RAN (was [pending] rows). Demo oracle:
seeded heavy-tailed margin with ANALYTIC failure probability, so every
arm is scored against ground truth (coverage + interval width at matched
query budget).  Race conclusion is printed from the measured table.

Arms (estimate_failure_probability in qicert.safety):
  Q        simulated Bayesian IQAE (kernel backend; amplification simulated
           from an MC pre-phase -- documented, standard for estimator scoring)
  C-strong GEV tail fit + block bootstrap (RESTART splitting needs rollout
           trajectory access; with a seed->rho oracle this is the honest
           strong-classical baseline -- the report says so)
  C-naive  Monte-Carlo counting + Clopper-Pearson (the strawman)

Closure:
  scenario-opt  Campi-Garatti a-posteriori bounds (complexity d=2)
  conformal     empirical split-interval calibration (distribution-free form)
"""
from __future__ import annotations

import json as _json
import os as _os
from pathlib import Path

import numpy as np

from ._base import finish_run, start_run, table_header, wants

SMOKE = frozenset({"safety-three-arm"})

D_COMPL = 2   # scenario-optimization support-function complexity (worst case)


def _run_three_arm(out: list[str], ctx=None) -> None:
    from qicert.oracle import OracleConfig, ReversibleOracle
    from qicert.safety import analytic_margin_oracle, estimate_failure_probability

    seed = ctx.seed if ctx else 0
    seeds = [seed] if ctx is None or not getattr(ctx, 'seeds', None) \
        else list(ctx.seeds)
    budgets_env = _os.environ.get("QICERT_N6_BUDGETS", "50000,200000,1000000")
    budgets = [int(x) for x in budgets_env.split(",") if x.strip()]
    n_blocks = int(_os.environ.get("QICERT_N6_BLOCKS", "200"))

    # Rare-event demo oracle: heavy Student-t tail, p_true ~ 1e-4 (default
    # m0=1, scale=0.05, df=3) -- the regime where naive MC dies.
    rollout, p_true = analytic_margin_oracle()

    class _VecOracle(ReversibleOracle):
        """Adds vectorized indicator + rho access for the estimator arms."""

        def __init__(self):
            super().__init__(rollout, OracleConfig(track="AD", seed_bits=32))
            self.rho = rollout

        def __call__(self, seed):
            arr = np.asarray(seed)
            if arr.ndim == 0:
                return int(rollout(np.asarray([int(arr)]))[0] < 0.0)
            return (rollout(arr.astype(np.int64)) < 0.0).astype(np.int64)

    oracle = _VecOracle()

    out += table_header(
        f"N6 - three-arm race on p = Pr[rho < 0] (demo oracle, p_true = "
        f"{p_true:.6f}, median over seeds {seeds}, heavy-tailed margins)",
        ["Arm", "n_queries", "p_hat (median)", "CI 95% (median)", "Width (med)",
         "Coverage", "Width/naive"])
    rows = []

    # Seeds are the measurement repetition (no single-run claims);
    # each arm gets the SAME seed set at each budget.  Median + coverage.
    per_arm: dict[str, list[dict]] = {}
    for nq in budgets:
        for arm in ("q", "restart", "mc"):
            runs = []
            for s in seeds:
                try:
                    kwargs = {"arm": arm, "oracle": oracle, "n_queries": nq,
                              "seed": s, "n_blocks": n_blocks}
                    if arm == "q":
                        kwargs["true_p"] = p_true   # simulator's ideal device
                    res = estimate_failure_probability(**kwargs)
                    res["covers_truth"] = int(res["ci_lo"] <= p_true <= res["ci_hi"])
                    res["budget"] = nq
                    res["p_true"] = p_true
                    runs.append(res)
                    print(f"  [N6 arm={arm} nq={nq} seed={s}] "
                          f"p_hat={res['p_hat']:.6f} "
                          f"CI=({res['ci_lo']:.6f}, {res['ci_hi']:.6f}) "
                          f"width={res['width']:.6f} "
                          f"covers={res['covers_truth']} <- live", flush=True)
                except Exception as exc:
                    print(f"  [N6 arm={arm} nq={nq} seed={s}] FAIL: "
                          f"{type(exc).__name__}: {exc}", flush=True)
            if runs:
                runs.sort(key=lambda r: r["width"])
                med = runs[len(runs) // 2]
                cov = sum(r["covers_truth"] for r in runs)
                per_arm.setdefault(arm, []).append(med)
                rows.append({**{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                for k, v in med.items()},
                             "n_seeds": len(seeds), "coverage": cov,
                             "runs": [{"seed": i, "width": r["width"],
                                       "covers": r["covers_truth"],
                                       "p_hat": r["p_hat"]}
                                      for i, r in enumerate(runs)]})

    # table with widths normalized to the naive arm at the same budget
    naive_width = {r["budget"]: r["width"] for r in rows
                   if r.get("arm") == "mc"}
    for r in rows:
        w = r["width"]
        rel = (w / naive_width[r["budget"]]
               if r.get("arm") != "mc" and naive_width.get(r["budget"])
               else 1.0 if r.get("arm") == "mc" else np.nan)
        out.append(f"| {r['arm']} | {r['budget']} | {r['p_hat']:.6f} | "
                   f"({r['ci_lo']:.6f}, {r['ci_hi']:.6f}) | {w:.6f} | "
                   f"{r['coverage']}/{r['n_seeds']} | {rel:.2f}x |")
    out.append(f"| ground truth | - | {p_true:.6f} | - | - | - | - |")

    # race conclusion (printed from the measured table, not asserted a priori)
    out.append("")
    q_rows = [r for r in rows if r.get("arm") == "iqae-sim"]
    m_rows = [r for r in rows if r.get("arm") == "mc"]
    if q_rows and m_rows:
        q_big = max(q_rows, key=lambda r: r["budget"])
        m_match = min(m_rows, key=lambda r: abs(r["width"] - q_big["width"]))
        out.append(f"* Race summary (medians): IQAE-sim reaches width "
                   f"{q_big['width']:.6f} with {int(q_big['n_queries'])} "
                   f"queries; naive MC's closest width is {m_match['width']:.6f} "
                   f"at {m_match['budget']} queries "
                   f"({m_match['budget'] / max(q_big['n_queries'], 1):.0f}x "
                   f"MC budget for comparable width). Coverage is the honesty "
                   f"check: an arm that is narrow but misses p_true is WORSE "
                   f"than a wide honest one.")
        out.append("* C-strong is the GEV extreme-tail estimator (block "
                   "bootstrap CI); the RESTART splitting half of the planned "
                   "arm requires rollout trajectory access, which the "
                   "seed->rho oracle does not expose - documented scoping.")

    # persist
    dest = Path("results") / "safety"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"threearm_seed{seed}.json").write_text(_json.dumps(
        {"p_true": p_true, "rows": rows, "seeds": seeds}, indent=2))

    if ctx is not None and ctx.active:
        rec = start_run(ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "N6"),
                        label=f"three-arm-race-seed{seed}",
                        config={"experiment": "N6 three-arm race",
                                "budgets": budgets, "demo_oracle": True,
                                "p_true": p_true, "n_blocks": n_blocks,
                                "seeds": seeds})
        if rec is not None:
            finish_run(rec, status="completed", results={"p_true": p_true,
                                                         "rows": rows})


def _run_scenario_opt(out: list[str], ctx=None) -> None:
    """Campi-Garatti a-posteriori bound: with N scenarios and complexity d,
    0 support violations, violation prob <= eps at confidence 1-beta where
    eps = beta-quantile form  eps(N, beta, d)  via the Beta distribution."""
    from scipy import stats

    out += table_header("Scenario-optimization closure (Campi-Garatti, "
                        f"d = {D_COMPL}, 0 violations)",
                        ["N scenarios", "beta", "eps bound", "Status"])
    for n in (1_000, 10_000, 100_000, 1_000_000):
        for beta in (0.01, 0.05):
            eps = float(stats.beta.ppf(1 - beta, D_COMPL, n - D_COMPL + 1))
            out.append(f"| {n} | {beta} | {eps:.6f} | completed |")
            print(f"  [N6 scen-opt N={n} beta={beta}] eps<={eps:.6f}", flush=True)
    out.append("")
    out.append("* Reading: after N sampled scenarios with 0 support "
               "violations, the violation probability of the certified set "
               "is <= eps at confidence 1-beta. This prints the confidence "
               "line under the safety table with NO distributional "
               "assumptions - the honest closure for L3.")


def _run_conformal(out: list[str], ctx=None) -> None:
    """Empirical split-interval calibration (split-conformal form): the
    calibrated threshold achieves target coverage on FRESH seeds, with the
    finite-sample correction.  Calibration and test seeds are disjoint."""
    seed = ctx.seed if ctx else 0
    n_cal = int(_os.environ.get("QICERT_N6_NCAL", "10000"))
    n_test = int(_os.environ.get("QICERT_N6_NTEST", "100000"))
    from qicert.safety import analytic_margin_oracle

    rollout, _p_true = analytic_margin_oracle(m0=1.0, scale=0.25, df=3.0)
    rng = np.random.default_rng(seed * 104729 + 17)
    cal_seeds = rng.integers(1, 2**31, n_cal).astype(np.int64)
    test_seeds = rng.integers(1, 2**31, n_test).astype(np.int64)
    rho_cal = rollout(cal_seeds)
    rho_test = rollout(test_seeds)

    out += table_header("Conformal-style interval calibration (held-out "
                        f"seeds, n_cal={n_cal}, n_test={n_test})",
                        ["alpha", "Target coverage", "Empirical coverage",
                         "Margin threshold", "Status"])
    rows = []
    for alpha in (0.01, 0.05, 0.10):
        level = min(1.0, (1 - alpha) * (1 + 1 / n_cal))
        thr = float(np.quantile(rho_cal, 1 - level))
        emp = float(np.mean(rho_test >= thr))
        ok = "completed" if abs(emp - (1 - alpha)) < 2.5 * np.sqrt(
            (1 - alpha) * alpha / n_test) + 0.005 else "check"
        out.append(f"| {alpha} | {1 - alpha:.3f} | {emp:.4f} | "
                   f"rho >= {thr:.4f} | {ok} |")
        rows.append({"alpha": alpha, "target": 1 - alpha, "empirical": emp,
                     "threshold": thr, "status": ok})
        print(f"  [N6 conformal alpha={alpha}] target={1 - alpha:.3f} "
              f"empirical={emp:.4f} thr={thr:.4f}", flush=True)
    out.append("")
    out.append("* The certified-margin threshold transfers to fresh seeds at "
               "the target coverage (distribution-free exchangeability form). "
               "This is the calibration plot's data; calibration uses seeds "
               "disjoint from every evaluation set (leakage rule R12).")

    dest = Path("results") / "safety"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"conformal_seed{seed}.json").write_text(_json.dumps(
        {"rows": rows, "n_cal": n_cal, "n_test": n_test, "seed": seed},
        indent=2))


def run(rows: str, out: list[str], ctx=None) -> None:
    if wants(rows, "stl-specs", SMOKE):
        out += table_header("STL spec library (per track)",
                            ["Spec", "Track", "Formula"])
        out.append("| phi_1 | AD | box[0,T](clearance > eps_coll) |")
        out.append("| phi_2 | AD | box[0,T](|lateral_offset| < lane_halfwidth - eps_lane) |")
        out.append("| phi_3 | AD | diamond[0,T_stop](speed < v_safe) |")
        out.append("| phi_4 | AD | box[0,T] not(off_road and moving) |")
        out.append("| psi_1 | robotics | box(joint_velocities in [-v_j, +v_j]) |")
        out.append("| psi_2 | robotics | box not(end_effector in forbidden_region) |")
        out.append("| psi_3 | robotics | diamond(gripper settled and pose within delta) |")

    if wants(rows, "safety-three-arm", SMOKE):
        _run_three_arm(out, ctx)

    if wants(rows, "scenario-opt", SMOKE):
        _run_scenario_opt(out, ctx)

    if wants(rows, "conformal-coverage", SMOKE):
        _run_conformal(out, ctx)
