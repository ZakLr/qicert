"""Syndrome-shadow monitor: false-alarm rate vs theorem budget, detection, latency.

All rows below are MEASURED with the real kernel objects
(qicert.kernels shadow_statistics / shadow_syndrome), on the stated
CPU edge profile:

  false-alarm rate : empirical alarm rate under the NULL (fresh
                     in-distribution latents) vs the gate's design FPR;
                     criterion: measured <= 2x design (the theorem budget)
  detection rates  : gray-list rate under (a) bit-flip corruption of the
                     compiled diagonal patterns (syndrome change) and
                     (b) brightness-drop-style input shift (shadow gate)
  added latency    : per-inference-step monitor cost (gate + syndrome)
"""
from __future__ import annotations

import time as _time

import numpy as np

from ._base import finish_run, start_run, table_header, wants

SMOKE = frozenset({"monitor-alarms"})

M_SHADOWS = 64
DIM = 32
FPR_DESIGN = 0.01
N_NULL_TRIALS = 4000
BLOCKS = 8
LATENCY_STEPS = 2000


def run(rows: str, out: list[str], ctx=None) -> None:
    if not wants(rows, "monitor-alarms", SMOKE):
        return

    from qicert.kernels import get_backend

    k = get_backend()
    rng = np.random.default_rng(0)
    P = rng.standard_normal((M_SHADOWS, DIM))

    # ---- baseline statistics (computed once on an in-distribution batch) ---
    X0 = rng.standard_normal((2000, DIM))
    s0 = (X0 @ P.T).ravel()                     # all projection scalars
    mu, sigma = float(s0.mean()), float(s0.std())

    def _gate(x: np.ndarray):
        return k.shadow_statistics(x, P, mu, sigma, blocks=BLOCKS,
                                   fpr=FPR_DESIGN)

    # ---- (1) empirical false-alarm rate under the null ---------------------
    alarms = 0
    for _ in range(N_NULL_TRIALS):
        v = _gate(rng.standard_normal(DIM))
        alarms += int(v.alarm)
    fpr_emp = alarms / N_NULL_TRIALS
    fpr_ok = fpr_emp <= 2 * FPR_DESIGN

    # ---- (2a) syndrome: bit-flip detection on compiled diagonal patterns ---
    base_pat = [rng.standard_normal(16) for _ in range(8)]
    syn0 = k.shadow_syndrome(base_pat)
    n_flip = 500
    flips_caught = 0
    for _ in range(n_flip):
        pat = [p.copy() for p in base_pat]
        j = int(rng.integers(0, 8))
        pat[j][int(rng.integers(0, 16))] += 0.5     # a real bit-flip scale hit
        flips_caught += int(k.shadow_syndrome(pat) != syn0)
    flip_rate = flips_caught / n_flip

    # ---- (2b) shadow gate: detection profile under latent drifts -----------
    # The gate is a LOCATION test on the median of block-means, so its
    # detection profile is direction-dependent:
    #   common-mode drift (all projections see the same sign) -> detected;
    #   random-sign drift (median robustness) -> NOT detected by design.
    # Both regimes are measured and reported — the second is a documented
    # limitation that motivates the conformal gate + Layer-1 fallback.
    w_common = P.sum(axis=0)
    w_common /= np.linalg.norm(w_common)      # direction all rows see +
    rng_dir = np.random.default_rng(5)
    w_random = rng_dir.standard_normal(DIM)
    w_random /= np.linalg.norm(w_random)
    det = {}
    for name, delta in (("0.5 sigma", 0.5), ("1.0 sigma", 1.0),
                        ("2.0 sigma", 2.0)):
        hits = 0
        n = 500
        for _ in range(n):
            x = rng.standard_normal(DIM) + delta * sigma * w_common
            hits += int(_gate(x).alarm)
        det[name] = hits / n
    det_random = {}
    for name, delta in (("1.0 sigma", 1.0), ("2.0 sigma", 2.0)):
        hits = 0
        n = 500
        for _ in range(n):
            x = rng.standard_normal(DIM) + delta * sigma * w_random
            hits += int(_gate(x).alarm)
        det_random[name] = hits / n

    # ---- (3) added per-step latency ----------------------------------------
    x_step = rng.standard_normal(DIM)
    pat_step = [p.copy() for p in base_pat]
    t0 = _time.perf_counter()
    for _ in range(LATENCY_STEPS):
        _gate(x_step)
        k.shadow_syndrome(pat_step)
    per_step_ms = (_time.perf_counter() - t0) / LATENCY_STEPS * 1e3

    out += table_header(
        f"N10 - monitor metrics (measured, kernel shadow objects, "
        f"M={M_SHADOWS}, d={DIM}, design FPR={FPR_DESIGN})",
        ["Metric", "Budget", "Measured", "Status"])
    out.append(f"| false-alarm rate vs shadow bound | <= 2x design "
               f"({2 * FPR_DESIGN:.3f}) | {fpr_emp:.4f} over "
               f"{N_NULL_TRIALS} null trials | "
               f"{'PASS' if fpr_ok else 'FAIL'} |")
    out.append(f"| syndrome detection (bit-flip on diagonal patterns) | "
               f"- | {flip_rate:.3f} over {n_flip} flips | "
               f"{'PASS' if flip_rate >= 0.99 else 'FAIL'} |")
    out.append(f"| shift detection, common-mode drift 0.5/1.0/2.0 sigma | - | "
               f"{det['0.5 sigma']:.3f} / {det['1.0 sigma']:.3f} / "
               f"{det['2.0 sigma']:.3f} | measured |")
    out.append(f"| shift detection, random-sign drift 1.0/2.0 sigma | - | "
               f"{det_random['1.0 sigma']:.3f} / "
               f"{det_random['2.0 sigma']:.3f} | measured (median "
               f"robustness: by design; conformal gate is the complement) |")
    out.append(f"| added end-to-end latency | < 5 ms | "
               f"{per_step_ms:.4f} ms/step (gate + syndrome) | "
               f"{'PASS' if per_step_ms < 5 else 'FAIL'} |")
    out.append("")
    out.append("* The theorem budget: median-of-means gate with design FPR "
               f"{FPR_DESIGN}; the measured null alarm rate must sit within "
               f"2x of design (concentration theorem's practical slack). "
               f"The detection profile is the honest tradeoff against that "
               f"FPR: strong on common-mode corruption of the watched stream "
               f"(the PDU threat model: stuck sensors, saturated features), "
               f"deliberately blind to zero-mean random-sign drifts, which "
               f"is what the conformal alarm gate and the Layer-1 certified "
               f"fallback cover — the three mechanisms compose, none alone "
               f"is the monitor.")

    if ctx is not None and ctx.active:
        rec = start_run(ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "N10"),
                        label=f"monitor-metrics-fpr{FPR_DESIGN}",
                        config={"experiment": "monitor-alarms",
                                "design_fpr": FPR_DESIGN,
                                "n_null_trials": N_NULL_TRIALS,
                                "M": M_SHADOWS, "dim": DIM})
        if rec is not None:
            finish_run(rec, status="completed",
                       results={"fpr_empirical": fpr_emp,
                                "fpr_budget": 2 * FPR_DESIGN,
                                "flip_detection_rate": flip_rate,
                                "shift_detection_common_mode": det,
                                "shift_detection_random_sign": det_random,
                                "per_step_ms": round(per_step_ms, 4)})
