"""Commuting-Pauli compiler: N7 - exactness, pruning certs, latency, hardware table.

2026-09-12: rows are now MEASURED against the real kernel objects on the
stated CPU edge profile (same operator construction as bench.latency):

  E-comp-1  compiler identity exactness: relative error of the compiled
            full-partition evaluation vs the dense operator, gate <= 1e-6
  E-comp-2  per-family pruning certificates: predicted = ||sum_{a in F_j} c_a||_1
            (the kernel's pruning_cost); measured = ||T_j||_2 = max_l |d_j(l)|
            (exact for a diagonalized family).  Triangle inequality makes
            predicted >= measured PROVABLE; the scored statistic is their
            correlation across families (>= 0.95 = the certificate is
            informative, not just sound).
  E-comp-3  full-table latency on the CPU edge profile (<= 100 ms)
  E-comp-4  hardware-table vs noise-sweep self-consistency: honestly out of
            scope until a noise simulator is pinned (no fake number).
"""
from __future__ import annotations

import os as _os
import time as _time

import numpy as np

from ._base import finish_run, start_run, table_header, wants

SMOKE = frozenset({"compiler-acceptance", "family-pruning", "hardware-table"})

K_P = int(_os.environ.get("QICERT_N7_KP", "4"))
N_STRINGS = int(_os.environ.get("QICERT_N7_S", "48"))
B_BATCH = 64
REPEATS = 5
EPS_SHOTS = float(_os.environ.get("QICERT_N7_EPS", "0.05"))


def run(rows: str, out: list[str], ctx=None) -> None:
    if not (wants(rows, "compiler-acceptance", SMOKE)
            or wants(rows, "family-pruning", SMOKE)
            or wants(rows, "hardware-table", SMOKE)):
        return

    from qicert.kernels import get_backend

    k = get_backend()
    rng = np.random.default_rng(0)
    n_reg = 2 ** K_P

    # ---- the compiler's input: a Pauli-string interaction table ------------
    table = [(float(rng.standard_normal()), "".join(
        rng.choice(["I", "X", "Y", "Z"], size=K_P))) for _ in range(N_STRINGS)]
    grouping = k.pauli_grouping(table, K_P)
    families = list(grouping.families)
    prune_costs = list(grouping.pruning_cost)
    m = len(families)

    # ---- offline compile passes (pass 3/4) ---------------------------------
    units, diags = [], []
    for fam in families:
        dz = k.pauli_diagonalize(fam, K_P)
        d = np.zeros(n_reg, dtype=np.complex128)
        for (c_a, _p), lam in zip(fam, dz.eigenvalues):
            d += c_a * lam
        units.append(dz.unitary)
        diags.append(d)

    # ---- dense reference (all strings) -------------------------------------
    Tmat = np.zeros((n_reg, n_reg), dtype=np.complex128)
    for c_a, p in table:
        Tmat += c_a * k.pauli_matrix(p, K_P)
    X = (rng.standard_normal((n_reg, B_BATCH))
         + 1j * rng.standard_normal((n_reg, B_BATCH)))
    ref = Tmat @ X
    ref_scale = float(np.max(np.abs(ref)))

    def _compiled_all():
        acc = np.zeros_like(X)
        for U, d in zip(units, diags):
            acc += U @ (d[:, None] * (U.conj().T @ X))
        return acc

    # ---- E-comp-1: exactness ------------------------------------------------
    err_abs = float(np.max(np.abs(_compiled_all() - ref)))
    err_rel = err_abs / max(ref_scale, 1e-30)
    e1_pass = err_rel <= 1e-6

    # ---- E-comp-2: predicted (L1 cert) vs measured (||T_j||_2) per family ----
    # measured = exact spectral norm of the summed family operator
    meas = []
    for fam in families:
        Tj = np.zeros((n_reg, n_reg), dtype=np.complex128)
        for c_a, p in fam:
            Tj += c_a * k.pauli_matrix(p, K_P)
        meas.append(float(np.linalg.norm(Tj, 2)))
    measured_norm = meas
    pred = np.asarray(prune_costs, dtype=np.float64)
    meas = np.asarray(measured_norm, dtype=np.float64)
    sound = bool(np.all(pred >= meas - 1e-9))
    # Correlation between the L1 certificate and the true family norm is a
    # RANDOM VARIABLE (depends on the coefficient draw).  Single-table r is
    # not a result (no single-run claims): measure the median
    # over R independent random tables (random Gaussian coefficients = the
    # cancellation-heavy PESSIMISTIC case for an L1 bound; learned real
    # coefficients are typically sign-coherent, i.e. easier).
    n_tables = int(_os.environ.get("QICERT_N7_TABLES", "12"))
    table_corrs = [float("nan")]
    if pred.std() > 0 and meas.std() > 0:
        table_corrs[0] = float(np.corrcoef(pred, meas)[0, 1])
    for t in range(1, n_tables):
        rng_t = np.random.default_rng(100 + t)   # ONE rng per table; a fresh
        # rng per string would repeat the same draw 48x (bug caught by the
        # n_tables sanity print)
        rt = [(float(rng_t.standard_normal()),
               "".join(rng_t.choice(["I", "X", "Y", "Z"], size=K_P)))
              for _ in range(N_STRINGS)]
        gt = k.pauli_grouping(rt, K_P)
        pt = np.asarray(gt.pruning_cost, dtype=np.float64)
        mt = []
        for fam in gt.families:
            Tj = np.zeros((n_reg, n_reg), dtype=np.complex128)
            for c_a, p in fam:
                Tj += c_a * k.pauli_matrix(p, K_P)
            mt.append(float(np.linalg.norm(Tj, 2)))
        mt = np.asarray(mt)
        if pt.std() > 0 and mt.std() > 0:
            table_corrs.append(float(np.corrcoef(pt, mt)[0, 1]))
    corrs_arr = np.asarray([c for c in table_corrs if np.isfinite(c)])
    corr_med = float(np.median(corrs_arr)) if len(corrs_arr) else float("nan")
    corr_iqr = (float(np.quantile(corrs_arr, 0.25)),
                float(np.quantile(corrs_arr, 0.75))) if len(corrs_arr) \
        else (float("nan"), float("nan"))
    corr = table_corrs[0]          # first-table value for the family table
    e2_pass = bool(np.isfinite(corr_med) and corr_med >= 0.95)

    # ---- E-comp-3: full-table latency ---------------------------------------
    best = float("inf")
    for _ in range(2):
        _compiled_all()
    for _ in range(REPEATS):
        t0 = _time.perf_counter()
        _compiled_all()
        best = min(best, _time.perf_counter() - t0)
    ms_full = best * 1e3
    e3_pass = ms_full <= 100.0

    if wants(rows, "compiler-acceptance", SMOKE):
        out += table_header("N7 - compiler acceptance (E-comp-1..4)",
                            ["Check", "Criterion", "Measured", "Status"])
        out.append(f"| E-comp-1 | rel err <= 1e-6 | {err_rel:.2e} "
                   f"({err_abs:.2e} abs) | "
                   f"{'PASS' if e1_pass else 'FAIL'} |")
        out.append(f"| E-comp-2 | median predicted-vs-measured corr >= 0.95 "
                   f"over {len(corrs_arr)} random tables | median r = "
                   f"{corr_med:.4f} (IQR {corr_iqr[0]:.3f}-"
                   f"{corr_iqr[1]:.3f}; first-table r = {corr:.4f}; sound: "
                   f"{'yes' if sound else 'NO'}) | "
                   f"{'PASS' if e2_pass else 'FAIL'} |")
        out.append(f"| E-comp-3 | full-table eval <= 100 ms (CPU edge) | "
                   f"{ms_full:.3f} ms (2^{K_P} register, B={B_BATCH}) | "
                   f"{'PASS' if e3_pass else 'FAIL'} |")
        out.append(f"| E-comp-4 | hardware-table vs noise-sweep "
                   f"self-consistency | - | scoped out: needs a pinned "
                   f"noise simulator (future work; no fake number) |")

    if wants(rows, "family-pruning", SMOKE):
        out += table_header(
            "Per-family pruning certificates (predicted ||sum c||_1 vs "
            "measured ||T_j||_2 = max|d_j|, exact)",
            ["Family", "Predicted L1", "Measured norm", "Cert ratio",
             "Sound"])
        for j in range(m):
            ratio = pred[j] / max(meas[j], 1e-30)
            out.append(f"| F_{j} | {pred[j]:.4f} | {meas[j]:.4f} | "
                       f"{ratio:.2f}x | "
                       f"{'yes' if pred[j] >= meas[j] - 1e-9 else 'NO'} |")
        out.append("")
        out.append("* The certificate is provably sound (triangle inequality: "
                   "|sum c_a lambda_a| <= sum |c_a|, verified per family). "
                   "Its INFORMATIVENESS is scored by correlation to the true "
                   "family norm, measured over independent random tables "
                   "(random Gaussian coefficients are the cancellation-heavy "
                   "pessimistic case; sign-coherent learned coefficients are "
                   "expected tighter). This is the predicted-vs-measured "
                   "column no approximation-based method can print.")

    if wants(rows, "hardware-table", SMOKE):
        out += table_header(
            "Hardware pathway (compiler pass 5): qubits/depth/shots per "
            "family at declared eps",
            ["Family", "Qubits", "Family depth (m)", "Shots >= (L1/eps)^2",
             "Status"])
        for j in range(m):
            shots = int(np.ceil((pred[j] / EPS_SHOTS) ** 2))
            out.append(f"| F_{j} | {K_P} | {m} | {shots} | computed |")
        out.append(f"| FULL TABLE | {K_P} register qubits | depth = {m} "
                   f"families (sequential) | sum over families | computed |")
        out.append("")
        out.append(f"* Shot formula: n >= (L1_j / eps)^2 at declared "
                   f"eps = {EPS_SHOTS} (standard Hoeffding-style sampling "
                   f"bound); qubits = register size; no hardware claimed - "
                   f"this is the printed pathway the challenge asks for.")

    if ctx is not None and ctx.active:
        rec = start_run(ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "N7"),
                        label=f"compiler-acceptance-kp{K_P}",
                        config={"experiment": "compiler-acceptance",
                                "register_qubits": K_P,
                                "n_strings": N_STRINGS, "n_families": m,
                                "eps_shots": EPS_SHOTS})
        if rec is not None:
            finish_run(rec, status="completed",
                       results={"e1_rel_err": err_rel, "e1_pass": e1_pass,
                                "e2_corr_median": corr_med,
                                "e2_corr_iqr": corr_iqr,
                                "e2_n_tables": len(corrs_arr),
                                "e2_sound": sound, "e2_pass": e2_pass,
                                "e3_full_table_ms": round(ms_full, 4),
                                "e3_pass": e3_pass})
