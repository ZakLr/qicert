"""Compression + Layer-1 certificates: N1 (baseline), N2, N2', N3 (submission/08)."""
from __future__ import annotations

from ._base import finish_run, pending_row, start_run, table_header, wants

SMOKE = frozenset({"compression-pareto", "kernel-smoke"})


def run(rows: str, out: list[str], ctx=None) -> None:
    # --- Phase 0.5: the Python reference kernels measured for real ---
    if wants(rows, "kernel-smoke", SMOKE):
        from qicert.kernels import parity_check, active_backend_name
        rec = start_run(ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "kernel-smoke"),
                        label="kernel-smoke",
                        config={"check": "kernel-smoke",
                                "backend": active_backend_name(),
                                "parity": "tests/test_kernels.py::test_parity_harness"})
        fp = parity_check()
        out += table_header(
            f"Phase-0.5 - kernel smoke (backend={active_backend_name()})",
            ["Check", "Value", "Tolerance", "Status"])
        checks = [
            ("TT-SVD reconstruction rel. err", fp["tt_svd_rel_err"], 1e-10, "lower"),
            ("Lipschitz tight-vs-dense rel. err", fp["lipschitz_tight_vs_dense"], 1e-4, "lower"),
            ("Pauli diagonalization identity err", fp["pauli_identity_err"], 1e-9, "lower"),
            ("IQAE interval covers known p", fp["iqae_interval_contains_p"], 1.0, "exact"),
        ]
        all_ok = True
        for name, val, tol, mode in checks:
            ok = (val == tol) if mode == "exact" else val <= tol
            all_ok = all_ok and ok
            bar = "= 1" if mode == "exact" else f"< {tol:.0e}"
            out.append(f"| {name} | {val:.3e} | {bar} | {'PASS' if ok else 'FAIL'} |")
            if rec:
                rec.metric(check=name, value=float(val), tol=tol, pass_ok=bool(ok))
        if rec:
            rec.sample_power()
            finish_run(rec, status="completed" if all_ok else "failed",
                       results={"checks": {k: float(v) for k, v in fp.items()}},
                       tolerance_note="parity harness tolerances per tests/test_kernels.py")
            out.append(f"\nrecorded: {rec.run_dir}")

    # --- N1: the load-bearing baseline (QLoRA + INT8 reference) ---
    if wants(rows, "baseline-int8", SMOKE):
        out += table_header("N1 - AD baseline: QLoRA fine-tune + INT8 bnb reference",
                            ["ID", "Experiment", "Seeds", "GPU-h", "Status"])
        out.append(pending_row("N1", "AD backbone fine-tune (QLoRA) + INT8 bnb reference", "3", "8"))

    # --- N2: {TT, QTT} x TT-cross sweep ---
    if wants(rows, "compression-pareto", SMOKE):
        out += table_header("N2 - compression Pareto curve (ratio vs accuracy vs certified-safe-set)",
                            ["ID", "Experiment", "Seeds", "GPU-h", "Status"])
        out.append(pending_row("N2", "{TT, QTT} x TT-cross sweep, 6 bond plans x 2 backbones",
                               "3", "24"))
        out.append(pending_row("N2'", "bit-ordering sensitivity (3 orderings, 1 layer, 1 seed)",
                               "1", "2"))

    # --- N3: Layer-1 certificate table ---
    if wants(rows, "lipschitz-table", SMOKE):
        out += table_header("N3 - Layer-1 exact Lipschitz table (L = prod_i ||G_i||_2)",
                            ["Layer", "Ratio", "Lipschitz", "Safe-set %", "Status"])
        out.append("| - | - | - | - | [pending] needs cores from N2 |")

    # --- N14: clean-env reproducibility audit ---
    if wants(rows, "clean-env-audit", SMOKE):
        out += table_header("N14 - clean-env reproducibility audit (fresh venv)",
                            ["ID", "Check", "GPU-h", "Status"])
        out.append("| N14 | pip install qicert + bench.all --rows=smoke on fresh venv | 4 | [pending] CI runs the smoke set |")
