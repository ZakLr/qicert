"""Compression + Layer-1 certificates: N1 (baseline), N2, N2', N3 (submission/08)."""
from __future__ import annotations

from ._base import pending_row, table_header, wants

SMOKE = frozenset({"compression-pareto"})


def run(rows: str, out: list[str]) -> None:
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
