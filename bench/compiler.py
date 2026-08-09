"""Commuting-Pauli compiler: N7 - exactness, pruning certs, latency, hardware table."""
from __future__ import annotations

from ._base import pending_row, table_header, wants

SMOKE = frozenset({"compiler-acceptance"})


def run(rows: str, out: list[str]) -> None:
    if wants(rows, "compiler-acceptance", SMOKE):
        out += table_header("N7 - compiler acceptance (E-comp-1..4)",
                            ["Check", "Criterion", "Status"])
        out.append("| E-comp-1 | ||T_exact - T_compiled|| <= 1e-6 relative (gate G2a) | [pending] |")
        out.append("| E-comp-2 | predicted vs measured prune error corr >= 0.95 | [pending] |")
        out.append("| E-comp-3 | latency <= 100 ms at full table | [pending] |")
        out.append("| E-comp-4 | hardware-table vs noise-sweep self-consistency | [pending] |")

    if wants(rows, "family-pruning", SMOKE):
        out += table_header("Per-family pruning certificates (||sum c_a||_1)",
                            ["Family", "Predicted ||sum c||_1", "Measured Delta", "Status"])
        out.append("| F_1..F_m | - | - | [pending] the column no other submission prints |")

    if wants(rows, "hardware-table", SMOKE):
        out += table_header("Hardware pathway (compiler pass 5 - Sec. 4.2)",
                            ["Family", "Qubits", "Depth", "Shots", "Status"])
        out.append("| - | ceil(log2 register) | m | f(||c||_1, eps) | [pending] from compiled artifact |")
