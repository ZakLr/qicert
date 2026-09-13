"""Cross-track mirror: the same compression + certificate pipeline on the second backbone."""
from __future__ import annotations

from ._base import pending_row, table_header, wants

SMOKE = frozenset({"cross-track"})


def run(rows: str, out: list[str], ctx=None) -> None:
    if wants(rows, "cross-track", SMOKE):
        out += table_header("N13 - cross-track generalization (AD -> robotics mirror of N2/N3/N5)",
                            ["ID", "Experiment", "Seeds", "GPU-h", "Status"])
        out.append(pending_row("N13", "N2/N3/N5 mirror on second backbone", "3", "12"))
