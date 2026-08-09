"""Training engine: N11 (ALS/DMRG-native fine-tune), N12 (safety-critical last pass)."""
from __future__ import annotations

from ._base import pending_row, table_header, wants

SMOKE = frozenset({"training-curves"})


def run(rows: str, out: list[str]) -> None:
    if wants(rows, "training-curves", SMOKE):
        out += table_header("N11 - ALS/DMRG-native fine-tune (both backbones)",
                            ["ID", "Experiment", "Seeds", "GPU-h", "Status"])
        out.append(pending_row("N11", "compressed training curves + final accuracy + margin", "3", "8"))

    if wants(rows, "tail-shrinkage", SMOKE):
        out += table_header("N12 - safety-critical last pass (P3)",
                            ["ID", "Experiment", "Seeds", "GPU-h", "Status"])
        out.append(pending_row("N12", "falsification rounds vs tail mass", "3", "3"))
