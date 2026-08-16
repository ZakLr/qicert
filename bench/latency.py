"""Latency profile: the <=100 ms budget, four-way comparison."""
from __future__ import annotations

from ._base import table_header, wants

SMOKE = frozenset({"latency-profile"})


def run(rows: str, out: list[str], ctx=None) -> None:
    if wants(rows, "latency-profile", SMOKE):
        out += table_header("Latency on stated profile (<=100 ms)",
                            ["Method", "ms", "Status"])
        out.append("| dense O(N^2) reference attention | - | [pending] |")
        out.append("| tuned Performer / FAVOR+ (matched capacity) | - | [pending] |")
        out.append("| qicert compiled families (full table) | - | [pending] |")
        out.append("| qicert compiled + certified pruning budget | - | [pending] |")
