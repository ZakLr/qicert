"""Syndrome-shadow monitor: N10 - false-alarm vs theorem, correction rate, latency."""
from __future__ import annotations

from ._base import pending_row, table_header, wants

SMOKE = frozenset({"monitor-alarms"})


def run(rows: str, out: list[str], ctx=None) -> None:
    if wants(rows, "monitor-alarms", SMOKE):
        out += table_header("N10 - monitor metrics (pre-registered)",
                            ["Metric", "Budget", "Status"])
        out.append("| false-alarm rate vs shadow bound | <= 2x theorem | [pending] |")
        out.append("| correction rate (bit-flip + brightness-drop) | - | [pending] |")
        out.append("| added end-to-end latency | < 5 ms | [pending] |")
