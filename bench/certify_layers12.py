"""Layer-2 certificates: N4 (SOS boxes), N5 (Lyapunov curve + predictor)."""
from __future__ import annotations

from ._base import pending_row, table_header, wants

SMOKE = frozenset({"lyapunov-survival"})


def run(rows: str, out: list[str]) -> None:
    if wants(rows, "sos-boxes", SMOKE):
        out += table_header("N4 - Layer-2a SOS-certified local boxes (3 Pareto points x 2 tracks)",
                            ["ID", "Box", "Margin m_k", "Certificate", "Status"])
        out.append("| N4 | - | - | SOS file in qicert.cert | [pending] needs Julia pin (Q22) |")

    if wants(rows, "lyapunov-survival", SMOKE):
        out += table_header("N5 - Layer-2b Lyapunov-margin degradation curve + predictor",
                            ["ID", "Experiment", "Seeds", "GPU-h", "Status"])
        out.append(pending_row("N5", "degradation curve + bond-spectrum predictor", "3", "6"))
