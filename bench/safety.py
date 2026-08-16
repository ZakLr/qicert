"""Layer-3 safety suite: N6 - STL + IQAE vs RESTART+GEV vs MC."""
from __future__ import annotations

from ._base import pending_row, table_header, wants

SMOKE = frozenset({"safety-three-arm"})


def run(rows: str, out: list[str], ctx=None) -> None:
    if wants(rows, "stl-specs", SMOKE):
        out += table_header("STL spec library (per track)",
                            ["Spec", "Track", "Formula"])
        out.append("| phi_1 | AD | box[0,T](clearance > eps_coll) |")
        out.append("| phi_2 | AD | box[0,T](|lateral_offset| < lane_halfwidth - eps_lane) |")
        out.append("| phi_3 | AD | diamond[0,T_stop](speed < v_safe) |")
        out.append("| phi_4 | AD | box[0,T] not(off_road and moving) |")
        out.append("| psi_1 | robotics | box(joint_velocities in [-v_j, +v_j]) |")
        out.append("| psi_2 | robotics | box not(end_effector in forbidden_region) |")
        out.append("| psi_3 | robotics | diamond(gripper settled and pose within delta) |")

    if wants(rows, "safety-three-arm", SMOKE):
        out += table_header("N6 - three-arm race on p = Pr[rho < 0]",
                            ["Arm", "Method", "Queries @p=1e-5", "Interval", "Status"])
        out.append("| Q | IQAE (simulated, Bayesian) | ~O(1/sqrt(p))*depth 1e2-1e3 | credible interval | [pending] |")
        out.append("| C-strong | RESTART + GEV | ~1e3-1e4 rollouts | profile-likelihood CI | [pending] |")
        out.append("| C-naive | Monte-Carlo counting | ~1e6+ (cost shown) | Clopper-Pearson | [pending] |")

    if wants(rows, "scenario-opt", SMOKE):
        out += table_header("Scenario-optimization closure (Campi-Garatti)",
                            ["N", "eps", "beta", "Status"])
        out.append("| - | - | - | [pending] printed from N6 |")

    if wants(rows, "conformal-coverage", SMOKE):
        out += table_header("Conformal calibration (held-out slice, leakage rule R12)",
                            ["Region", "Coverage", "Budget", "Status"])
        out.append("| - | - | - | [pending] printed from N6 |")
