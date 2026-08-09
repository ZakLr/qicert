"""Safety evaluation engine (N6): STL specs + three-arm estimator race.

Arms: Q (IQAE, simulated, Bayesian posterior) vs C-strong (RESTART + GEV)
vs C-naive (Monte-Carlo). Closed by Campi-Garatti scenario-optimization
bounds and split-conformal calibration.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class STLSpec:
    """A formal safety requirement with quantitative robustness rho."""
    name: str
    formula: str          # e.g. "box[0,T](clearance > eps_coll)"
    track: str            # "AD" | "robotics"
    predicates: list[str]


AD_SPECS = [
    STLSpec("phi_1", "box[0,T](clearance(ego, agents) > eps_coll)", "AD",
            ["clearance", "eps_coll"]),
    STLSpec("phi_2", "box[0,T](|lateral_offset| < lane_halfwidth - eps_lane)", "AD",
            ["lateral_offset", "lane_halfwidth", "eps_lane"]),
    STLSpec("phi_3", "diamond[0,T_stop](speed < v_safe)", "AD", ["speed", "v_safe"]),
    STLSpec("phi_4", "box[0,T] not(off_road and moving)", "AD", ["off_road", "moving"]),
]

ROBOTICS_SPECS = [
    STLSpec("psi_1", "box(joint_velocities in [-v_j, +v_j])", "robotics",
            ["joint_velocities", "v_j"]),
    STLSpec("psi_2", "box not(end_effector in forbidden_region)", "robotics",
            ["end_effector", "forbidden_region"]),
    STLSpec("psi_3", "diamond(gripper settled and task_pose within delta)", "robotics",
            ["gripper", "task_pose", "delta"]),
]


def estimate_failure_probability(arm: str, oracle, **kwargs):
    """Three-arm race on p = Pr[rho < 0].

    arm="q"  -> simulated IQAE (Bayesian posterior)
    arm="restart" -> RESTART importance splitting + GEV tail fit
    arm="mc" -> Monte-Carlo counting (cost shown, not run full)
    """
    raise NotImplementedError(f"estimator arm '{arm}' lands with N6.")
