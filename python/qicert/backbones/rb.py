"""Robotics backbone adapter (N4/N13-RB mirror).

Reference: MiniVLA/OpenVLA-class (matches OpenVLA-7B on LIBERO-90 at 2.5x speed).
"""
from __future__ import annotations


class RoboticsBackbone:
    """Wraps the robotics VLAM through the same 6-stage pipeline."""

    def __init__(self, model_name: str = "minivla-1b"):
        self.model_name = model_name

    def load(self):
        raise NotImplementedError("Robotics backbone download + license verify (Q11).")

    def fine_tune(self):
        raise NotImplementedError("N13 mirror — LoRA fine-tune, 3 seeds.")
