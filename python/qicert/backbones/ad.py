"""Autonomous-driving backbone adapter (N1, N13-AD).

Reference: reasoning-style action head on a vision-language backbone for
autonomous driving. Fine-tune via QLoRA (quantized low-rank adaptation).
"""
from __future__ import annotations


class ADBackbone:
    """Wraps the AD VLAM: load -> fine-tune (P1) -> compress (P2) -> certify."""

    def __init__(self, model_name: str = "llava-v1.5-7b", qlora: bool = True):
        self.model_name = model_name
        self.qlora = qlora

    def load(self):
        raise NotImplementedError("Autonomous-driving backbone download + license check.")

    def fine_tune(self):
        raise NotImplementedError("QLoRA fine-tune, 3 seeds.")
