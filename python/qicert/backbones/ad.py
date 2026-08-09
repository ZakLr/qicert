"""Autonomous-driving backbone adapter (N1, N13-AD).

Reference: Alpamayo-R1-adjacent action head on LLaVA/AD backbone, justified
per challenge Sec. 5.4. QLoRA fine-tune runs on Kaggle 2xT4.
"""
from __future__ import annotations


class ADBackbone:
    """Wraps the AD VLAM: load -> fine-tune (P1) -> compress (P2) -> certify."""

    def __init__(self, model_name: str = "llava-v1.5-7b", qlora: bool = True):
        self.model_name = model_name
        self.qlora = qlora

    def load(self):
        raise NotImplementedError("AD backbone download + license verify (Q11/Q19).")

    def fine_tune(self):
        raise NotImplementedError("N1 — QLoRA fine-tune, 3 seeds, ~8 GPU-h.")
