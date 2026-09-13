"""PyTorch layers: TTLinear and FamilyAttention (Pillar B / training engine).

Phase-0 skeleton: signatures pinned per ``submission/11-package-spec.md``;
forward passes land with the tt_cross and pauli_family kernels.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TTLinear(nn.Module):
    """Linear layer stored as a chain of QTT/TT cores.

    The cores are the same objects whose spectral norms give the exact
    Lipschitz product (Layer-1 certificate) — no extra compute.
    """

    def __init__(self, in_features: int, out_features: int, bond_dim: int = 8,
                 mode_factors: tuple[int, ...] | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bond_dim = bond_dim
        # Core shapes are assigned by the safety-budgeted bond allocator
        # (compression budget follows certified margins, not heuristics).
        self.cores: nn.ParameterList = nn.ParameterList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "TTLinear.forward needs the tt_cross kernel (pending pinned GPU env)."
        )


class FamilyAttention(nn.Module):
    """Cross-modal interaction block compiled into commuting-Pauli families.

    The compiled table (pass 4 of the compiler) is an exact identity:
    T = sum_j C_j^dag D_j C_j. This layer applies the diagonal kernels.
    """

    def __init__(self, embed_dim: int, num_families: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_families = num_families

    def forward(self, vision: torch.Tensor, language: torch.Tensor,
                goal: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError(
            "FamilyAttention.forward needs the commuting-Pauli kernel (compiler stage)."
        )
