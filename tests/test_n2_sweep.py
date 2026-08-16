"""Unit tests for the N2 compression sweep helpers (bench/n2_sweep.py).

These test the pure helper math (factor dims, rank solver, param counts);
the GPU eval path is exercised by the bench runner itself (weights needed).
"""
import numpy as np
import pytest

from qicert.bench.n2_sweep import (
    BACKBONES,
    BOND_PLANS,
    _core_params,
    _factor_dims,
    _llm_linear_keys,
    _rank_for_fraction,
)


def test_factor_dims_bit_reversed():
    # 896 = 32 * 28; bit-reversed = largest first
    assert _factor_dims(896, 2) == (32, 28)
    # 4-mode split of 896, product preserved
    fd = _factor_dims(896, 4)
    assert len(fd) == 4 and int(np.prod(fd)) == 896
    assert fd[0] >= fd[-1]  # descending (bit-reversed semantics)
    # natural ordering is ascending-first
    nat = _factor_dims(896, 4, bit_reversed=False)
    assert nat[0] <= nat[-1] and int(np.prod(nat)) == 896


def test_core_params_matches_dense_at_full_rank():
    m_dims = (32, 28)
    n_dims = (32, 28)
    # d=2, single bond: params = r*(m0*n0 + m1*n1) for r>=2
    r = 8
    got = _core_params(r, m_dims, n_dims, 2)
    expected = r * (32 * 32 + 28 * 28)
    assert got == expected


@pytest.mark.parametrize("frac", [0.02, 0.04, 0.08, 0.16, 0.33, 0.50])
def test_rank_for_fraction_hits_target(frac):
    for d in (2, 4):
        m_dims = _factor_dims(896, d)
        n_dims = _factor_dims(896, d)
        r = _rank_for_fraction(frac, m_dims, n_dims)
        assert r >= 1
        actual = _core_params(r, m_dims, n_dims, d) / (896 * 896)
        # within ~2x of the target fraction (rank is integer-quantized)
        assert 0.5 * frac <= actual <= 2.0 * frac


def test_rank_for_fraction_monotonic():
    m_dims = _factor_dims(896, 2)
    n_dims = _factor_dims(896, 2)
    rs = [_rank_for_fraction(f, m_dims, n_dims) for f in BOND_PLANS]
    assert rs == sorted(rs)
    assert rs[0] >= 1


def test_llm_linear_keys_inventory_shape():
    # A fake flat state dict with the fork's key structure.
    sd = {
        "llm.model.layers.0.self_attn.q_proj.weight": np.zeros((4, 4)),
        "llm.model.layers.0.self_attn.k_proj.weight": np.zeros((4, 4)),
        "llm.model.layers.0.mlp.gate_proj.weight": np.zeros((4, 4)),
        "llm.model.layers.3.mlp.down_proj.weight": np.zeros((4, 4)),
        "llm.model.embed_tokens.weight": np.zeros((4, 4)),  # excluded
        "vision_backbone.dino.features": np.zeros((4, 4)),   # excluded
    }
    inv = _llm_linear_keys(sd)
    assert len(inv) == 4
    assert all("weight" in key for _, key in inv)
    assert any("layers.3" in key for _, key in inv)


def test_bond_plan_contract():
    # The pre-registered contract: 6 plans x 2 backbones.
    assert len(BOND_PLANS) == 6
    assert BACKBONES == ("TT", "QTT")
