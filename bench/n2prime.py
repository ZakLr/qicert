"""N2' — QTT bit-ordering sensitivity sweep (Q20; submission/08-experiments.md).

Pre-registered definition: compare 3 orderings (natural / bit-reversed /
interleaved) on 1 representative layer, 1 seed, at matched bond dimension;
choose by lowest reconstruction error; commit the winner before N2's
24-GPU-h sweep. Results are tagged by layer type so the conclusion
transfers to LLaVA-7B's same layer types (AD-transfer watch-list).

Layer: MiniVLA q_proj (896x896, attention projection) — tag
`attention-q-proj`. 1 seed (0). Matched bond: TT rank 8 (the smoke's
operating point).

Ordering -> mode-dim pairing on the TT-matrix kernel (each is a different
way of grouping the row/column index digits into the fused TT modes):
    natural      m_dims=(32, 28), n_dims=(32, 28)   row digit k pairs with col digit k
    bit-reversed m_dims=(28, 32), n_dims=(28, 32)   index digits read least-significant first
    interleaved  m_dims=(32, 28), n_dims=(28, 32)   row digit 0 pairs with col digit 1 (crossed)

All three reconstruct the SAME matrix W; only the mode grouping differs.
Reconstruction error at rank 8 decides. This is a scored-but-tiny
experiment: the winner becomes N2's bit-ordering default (Q20).

Run (container, weights mounted):
    python -m qicert.bench.all --module n2prime --rows=bit-ordering --out results \
        --exp-id N2prime --run-tag qproj-natural-vs-rev-vs-int
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ._base import finish_run, start_run, table_header, wants

SMOKE = frozenset()  # needs the checkpoint on disk; not part of --rows=smoke

CKPT = Path(__file__).resolve().parents[1] / "weights" / "ckpt" / "checkpoints" / \
    "step-122500-epoch-55-loss=0.0743.pt"
LAYER_KEY = "llm.model.layers.0.self_attn.q_proj.weight"
LAYER_TAG = "attention-q-proj"
RANKS = (8,)
# The three orderings as (m_dims, n_dims) mode groupings (see module docstring)
ORDERINGS = {
    "natural": ((32, 28), (32, 28)),
    "bit-reversed": ((28, 32), (28, 32)),
    "interleaved": ((32, 28), (28, 32)),
}


def _load_layer() -> np.ndarray:
    import torch  # local import: torch-optional module
    if not CKPT.exists():
        raise FileNotFoundError(f"checkpoint not found: {CKPT} (run "
                                "scripts/download_backbones.py first)")
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    return sd["model"]["llm_backbone"][LAYER_KEY].detach().float().cpu().numpy()


def run(rows: str, out: list[str], ctx=None) -> None:
    if not wants(rows, "bit-ordering", SMOKE):
        return

    W = _load_layer()
    M, N = W.shape

    rec = start_run(ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "N2prime"),
                    label="bit-ordering",
                    config={"layer": LAYER_KEY, "layer_type": LAYER_TAG,
                            "seed": ctx.seed if ctx else 0,
                            "ranks": list(RANKS),
                            "orderings": {k: {"m_dims": list(v[0]),
                                              "n_dims": list(v[1])}
                                          for k, v in ORDERINGS.items()},
                            "decision": "lowest reconstruction error at matched bond",
                            "note": "Q20 — winner becomes N2's bit-ordering default"})

    from qicert.kernels import get_backend
    k = get_backend()

    out += table_header(
        f"N2' - bit-ordering sweep ({LAYER_TAG}, {M}x{N}, rank={RANKS[0]}, seed={ctx.seed if ctx else 0})",
        ["Ordering", "Mode pairing (m_dims x n_dims)", "Recon rel. err", "Params", "Status"])

    results: dict[str, float] = {}
    rows_out: list[tuple[str, str, float]] = []
    for name, (m_dims, n_dims) in ORDERINGS.items():
        cs = k.tt_svd(W, m_dims, n_dims, RANKS)
        rec_ = float(np.linalg.norm(
            k.contract_cores(cs.arrays, m_dims, n_dims) - W) / np.linalg.norm(W))
        params = int(sum(np.prod(g.shape) for g in cs.arrays))
        results[name] = rec_
        rows_out.append((name, f"{m_dims} x {n_dims}", rec_))
        out.append(f"| {name} | {m_dims} x {n_dims} | {rec_:.6e} | {params} | - |")
        if rec:
            rec.metric(ordering=name, m_dims=list(m_dims), n_dims=list(n_dims),
                       recon_rel_err=float(rec_), params=params)

    winner = min(results, key=results.get)
    runner_up = sorted(results, key=results.get)[1]
    margin = (results[runner_up] - results[winner]) / max(results[winner], 1e-12)
    out.append("")
    out.append(f"* **Winner: {winner}** (recon {results[winner]:.4e}; "
               f"runner-up {runner_up} {results[runner_up]:.4e}; "
               f"relative margin {margin:.2f}x)")
    out.append(f"* Decision (Q20): N2's 24-GPU-h sweep uses the **{winner}** "
               f"ordering for {LAYER_TAG}-type layers.")

    if rec:
        rec.sample_power()
        finish_run(rec, status="completed",
                   results={"layer": LAYER_KEY, "layer_type": LAYER_TAG,
                            "ranks": list(RANKS),
                            "recon_err": {k: float(v) for k, v in results.items()},
                            "winner": winner,
                            "margin_vs_runner_up": round(margin, 4),
                            "kill_criterion": "no kill criterion — ordering choice, "
                                              "not a correctness claim",
                            "verdict": f"use {winner} for N2"},
                   tolerance_note="matched bond rank=8; see bench/n2prime.py")
        out.append(f"\nrecorded: {rec.run_dir}")
