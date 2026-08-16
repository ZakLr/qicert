"""Step 0 — single-layer warm-up smoke on a REAL MiniVLA layer.

Warm-up ladder (docs/backbones.md §6, step 1): pull one weight matrix out of
the MiniVLA checkpoint and push it through the qicert kernel pipeline:

    1. TT-SVD + TT-cross of the layer weight (compression stage)
    2. exact Lipschitz product vs tight operator norm vs dense norm (L1 cert)
    3. INT8 (torch.quantization-style) + SVD at matched ratio (baselines)
    4. compiler identity on one interaction block (commuting-Pauli families)

This is a *toolchain* smoke, NOT a scored experiment (AGENTS.md: noiseless
sanity check — "is the pipeline right", not "does it win"). Scored tables
are N1/N2/N3; this row exists to catch kernel/wiring bugs on a real layer
before any GPU-h is spent. No seed variance is claimed; results are for
pipeline validation only.

Run (torch present; container):
    python -m qicert.bench.all --module step0 --rows=layer-smoke --out results \
        --exp-id STEP0 --run-tag minivla-qproj

Kept OUT of bench.all's default MODULES: the default all/CI smoke must stay
torch-free (clean-env contract), and this row needs the checkpoint on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._base import finish_run, start_run, table_header, wants

SMOKE = frozenset()  # never part of --rows=smoke (needs torch + weights on disk)

# Layer to certify: Qwen2.5-0.5B self-attn q_proj (896x896 in MiniVLA).
DEFAULT_CKPT = "weights/ckpt/checkpoints/step-122500-epoch-55-loss=0.0743.pt"
# Checkpoint layout: sd["model"]["llm_backbone"] is a flat dict of dotted
# keys, e.g. "llm.model.layers.0.self_attn.q_proj.weight" (verified 2026-08-16).
BACKBONE_KEY = "llm_backbone"
LAYER_KEY = "llm.model.layers.0.self_attn.q_proj.weight"
M_DIMS = (32, 28)
N_DIMS = (32, 28)
RANKS = (8,)


def _load_layer(ckpt: str, key: str) -> np.ndarray:
    import torch  # local import: this module is torch-optional
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt} (run "
                                "scripts/download_backbones.py first)")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m = sd.get("model", sd)
    # m["llm_backbone"] holds flat dotted keys; look the full key up directly.
    return m[BACKBONE_KEY][key].detach().float().cpu().numpy()


def _int8_roundtrip(W: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-tensor symmetric INT8 quantize->dequantize (torch.quantization
    semantics for a weight tensor; CPU fallback on Blackwell per backbones.md
    §4 — v1 policy: torch.quantization, no bitsandbytes)."""
    scale = float(np.abs(W).max()) / 127.0
    q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    Wq = q.astype(np.float32) * scale
    return Wq, {"scale": scale, "dtype": "int8-symmetric-per-tensor"}


def _svd_rank_for(W: np.ndarray, tt_params: int) -> int:
    """SVD rank that matches the TT parameter budget (matched-ratio A1)."""
    M, N = W.shape
    r = max(1, int(round(tt_params / (M + N))))
    return min(r, min(M, N))


def run(rows: str, out: list[str], ctx=None) -> None:
    if not wants(rows, "layer-smoke", SMOKE):
        return

    ckpt = getattr(ctx, "layer_ckpt", None) or DEFAULT_CKPT
    W = _load_layer(ckpt, LAYER_KEY)
    M, N = W.shape

    rec = start_run(ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "STEP0"),
                    label="layer-smoke",
                    config={"layer": LAYER_KEY, "ckpt": ckpt,
                            "m_dims": list(M_DIMS), "n_dims": list(N_DIMS),
                            "ranks": list(RANKS),
                            "note": "toolchain warm-up, not a scored run"})

    from qicert.kernels import get_backend
    k = get_backend()

    out += table_header(
        "Step 0 - single-layer kernel smoke on a real MiniVLA layer "
        f"(q_proj {M}x{N}, ranks={RANKS})",
        ["Stage", "Metric", "Value", "Status"])

    rows_out: list[tuple[str, str, float, bool]] = []

    # 1. compression: TT-SVD vs TT-cross reconstruction
    cs_svd = k.tt_svd(W, M_DIMS, N_DIMS, RANKS)
    cs_cross = k.tt_cross(W, M_DIMS, N_DIMS, RANKS)
    rec_svd = float(np.linalg.norm(k.contract_cores(cs_svd.arrays, M_DIMS, N_DIMS) - W)
                    / np.linalg.norm(W))
    rec_cross = float(np.linalg.norm(k.contract_cores(cs_cross.arrays, M_DIMS, N_DIMS) - W)
                      / np.linalg.norm(W))
    rows_out.append(("compression", "TT-SVD reconstruction rel. err", rec_svd, True))
    rows_out.append(("compression", "TT-cross reconstruction rel. err", rec_cross, True))

    # 2. Layer-1 certificate: product bound vs tight vs dense
    L_prod = k.lipschitz_product(cs_cross.arrays)
    L_tight = k.operator_norm_tight(cs_cross.arrays, M_DIMS, N_DIMS)
    L_dense = float(np.linalg.norm(W, 2))
    rows_out.append(("cert-l1", "Lipschitz product bound", L_prod, True))
    rows_out.append(("cert-l1", "tight op-norm (Rayleigh it.)", L_tight, True))
    rows_out.append(("cert-l1", "dense op-norm (SVD)", L_dense, True))
    tight_ratio = abs(L_prod - L_dense) / max(L_dense, 1e-12)
    rows_out.append(("cert-l1", "product/dense relative gap", tight_ratio, True))

    # 3. baselines at matched budget: INT8 + SVD with same parameter count
    Wq, i8_meta = _int8_roundtrip(W)
    i8_err = float(np.linalg.norm(Wq - W) / np.linalg.norm(W))
    tt_params = sum(int(np.prod(g.shape)) for g in cs_cross.arrays)
    r_svd = _svd_rank_for(W, tt_params)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    Wsvd = (U[:, :r_svd] * S[:r_svd]) @ Vt[:r_svd, :]
    svd_err = float(np.linalg.norm(Wsvd - W) / np.linalg.norm(W))
    # TT-SVD is the *best* TT approximation at these ranks: it must be within
    # a small factor of the matched-budget SVD (kernel correctness, not a
    # compression claim — real q_proj weights are high-rank at 1.8% params).
    tt_svd_ratio = rec_svd / max(svd_err, 1e-12)
    # tt_cross is a heuristic: it must track TT-SVD quality within 2x.
    cross_ratio = rec_cross / max(rec_svd, 1e-12)
    rows_out.append(("baseline", f"INT8 ({i8_meta['dtype']}) rel. err", i8_err, True))
    # SVD at matched budget is *informative* (what a classical baseline achieves
    # at the same parameter count) — no tolerance: it is the reference, not a check.
    rows_out.append(("info", f"SVD r={r_svd} (params={tt_params}) rel. err",
                     svd_err, True))
    rows_out.append(("compression", "TT-SVD err / SVD-matched err", tt_svd_ratio, True))
    rows_out.append(("compression", "TT-cross err / TT-SVD err", cross_ratio, True))

    # 4. compiler identity on one interaction block (3-qubit commuting family)
    fam = [(1.0, "ZII"), (0.5, "ZIZ"), (0.25, "IZZ")]
    pg = k.pauli_grouping([(c, s) for c, s in fam], k=3)
    dg = k.pauli_diagonalize(pg.families[0], k=3)
    Uc = dg.unitary
    T = sum(c * k.pauli_matrix(s, 3) for c, s in fam)
    comp_ident = float(np.linalg.norm(
        Uc @ np.diag(np.diag(Uc.conj().T @ T @ Uc)) @ Uc.conj().T - T))
    rows_out.append(("compiler", "interaction-block identity err", comp_ident, True))

    # tolerances: kernel-correctness and certificate-soundness checks only.
    # Compression QUALITY is a scored N2 question — the smoke must not pretend
    # rank-8 on a real layer is a good operating point (AGENTS.md: noiseless
    # results are proofs of concept, and this is a toolchain smoke, not a win).
    tol = {
        "compression": 2.0,    # TT-SVD within 2x of matched-budget SVD; cross within 2x of TT-SVD
        "cert-l1": None,       # informative values, no fixed tolerance
        "baseline": 1e-1,      # INT8 reference quality
        "compiler": 1e-9,      # exact identity — must be machine-exact
    }
    # certificate soundness: product bound must upper-bound the tight norm
    bound_sound = bool(L_prod >= L_tight * (1 - 1e-9))
    out.append(f"| cert-l1 | product bound >= tight norm (soundness) | "
               f"{L_prod:.3e} >= {L_tight:.3e} | - | {'PASS' if bound_sound else 'FAIL'} |")
    if rec:
        rec.metric(stage="cert-l1", metric="product bound >= tight norm (soundness)",
                   value=float(L_prod), tol=None, pass_ok=bound_sound)

    all_ok = bound_sound
    for stage, name, val, _ in rows_out:
        t = tol.get(stage)
        if t is None:
            ok, bar = True, "-"
        else:
            ok = val <= t
            bar = f"< {t:.0e}"
        all_ok = all_ok and ok
        out.append(f"| {stage} | {name} | {val:.3e} | {bar} | {'PASS' if ok else 'FAIL'} |")
        if rec:
            rec.metric(stage=stage, metric=name, value=float(val),
                       tol=tol.get(stage), pass_ok=bool(ok))

    # parameter-count note for the report's resource honesty
    out.append("")
    out.append(f"* TT parameters: {tt_params} vs dense {M * N} "
               f"({tt_params / (M * N) * 100:.2f}% of dense) at rank {RANKS}")

    if rec:
        rec.sample_power()
        finish_run(rec, status="completed" if all_ok else "failed",
                   results={
                       "layer": LAYER_KEY, "shape": [M, N],
                       "ranks": list(RANKS),
                       "tt_params": tt_params, "dense_params": M * N,
                       "rec_svd": rec_svd, "rec_cross": rec_cross,
                       "tt_svd_over_svd": tt_svd_ratio,
                       "cross_over_svd": cross_ratio,
                       "L_product": L_prod, "L_tight": L_tight,
                       "L_dense": L_dense,
                       "int8_err": i8_err, "int8_meta": i8_meta,
                       "svd_rank": r_svd, "svd_err": svd_err,
                       "compiler_identity_err": comp_ident,
                       "kill_criterion": "toolchain smoke — no kill criterion",
                       "verdict": "warm-up only, not a scored claim"},
                   tolerance_note="toolchain validation tolerances, see bench/step0.py")
        out.append(f"\nrecorded: {rec.run_dir}")


if __name__ == "__main__":  # pragma: no cover - convenience
    run("layer-smoke", [])
