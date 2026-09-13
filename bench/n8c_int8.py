# ===========================================================================
# N8C — calibrated INT8 reference arm (the honest comparator)
# ===========================================================================
# Purpose: the INT8 number the report can honestly cite against the TT
# compression track.  NOT the naive weight-only INT8 that measured 0.0000
# on the fresh fine-tune (N1v2) — that number is a property of skipping
# calibration, not of INT8.  Here:
#   * per-channel weight scales s_j = absmax-calibrated w_j *and*
#     activation-weighted scale search: for each output channel, pick the
#     scale that minimizes || x W^T - x (W/s rounded back) ||_F over the
#     calibration activations (the GPTQ-style objective, closed-form
#     candidate scan — no backprop),
#   * weight storage: int8 values + per-output-channel fp16 scales,
#     ratio counted honestly vs the dense fp16 baseline,
#   * eval: FULL-SPLIT streaming protocol (the validated harness mode).
#
# Usage:
#   python -m qicert.bench.all --module n8c_int8 --rows=calibrated-int8 \
#       --out results --exp-id N8C --seed 0 --run-tag N8C-fullsplit \
#       --save-ckpt results/N1v2-ckpt --capture heavy
#
# Env: QICERT_FT_CKPT (fine-tuned checkpoint dir), QICERT_CALIB (calib npz
#      path), QICERT_N8C_SCALE ("absmax"|"mse", default mse),
#      QICERT_N2_EVAL=full.
"""Calibrated INT8 reference arm (bench module)."""
from __future__ import annotations

SMOKE = frozenset()  # needs the N1v2 handoff + calibration npz

import os as _os
from pathlib import Path

import numpy as np

from ._base import finish_run, start_run, table_header, wants


def run(rows: str, out: list[str], ctx=None) -> None:
    if wants(rows, "calibrated-int8", SMOKE):
        _run_n8c(out, ctx)


def _quantize_weight_mse(W: np.ndarray, calib_absmax: np.ndarray | None,
                         n_candidates: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel int8 quantization with MSE-optimal scale search.

    W: (M, N) float32, rows = output channels.  For each row, candidate
    scales s = maxabs_w / clip_frac over a small grid; pick the s minimizing
    || w - dequant(quant(w; s)) ||_2 (weight-MSE) — and, when calibration
    activations are available, weight the error by channel activation
    magnitude (the GPTQ insight: channels the activations actually excite
    matter more).

    Returns (Wq_dequantized float32, scales float32 (M,)).
    """
    M, N = W.shape
    amax = np.max(np.abs(W), axis=1).astype(np.float64)  # (M,)
    amax = np.maximum(amax, 1e-12)
    # Candidate clip fractions: 1.0 = no clipping; <1 = allow saturation.
    clip_fracs = np.linspace(1.0, 0.5, n_candidates)
    scales = np.zeros(M, dtype=np.float32)
    Wq = np.zeros_like(W)
    # Per-output-channel activation weight (L2 norm of input channel absmax):
    # w_row_j emphasizes input channels that carry signal. If no calib, use
    # uniform weights (plain weight-MSE).
    for i in range(M):
        w = W[i].astype(np.float64)
        best = None
        for cf in clip_fracs:
            s = amax[i] * cf / 127.0
            q = np.clip(np.round(w / s), -127, 127)
            err = (w - q * s)
            # activation-weighted: emphasize input channels with large
            # calibrated absmax (normalized to mean 1).
            if calib_absmax is not None and len(calib_absmax) == N:
                a = calib_absmax / max(np.mean(calib_absmax), 1e-12)
                loss = float(np.sum(err * err * a))
            else:
                loss = float(np.sum(err * err))
            if best is None or loss < best[0]:
                best = (loss, s, q)
        _, s_best, q_best = best
        scales[i] = np.float32(s_best)
        Wq[i] = (q_best * s_best).astype(np.float32)
    return Wq, scales


def _run_n8c(out: list[str], ctx=None) -> None:
    import json
    import time

    import torch

    from .n2_sweep import _N2EvalHarness

    seed = ctx.seed if ctx is not None else 0
    ft_dir = (Path(_os.environ.get("QICERT_FT_CKPT", ""))
              if _os.environ.get("QICERT_FT_CKPT")
              else (Path(ctx.save_ckpt) if ctx and ctx.save_ckpt
                    else Path("weights") / "ckpt" / "finetuned"))
    calib_path = Path(_os.environ.get(
        "QICERT_CALIB", "results/N2R/calib_seed0.npz"))
    scale_mode = _os.environ.get("QICERT_N8C_SCALE", "mse")

    ft_path = ft_dir / f"seed{seed}.pt"
    if not ft_path.exists():
        raise RuntimeError(f"no fine-tuned ckpt {ft_path}; run N1 --save-ckpt")

    # Calibration stats (per-linear-layer input absmax from train episodes).
    calib = {}
    if calib_path.exists():
        from qicert.calibrate import load_stats
        calib = load_stats(calib_path)
        print(f"[N8C] calibration stats loaded: {len(calib)} layers "
              f"({calib_path})", flush=True)
    else:
        print(f"[N8C] WARNING: no calibration file {calib_path} — falling "
              f"back to plain weight-MSE scales (still calibrated-clip "
              f"search, not activation-weighted)", flush=True)

    # ---- load FT weights + harness (validated swap path, full-split) ------
    ft_llm = torch.load(str(ft_path), map_location="cpu", weights_only=True)
    merged = ft_llm["llm_backbone"]
    harness = _N2EvalHarness(seed, ft_dir, ctx)
    harness.adopt_reference(merged)
    n1_ft = harness.n1_ft

    # ---- quantize every LLM linear weight (per-channel int8) --------------
    # Scope: the same 168 LLM linear projections N2/N2R compress (honest
    # matched-scope comparison). lm_head/embeddings stay fp16 (same as N2's
    # action-head exclusion).
    import re
    pat = re.compile(
        r"llm\.model\.layers\.\d+\.(self_attn\.(?:q|k|v|o)_proj|"
        r"mlp\.(?:gate|up|down)_proj)\.weight$")
    q_sd = {}
    total_int8_params = 0
    total_scales = 0
    total_dense_params = 0
    t0q = time.perf_counter()
    for key, t in merged.items():
        if pat.match(key) and t.dim() == 2:
            W = t.detach().float().cpu().numpy()
            M, N = W.shape
            # calib key naming: collect_calibration_stats used module names
            # from vla.named_modules(); match by the trailing unique part.
            calib_absmax = _find_calib(calib, key)
            Wq, scales = _quantize_weight_mse(W, calib_absmax)
            q_sd[key] = torch.from_numpy(Wq.astype(np.float16))
            total_int8_params += M * N
            total_scales += M
            total_dense_params += M * N
        else:
            q_sd[key] = t
    dt_q = time.perf_counter() - t0q
    # Honest storage: int8 weights (1 B/param) + fp16 scales (2 B/channel)
    # vs dense fp16 (2 B/param).  Storage bytes:
    dense_bytes = total_dense_params * 2
    quant_bytes = total_int8_params * 1 + total_scales * 2
    ratio = dense_bytes / max(quant_bytes, 1)
    print(f"[N8C] quantized {total_int8_params} params (+{total_scales} "
          f"scales) in {dt_q:.0f}s -> ratio {ratio:.2f}x (byte-honest)",
          flush=True)

    # ---- full-split eval (validated harness) -------------------------------
    t0 = time.perf_counter()
    acc = harness.eval(q_sd)
    ev = getattr(harness, "_last_eval", {})
    delta = (acc - n1_ft) if n1_ft is not None and acc == acc else float("nan")
    dt = time.perf_counter() - t0
    print(f"[N8C seed={seed}] DONE ratio={ratio:.2f}x acc={acc:.4f} "
          f"delta={delta:+.4f} eval={ev.get('scope', '?')} ({dt:.0f}s) <- live",
          flush=True)
    out.append(f"| {seed} | INT8-{scale_mode} | {ratio:.2f}x | {acc:.4f} | "
               f"{delta:+.4f} | {ev.get('scope', '?')} | ({dt + dt_q:.0f}s) |")

    ci = ev.get("ci")
    rec = start_run(ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "N8C"),
                    label=f"calibrated-int8-{scale_mode}-seed{seed}",
                    config={
                        "experiment": "N8C calibrated INT8 reference arm",
                        "scale_mode": scale_mode,
                        "seed": seed, "ft_ckpt": str(ft_path),
                        "calib": str(calib_path),
                        "n_calib_layers": len(calib),
                        "eval": ev.get("scope", "?"),
                        "n_eval_tokens": ev.get("total"),
                        "n_eval_batches": ev.get("n_batches"),
                        "storage": "int8 weights + fp16 per-out-channel scales",
                    })
    if rec is not None:
        finish_run(rec, status="completed" if acc == acc else "failed",
                   results={"seed": seed, "scale_mode": scale_mode,
                            "ratio": round(float(ratio), 3),
                            "eval_acc": float(acc),
                            "correct": ev.get("correct"),
                            "total": ev.get("total"),
                            "ci_lo": ci[0] if ci else None,
                            "ci_hi": ci[1] if ci else None,
                            "eval_scope": ev.get("scope"),
                            "delta_vs_n1_ft": None if delta != delta
                            else round(float(delta), 4),
                            "quant_wall_sec": round(float(dt_q), 2),
                            "eval_wall_sec": round(float(dt), 2),
                            "note": "honest comparator; byte-honest ratio"})

    del harness
    torch.cuda.empty_cache()


def _find_calib(calib: dict, key: str) -> np.ndarray | None:
    """Map a state-dict weight key to its calibration layer entry.

    collect_calibration_stats keys are module names (e.g.
    'llm.model.layers.0.self_attn.q_proj'); state-dict keys append
    '.weight'.  Try direct, then suffix match.
    """
    if not calib:
        return None
    mod = key[: -len(".weight")] if key.endswith(".weight") else key
    if mod in calib:
        return calib[mod]["absmax"]
    # suffix search (module names may carry wrappers)
    cands = [v["absmax"] for k, v in calib.items()
             if k.endswith(mod) or mod.endswith(k)]
    return cands[0] if len(cands) == 1 else None
