# ===========================================================================
# Residual-compensated compression: TT-SVD + closed-form residual repair
# ===========================================================================
# Built directly on the uniform-truncation NO-GO result: uniform
# TT/QTT truncation collapsed the fine-tuned model (eval acc 0.4468 -> 0.0000
# at every ratio >= 2x, certificates sound).  The pre-registered repair keeps
# the TT reconstruction and folds back the best rank-r' approximation of the
# residual (Eckart-Young), with honest ratio accounting that INCLUDES the
# residual factors, and a per-layer sound Lipschitz bound for the repaired
# operator.
#
# Usage:
#   python -m qicert.bench.all --module n2r_sweep --rows=residual-go-no-go \
#       --out results --exp-id N2R --seed 0 --run-tag N2R-go-no-go-seed0 \
#       --save-ckpt results/N1v2-ckpt --capture heavy
#
# Env: QICERT_FT_CKPT (fine-tuned checkpoint dir), QICERT_N2R_PLANS
#      ("0.50,0.33"), QICERT_N2R_RRANKS ("8,32"), QICERT_N2_LAYERS
#      (all|layer0).
"""Residual-compensated compression sweep (bench module)."""
from __future__ import annotations

# Not part of the no-weights smoke set (needs fine-tuned weights + checkpoint).
SMOKE = frozenset()

import os as _os
from pathlib import Path

import numpy as np

from ._base import finish_run, start_run, table_header, wants
from .compress import CKPT, DATA_ROOT
from .n2_sweep import (
    BACKBONES,
    _factor_dims,
    _llm_linear_keys,
    _rank_for_fraction,
)


def run(rows: str, out: list[str], ctx=None) -> None:
    if wants(rows, "residual-go-no-go", SMOKE):
        _run_n2r(out, ctx)


def _plans_from_env() -> tuple[float, ...]:
    env = _os.environ.get("QICERT_N2R_PLANS", _os.environ.get("QICERT_N2_PLANS", "0.50,0.33"))
    return tuple(float(x) for x in env.split(",") if x.strip())


def _rranks_from_env() -> tuple[int, ...]:
    env = _os.environ.get("QICERT_N2R_RRANKS", "8,32")
    return tuple(int(x) for x in env.split(",") if x.strip())


def _run_n2r(out: list[str], ctx=None) -> None:
    import time

    import torch

    from qicert.compress_residual import (
        compressed_params,
        layer_report,
        lipschitz_bound,
        svd_residual,
    )
    from qicert.kernels import get_backend

    k = get_backend()

    seeds = (ctx.seeds if ctx and ctx.seeds else [ctx.seed if ctx else 0])
    ft_dir = (Path(_os.environ.get("QICERT_FT_CKPT", ""))
              if _os.environ.get("QICERT_FT_CKPT")
              else (Path(ctx.save_ckpt) if ctx and ctx.save_ckpt
                    else Path("weights") / "ckpt" / "finetuned"))
    layers_scope = _os.environ.get("QICERT_N2_LAYERS", "all")
    plans = _plans_from_env()
    rranks = _rranks_from_env()
    ttsplit = float(_os.environ.get("QICERT_N2R_TTSPLIT", "0.6"))
    bb = "TT"  # repair grid v1: TT only (QTT 16x needs more than a rank-32 patch)
    d = 2

    # --- load the BASE checkpoint for inventory + mode dims -----------------
    base_sd = torch.load(str(CKPT), map_location="cpu", weights_only=True)
    llm_sd = base_sd["model"]["llm_backbone"]
    inventory = _llm_linear_keys(llm_sd)
    if layers_scope == "layer0":
        inventory = [(t, key) for t, key in inventory if ".layers.0." in key]
    if not inventory:
        raise RuntimeError("no LLM linear layers found in checkpoint")

    plan_info = {}
    for layer_type, key in inventory:
        W = llm_sd[key].detach().float().cpu().numpy()
        M, N = W.shape
        m_dims, n_dims = _factor_dims(M, d), _factor_dims(N, d)
        plan_info[key] = {"layer_type": layer_type, "M": M, "N": N,
                          "dense_params": M * N, "m_dims": m_dims,
                          "n_dims": n_dims}
    # Release the 5.3 GB base checkpoint (inventory + shapes only).
    del base_sd, llm_sd
    import gc as _gc
    _gc.collect()

    # --- one VLA load per seed; every point reuses the same harness ---------
    # _N2EvalHarness gives us: the fixed matched eval batch, the FT reference
    # accuracy, the FT-restore-after-eval contamination guard, and the
    # dense-swap eval protocol.
    from .n2_sweep import _N2EvalHarness

    out += table_header(
        f"N2R — residual-compensated TT compression ({len(inventory)} layers, "
        f"plans={plans}, rranks={rranks}, tt_split={ttsplit}, seeds={seeds})",
        ["Seed", "Plan", "Rr", "Ratio", "Eval acc", "Delta vs N1 FT",
         "Cert", "Status"])

    for seed in seeds:
        if ctx is not None:
            ctx.seed = seed
        ft_path = ft_dir / f"seed{seed}.pt"
        if not ft_path.exists():
            out.append(f"| {seed} | - | - | - | - | - | - | "
                       f"FAIL (no fine-tuned ckpt {ft_path}; run N1 --save-ckpt) |")
            continue
        ft_llm = torch.load(str(ft_path), map_location="cpu", weights_only=True)
        merged = ft_llm["llm_backbone"]
        harness = _N2EvalHarness(seed, ft_dir, ctx)
        harness.adopt_reference(merged)  # FT weights live; swap path validated
        n1_ft = harness.n1_ft

        for frac in plans:
            for rrank in rranks:
                t0 = time.perf_counter()
                try:
                    print(f"  [N2R seed={seed} frac={frac:.2f} r'={rrank}] "
                          f"compress+repair {len(inventory)} layers...",
                          flush=True)
                    comp = dict(merged)
                    total_comp_params = 0
                    total_dense_params = 0
                    n_sound = 0
                    for i_layer, (layer_type, key) in enumerate(inventory):
                        info = plan_info[key]
                        W = merged[key].detach().float().cpu().numpy()
                        M, N = info["M"], info["N"]
                        # Budget rule (N2'' design): the plan fraction is a
                        # TOTAL parameter budget for the layer, SPLIT between
                        # the TT cores (tt_split) and the residual factors.
                        # (At plan 0.50 a TT rank that consumes the whole
                        # budget would leave rank 0 for the residual and
                        # degenerate to the failed uniform N2.)
                        budget = int(frac * info["dense_params"])
                        tt_frac = frac * ttsplit
                        r = _rank_for_fraction(tt_frac, info["m_dims"], info["n_dims"])
                        cs = k.tt_svd(W, info["m_dims"], info["n_dims"],
                                      tuple(r for _ in range(len(info["m_dims"]) - 1)))
                        tt_params = int(sum(int(np.prod(g.shape)) for g in cs.arrays))
                        cap = max(0, budget - tt_params - info["M"] - info["N"])
                        r_eff = int(min(rrank, cap // (info["M"] + info["N"])))
                        comp_l = svd_residual(cs.arrays, info["m_dims"],
                                              info["n_dims"], W,
                                              residual_rank=max(r_eff, 0))
                        # Wc written back into the live dict (eval restores
                        # FT weights afterwards — contamination guard).
                        Wc = _dense_from_comp(comp_l, info["m_dims"], info["n_dims"])
                        comp[key] = torch.from_numpy(Wc.astype(np.float16))
                        total_comp_params += compressed_params(comp_l)
                        total_dense_params += info["dense_params"]
                        rep = layer_report(comp_l, W, info["m_dims"], info["n_dims"])
                        n_sound += 1 if rep["sound"] else 0
                        if (i_layer + 1) % 24 == 0 or i_layer + 1 == len(inventory):
                            print(f"    layer {i_layer + 1}/{len(inventory)} "
                                  f"({layer_type}) r'={comp_l['residual_rank']} "
                                  f"L={rep['lipschitz_bound']:.3f} "
                                  f"recon={rep['recon_rel_err']:.4f}", flush=True)

                    ratio = total_dense_params / max(total_comp_params, 1)
                    acc = harness.eval(comp)
                    ev = getattr(harness, "_last_eval", {})
                    delta = (acc - n1_ft) if n1_ft is not None and acc == acc \
                        else float("nan")
                    dt = time.perf_counter() - t0
                    status = "completed" if acc == acc else "failed"
                    print(f"  [N2R seed={seed} frac={frac:.2f} r'={rrank}] DONE "
                          f"ratio={ratio:.2f}x acc={acc:.4f} delta={delta:+.4f} "
                          f"sound={n_sound}/{len(inventory)} "
                          f"eval={ev.get('scope', '?')} ({dt:.0f}s) <- live",
                          flush=True)
                    out.append(f"| {seed} | {frac:.3f} | {rrank} | {ratio:.2f}x | "
                               f"{acc:.4f} | {delta:+.4f} | {n_sound}/"
                               f"{len(inventory)} | {status} |")
                    _record_n2r_point(ctx, seed, frac, rrank, ratio, acc, delta,
                                      n_sound, len(inventory), dt, layers_scope,
                                      ft_path, harness.batch_is_matched, ev)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    print(f"  [N2R seed={seed} frac={frac:.2f} r'={rrank}] "
                          f"FAILED: {type(exc).__name__}: {exc}", flush=True)
                    out.append(f"| {seed} | {frac:.3f} | {rrank} | - | - | - | "
                               f"- | FAIL ({type(exc).__name__}: {exc}) |")

        # Free the per-seed VLA harness before the next seed (GPU memory).
        del harness
        torch.cuda.empty_cache()

    out.append("")
    out.append("* N2R points are measured against the N1 fine-tuned baseline "
               "at matched budget (same eval batch, same protocol). Ratios "
               "count TT cores + residual factors + scales (honest storage). "
               "R1 bar: some point at ratio >= 2x with delta >= -0.05.")


def _dense_from_comp(comp_l: dict, m_dims, n_dims) -> np.ndarray:
    """Materialize the compensated dense weight for the state-dict swap."""
    from qicert.compress_residual import to_dense
    return to_dense(comp_l, m_dims, n_dims)


def _record_n2r_point(ctx, seed, frac, rrank, ratio, acc, delta, n_sound,
                      n_layers, dt, layers_scope, ft_path, matched_batch,
                      eval_info: dict | None = None):
    """Record one N2R point (run.json + metrics + ledger row)."""
    ev = eval_info or {}
    rec = start_run(ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "N2R"),
                    label=f"residual-TT-{frac:.3f}-rr{rrank}-seed{seed}",
                    config={
                        "experiment": "N2R residual-compensated compression",
                        "backbone": "TT", "plan_fraction": frac,
                        "residual_rank_cap": rrank, "tt_split": ttsplit,
                        "layers_scope": layers_scope,
                        "seed": seed, "ft_ckpt": str(ft_path),
                        "kernel": "tt_svd + closed-form SVD residual (QuaSAR-style)",
                        "ordering": "bit-reversed (N2' winner)",
                        "eval": ev.get("scope", "N1 matched-budget protocol"),
                        "matched_batch": matched_batch,
                        "n_eval_tokens": ev.get("total"),
                        "n_eval_batches": ev.get("n_batches"),
                    })
    if rec is None:
        return
    ci = ev.get("ci")
    finish_run(rec, status="completed" if acc == acc else "failed",
               results={"seed": seed, "backbone": "TT", "plan_fraction": frac,
                        "residual_rank_cap": rrank,
                        "ratio": round(float(ratio), 3),
                        "eval_acc": float(acc),
                        "correct": ev.get("correct"), "total": ev.get("total"),
                        "ci_lo": ci[0] if ci else None,
                        "ci_hi": ci[1] if ci else None,
                        "eval_scope": ev.get("scope"),
                        "n_eval_batches": ev.get("n_batches"),
                        "delta_vs_n1_ft": None if delta != delta else round(float(delta), 4),
                        "cert_sound_layers": n_sound,
                        "n_layers_compressed": n_layers,
                        "matched_batch": bool(matched_batch),
                        "wall_sec": round(float(dt), 2),
                        "note": "residual arm; ratio includes residual storage"})
