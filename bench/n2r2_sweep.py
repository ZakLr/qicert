# ===========================================================================
# N2R-v2 — activation-weighted residual compensation + mixed allocation
# ===========================================================================
# Built 2026-09-12 on the honest full-split results:
#   calibrated INT8 @ 2.00x -> 0.4466 (delta -0.0001)  [the comparator]
#   TT+residual   @ 2.86x   -> 0.1529 (delta -0.2939)  [R1 bar NOT met]
# The v1 residual fit minimized RAW WEIGHT error (Frobenius).  Quantization
# literature (GPTQ-style activation-weighted objectives) says the correction
# budget must be spent
# where the model ACTUALLY computes: weight the fit by per-input-channel
# activation statistics collected on TRAIN episodes (never held-out data).
#
# Second lever (pre-registered design option): MIXED ALLOCATION —
# attention q/k/v/o projections stay FULL-precision (their per-layer
# reconstruction error was the dominant outlier driver in the layer
# spectra), everything else is compressed.  Honest accounting: kept layers
# count their dense params in the compressed total, so the reported ratio
# is never flattered.
#
# Usage (full-split, the honest protocol):
#   QICERT_N2_EVAL=full QICERT_N2R2_MODE=weighted \
#   python -m qicert.bench.all --module n2r2_sweep --rows=n2r2-go-no-go \
#       --out results --exp-id N2R2 --seed 0 --run-tag N2R2-weighted-seed0 \
#       --save-ckpt results/N1v2-ckpt --capture heavy
#
# Env: QICERT_N2R2_PLANS ("0.50"), QICERT_N2R2_RRANKS ("32,64"),
#      QICERT_N2R2_MODE ("weighted"|"plain"), QICERT_N2R2_MIXED ("1"=on),
#      QICERT_N2R2_KEEP ("q_proj,k_proj,v_proj,o_proj"),
#      QICERT_N2R2_STAT ("mean"|"absmax"), QICERT_CALIB (path to npz),
#      QICERT_N2_LAYERS (all|layer0), QICERT_N2_EVAL (batch|full),
#      QICERT_FT_CKPT (fine-tuned checkpoint dir).
"""Activation-weighted residual sweep (bench module)."""
from __future__ import annotations

# Not part of the no-weights smoke set (needs fine-tuned weights + checkpoint).
SMOKE = frozenset()

import os as _os
from pathlib import Path

import numpy as np

from ._base import finish_run, start_run, table_header, wants
from .compress import CKPT, DATA_ROOT
from .n2_sweep import (
    _factor_dims,
    _llm_linear_keys,
    _rank_for_fraction,
)

_DEFAULT_KEEP = "q_proj,k_proj,v_proj,o_proj"


def run(rows: str, out: list[str], ctx=None) -> None:
    if wants(rows, "n2r2-go-no-go", SMOKE):
        _run_n2r2(out, ctx)


def _plans_from_env() -> tuple[float, ...]:
    env = _os.environ.get("QICERT_N2R2_PLANS",
                          _os.environ.get("QICERT_N2R_PLANS", "0.50"))
    return tuple(float(x) for x in env.split(",") if x.strip())


def _rranks_from_env() -> tuple[int, ...]:
    env = _os.environ.get("QICERT_N2R2_RRANKS",
                          _os.environ.get("QICERT_N2R_RRANKS", "32,64"))
    return tuple(int(x) for x in env.split(",") if x.strip())


def _keep_types() -> set[str]:
    env = _os.environ.get("QICERT_N2R2_KEEP", _DEFAULT_KEEP)
    return {t.strip() for t in env.split(",") if t.strip()}


def _is_kept(layer_type: str, keep_types) -> bool:
    """Mixed-allocation match by projection suffix.

    Audit 2026-09-13: layer_type values from _llm_linear_keys are the full
    dashed names ("self_attn-q_proj", "mlp-gate_proj") while keep_types
    holds bare projection names ("q_proj", ...).  The old direct
    membership test was therefore ALWAYS False and every 'mixed' run
    silently kept zero layers, degenerating into a duplicate of the
    non-mixed arm.  Matching on the suffix after the last '-' restores
    the intended semantics for both bare and prefixed env values.
    """
    suffix = layer_type.rsplit("-", 1)[-1]
    return suffix in {k.rsplit("-", 1)[-1] for k in keep_types}


def _load_activation_weights(inventory, stat: str) -> dict[str, np.ndarray]:
    """Per-layer weight vector w_j from the calibration npz (train episodes).

    stat="mean" -> mean |activation| per channel (preferred);
    stat="absmax" -> per-channel absmax (GPTQ-style outlier scaling).
    Falls back gracefully if the stored npz predates mean_abs.
    """
    from qicert.calibrate import load_stats

    path = Path(_os.environ.get(
        "QICERT_CALIB",
        str(Path("results") / "N2R" / "calib_seed0.npz")))
    if not path.exists():
        raise RuntimeError(
            f"calibration file not found: {path} — collect it first with "
            "`python -m qicert.calibrate` (train-episodes-only, ~5 min)")
    calib = load_stats(path)
    weights: dict[str, np.ndarray] = {}
    n_mean = n_absmax = 0
    for _lt, key in inventory:
        mod = key[: -len(".weight")] if key.endswith(".weight") else key
        entry = calib.get(mod)
        if entry is None:
            cands = [v for kk, v in calib.items()
                     if kk.endswith(mod) or mod.endswith(kk)]
            entry = cands[0] if len(cands) == 1 else None
        if entry is None:
            continue
        if stat == "mean" and "mean_abs" in entry:
            weights[key] = np.asarray(entry["mean_abs"], dtype=np.float64)
            n_mean += 1
        elif "absmax" in entry:
            weights[key] = np.asarray(entry["absmax"], dtype=np.float64)
            n_absmax += 1
    if not weights:
        raise RuntimeError(
            "no calibration entries matched any inventoried layer — "
            "the calib npz was probably collected from a different model")
    used = n_mean + n_absmax
    print(f"[N2R2] activation weights for {used}/{len(inventory)} layers "
          f"(stat={stat}: mean={n_mean}, absmax-fallback={n_absmax})",
          flush=True)
    return weights


def _run_n2r2(out: list[str], ctx=None) -> None:
    import time

    import torch

    from qicert.compress_residual import (
        compressed_params,
        layer_report,
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
    mode = _os.environ.get("QICERT_N2R2_MODE", "weighted")
    mixed = _os.environ.get("QICERT_N2R2_MIXED", "0") == "1"
    keep_types = _keep_types()
    stat = _os.environ.get("QICERT_N2R2_STAT", "mean")
    ttsplit = float(_os.environ.get("QICERT_N2R_TTSPLIT", "0.6"))
    bb = "TT"
    d = 2
    eval_mode = _os.environ.get("QICERT_N2_EVAL", "batch")
    if mode not in ("weighted", "plain"):
        raise RuntimeError(f"QICERT_N2R2_MODE must be weighted|plain, got {mode}")

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
        plan_info[key] = {"layer_type": layer_type, "M": M, "N": N,
                          "dense_params": M * N,
                          "m_dims": _factor_dims(M, d),
                          "n_dims": _factor_dims(N, d)}
    del base_sd, llm_sd
    import gc as _gc
    _gc.collect()

    act_w: dict[str, np.ndarray] = {}
    if mode == "weighted":
        act_w = _load_activation_weights(inventory, stat)

    from .n2_sweep import _N2EvalHarness

    out += table_header(
        f"N2R-v2 ({mode}{'+mixed' if mixed else ''}, stat={stat}, "
        f"eval={eval_mode}, {len(inventory)} layers, plans={plans}, "
        f"rranks={rranks}, tt_split={ttsplit}, seeds={seeds})",
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
        harness.adopt_reference(merged)
        n1_ft = harness.n1_ft

        for frac in plans:
            for rrank in rranks:
                t0 = time.perf_counter()
                try:
                    tag = f"{mode}{'+mixed' if mixed else ''}"
                    print(f"  [N2R2 seed={seed} {tag} frac={frac:.2f} "
                          f"r'={rrank}] compress+repair {len(inventory)} "
                          f"layers...", flush=True)
                    comp = dict(merged)
                    total_comp_params = 0
                    total_dense_params = 0
                    n_sound = 0
                    n_compressed = 0
                    n_kept = 0
                    for i_layer, (layer_type, key) in enumerate(inventory):
                        info = plan_info[key]
                        W = merged[key].detach().float().cpu().numpy()
                        M, N = info["M"], info["N"]
                        if mixed and _is_kept(layer_type, keep_types):
                            # Mixed allocation: keep FULL-precision. Honest
                            # storage: the dense params count as compressed.
                            comp[key] = merged[key].clone()
                            total_comp_params += info["dense_params"]
                            total_dense_params += info["dense_params"]
                            n_kept += 1
                            if (i_layer + 1) % 24 == 0 or i_layer + 1 == len(inventory):
                                print(f"    layer {i_layer + 1}/{len(inventory)} "
                                      f"({layer_type}) KEPT full", flush=True)
                            continue
                        budget = int(frac * info["dense_params"])
                        tt_frac = frac * ttsplit
                        r = _rank_for_fraction(tt_frac, info["m_dims"],
                                               info["n_dims"])
                        cs = k.tt_svd(W, info["m_dims"], info["n_dims"],
                                      tuple(r for _ in range(len(info["m_dims"]) - 1)))
                        tt_params = int(sum(int(np.prod(g.shape)) for g in cs.arrays))
                        cap = max(0, budget - tt_params - info["M"] - info["N"])
                        r_eff = int(min(rrank, cap // (info["M"] + info["N"])))
                        w = act_w.get(key) if mode == "weighted" else None
                        comp_l = svd_residual(
                            cs.arrays, info["m_dims"], info["n_dims"], W,
                            residual_rank=max(r_eff, 0),
                            activation_weight=w,
                            scale=(stat if w is not None else "none"))
                        from .n2r_sweep import _dense_from_comp
                        Wc = _dense_from_comp(comp_l, info["m_dims"], info["n_dims"])
                        comp[key] = torch.from_numpy(Wc.astype(np.float16))
                        total_comp_params += compressed_params(comp_l)
                        total_dense_params += info["dense_params"]
                        n_compressed += 1
                        rep = layer_report(comp_l, W, info["m_dims"], info["n_dims"])
                        n_sound += 1 if rep["sound"] else 0
                        if (i_layer + 1) % 24 == 0 or i_layer + 1 == len(inventory):
                            print(f"    layer {i_layer + 1}/{len(inventory)} "
                                  f"({layer_type}) r'={comp_l['residual_rank']} "
                                  f"L={rep['lipschitz_bound']:.3f} "
                                  f"recon={rep['recon_rel_err']:.4f}"
                                  f"{' [w]' if w is not None else ''}", flush=True)

                    ratio = total_dense_params / max(total_comp_params, 1)
                    acc = harness.eval(comp)
                    ev = getattr(harness, "_last_eval", {})
                    delta = (acc - n1_ft) if n1_ft is not None and acc == acc \
                        else float("nan")
                    dt = time.perf_counter() - t0
                    status = "completed" if acc == acc else "failed"
                    print(f"  [N2R2 seed={seed} {tag} frac={frac:.2f} "
                          f"r'={rrank}] DONE ratio={ratio:.2f}x acc={acc:.4f} "
                          f"delta={delta:+.4f} sound={n_sound}/{n_compressed} "
                          f"kept={n_kept} eval={ev.get('scope', '?')} "
                          f"({dt:.0f}s) <- live", flush=True)
                    out.append(f"| {seed} | {frac:.3f} | {rrank} | {ratio:.2f}x | "
                               f"{acc:.4f} | {delta:+.4f} | {n_sound}/"
                               f"{n_compressed} | {status} |")
                    _record_n2r2_point(ctx, seed, frac, rrank, mode, mixed,
                                       stat, ttsplit, ratio, acc, delta, n_sound,
                                       n_compressed, n_kept, dt, layers_scope,
                                       eval_mode, ft_path,
                                       harness.batch_is_matched, ev)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    print(f"  [N2R2 seed={seed} frac={frac:.2f} r'={rrank}] "
                          f"FAILED: {type(exc).__name__}: {exc}", flush=True)
                    out.append(f"| {seed} | {frac:.3f} | {rrank} | - | - | - | "
                               f"- | FAIL ({type(exc).__name__}: {exc}) |")

        del harness
        torch.cuda.empty_cache()

    out.append("")
    out.append("* N2R-v2 points are measured against the N1 fine-tuned "
               "baseline on the SAME harness (full-split protocol when "
               "QICERT_N2_EVAL=full — the validated, N1-identical one). "
               "Weighted arm: residual fit minimizes activation-weighted "
               "error (GPTQ-style, train-episode stats). Mixed arm: kept "
               "layers count their dense params (honest ratio). "
               "R1 bar: some point at ratio >= 2x with delta >= -0.05.")


def _record_n2r2_point(ctx, seed, frac, rrank, mode, mixed, stat, ttsplit, ratio, acc,
                       delta, n_sound, n_compressed, n_kept, dt, layers_scope,
                       eval_mode, ft_path, matched_batch, eval_info=None):
    """Record one N2R-v2 point (run.json + metrics + ledger row)."""
    ev = eval_info or {}
    rec = start_run(ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "N2R2"),
                    label=f"n2r2-{mode}{'-mixed' if mixed else ''}"
                          f"-{frac:.3f}-rr{rrank}-seed{seed}",
                    config={
                        "experiment": "N2R-v2 activation-weighted residual",
                        "backbone": "TT", "plan_fraction": frac,
                        "residual_rank_cap": rrank, "tt_split": ttsplit,
                        "fit_mode": mode, "mixed_allocation": mixed,
                        "calib_stat": stat if mode == "weighted" else None,
                        "layers_scope": layers_scope,
                        "eval_mode": eval_mode,
                        "seed": seed, "ft_ckpt": str(ft_path),
                        "kernel": "tt_svd + activation-weighted SVD residual",
                        "eval": ev.get("scope", "harness protocol"),
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
                        "fit_mode": mode, "mixed_allocation": mixed,
                        "calib_stat": stat if mode == "weighted" else None,
                        "ratio": round(float(ratio), 3),
                        "eval_acc": float(acc),
                        "correct": ev.get("correct"), "total": ev.get("total"),
                        "ci_lo": ci[0] if ci else None,
                        "ci_hi": ci[1] if ci else None,
                        "eval_scope": ev.get("scope"),
                        "n_eval_batches": ev.get("n_batches"),
                        "delta_vs_n1_ft": None if delta != delta else round(float(delta), 4),
                        "cert_sound_layers": n_sound,
                        "n_layers_compressed": n_compressed,
                        "n_layers_kept_full": n_kept,
                        "matched_batch": bool(matched_batch),
                        "wall_sec": round(float(dt), 2),
                        })
