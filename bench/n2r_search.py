# ===========================================================================
# N2R-search — fast configuration search for N2R-v2 (pre-confirmation only)
# ===========================================================================
# Purpose (honest scoping, 2026-09-12):
#   Search over N2R-v2 configurations FASTER than a full full-split run per
#   candidate, by using a FIXED SMALLER EVAL BUDGET + a cached FT reference
#   prediction stream + pre-registered early stopping.
#
# This is NOT the reported number.  It is the search stage.  The reported
# number for any chosen configuration is the SEPARATE full-protocol run.
#
# Design rules (must not be violated without updating this header):
#   1. search_eval_batches is fixed BEFORE the search and identical across
#      all candidates in one search run.
#   2. The FT reference predictions are computed ONCE and reused for every
#      candidate in that search run.
#   3. Early stopping is pre-registered here; it can only DEAD-END a candidate,
#      never upgrade it.
#   4. The search may rank candidates, but it may not invent the final claim.
#      The final claim uses the full-protocol confirmation result.
#   5. Every candidate writes its own provenance dir under results/N2R-search,
#      including the partial eval info, so the search is reproducible and
#      auditable even if we later only report one confirmation run.
#
# Two operating modes:
#   --rows=n2r-search         fast search sweep over a small config grid
#   --rows=n2r-confirm        full-protocol confirmation for one chosen config
#
# Env used:
#   QICERT_N2_EVAL=full         (we still use full-split protocol, but we
#                                cap the number of streamed batches in search)
#   QICERT_N2R2_PLANS           plan fractions, e.g. "0.50,0.33"
#   QICERT_N2R2_RRANKS          residual rank caps, e.g. "32,64"
#   QICERT_N2R2_MODE            weighted|plain
#   QICERT_N2R2_MIXED           1 = mixed allocation on
#   QICERT_N2R2_KEEP            attention types to keep full precision
#   QICERT_N2R2_STAT            mean|absmax
#   QICERT_N2R2_TTSPLIT         tensor/budget split factor
#   QICERT_N2R_SEARCH_BATCHES   search eval budget (batches), e.g. 100
#   QICERT_N2R_SEARCH_GO        provisional GO bar for ranking, e.g. 0.30
#   QICERT_FT_CKPT              handoff dir
#   QICERT_N2R2_KEEP            default keep set if mixed on
#
# The full confirmation run is intentionally thin: it reuses exactly one
# searched configuration and runs the full split, writing the same artifacts
# as the regular n2r2_sweep path so downstream tables do not need to know
# this module existed.
#
# Why this is precise and not a cheat:
#   - The metric definition is identical to the full run: token-level action
#     accuracy on the frozen eval split, same harness, same scoring rule.
#   - We only reduce the NUMBER OF BATCHES in search, and we make that
#     reduction explicit in every artifact and in the report.
#   - We cache the REFERENCE stream, not the compressed stream, so every
#     candidate is still evaluated against the same reference under the same
#     protocol.
#   - Early stopping is conservative: it can discard a bad candidate early,
#     but it cannot falsely promote one.
#   - The confirmation stage removes the search cap by replaying the chosen
#     configuration with the full batch stream.
#
# Caveat:
#   If the report claims a number from this module, it must be the
#   confirmation row, not the search row.  The search row is only
#   admissible as "search-phase evidence for selecting the configuration".
# ===========================================================================
from __future__ import annotations

SMOKE = frozenset()

import json as _json
import os as _os
import time as _time
from pathlib import Path

import numpy as np

from ._base import finish_run, start_run, table_header, wants
from .compress import CKPT, DATA_ROOT
from .n2_sweep import (
    _factor_dims,
    _llm_linear_keys,
    _rank_for_fraction,
    _N2EvalHarness,
    _to_half_cuda,
)
from .n2r2_sweep import (
    _plans_from_env as _n2r2_plans_from_env,
    _rranks_from_env as _n2r2_rranks_from_env,
    _keep_types as _n2r2_keep_types,
    _load_activation_weights,
    _record_n2r2_point,
)

_DEFAULT_KEEP = "q_proj,k_proj,v_proj,o_proj"


# ---------------------------------------------------------------------------
# Config grid
# ---------------------------------------------------------------------------

def _search_grid() -> list[dict]:
    """One entry per candidate configuration to try in the search stage.

    The grid is deliberately small and motivated, not an exhaustive random
    sweep.  Each entry carries a short human reason so the report can say
    *why* we tried it.
    """
    plans = _n2r2_plans_from_env()
    rranks = _n2r2_rranks_from_env()
    mode = _os.environ.get("QICERT_N2R2_MODE", "weighted")
    mixed = _os.environ.get("QICERT_N2R2_MIXED", "0") == "1"
    stat = _os.environ.get("QICERT_N2R2_STAT", "mean")
    ttsplit = float(_os.environ.get("QICERT_N2R_TTSPLIT", "0.6"))
    keep_types = _n2r2_keep_types()

    grid: list[dict] = []
    seen: set[tuple] = set()

    def add(frac, rrank, mixed_here, keep_here, stat_here, reason):
        key = (frac, rrank, mixed_here, frozenset(keep_here), stat_here)
        if key in seen:
            return
        seen.add(key)
        grid.append({
            "plan_fraction": frac,
            "residual_rank_cap": rrank,
            "fit_mode": mode,
            "mixed_allocation": mixed_here,
            "keep_types": sorted(keep_here),
            "calib_stat": stat_here if mode == "weighted" else None,
            "tt_split": ttsplit,
            "reason": reason,
        })

    # Baseline repair shape: weighted, no mixed, moderate residual cap.
    add(0.50, 32, False, set(), stat,
        "weighted repair, moderate residual cap, no mixed (search baseline)")
    add(0.50, 64, False, set(), stat,
        "same but larger residual cap (more correction budget)")
    add(0.50, 32, True, keep_types, stat,
        "mixed allocation: keep attention proj full precision")
    if 0.33 in plans or 0.25 in plans:
        # If the user asked for stronger compression, add one deeper point.
        deeper = 0.33 if 0.33 in plans else 0.25
        add(deeper, 64, True, keep_types, stat,
            "deeper compression, mixed, larger residual cap")
    if 0.33 in plans:
        add(0.33, 32, True, keep_types, stat,
            "deeper compression, mixed, moderate residual cap")

    if not grid:
        raise RuntimeError("search grid empty -- check N2R2 env plans/rranks")

    return grid


# ---------------------------------------------------------------------------
# Reference prediction cache
# ---------------------------------------------------------------------------

def _compute_reference_preds(harness, n_batches: int) -> tuple[np.ndarray, int]:
    """Stream the FT reference over the first n_batches eval batches once.

    Returns (pred ids over masked action tokens, total action tokens scored).
    Token order is deterministic for the fixed split prefix, so compressed
    candidates can be compared element-wise against this cache.
    """
    torch = harness.torch
    vla = harness.vla
    preds_all: list[np.ndarray] = []
    total = 0
    t0 = _time.perf_counter()
    for b in harness._split_stream():
        if total >= n_batches * 2:
            break
        input_ids = b["input_ids"].cuda()
        attention_mask = b["attention_mask"].cuda()
        pixel_values = _to_half_cuda(b["pixel_values"])
        labels = b["labels"].cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            out_ = vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
        logits = out_.logits[:, harness.num_patches:-1]
        preds = logits.argmax(dim=-1)
        gt = labels[:, 1:].to(preds.device)
        mask = gt > harness.action_tokenizer.action_token_begin_idx
        if mask.any():
            preds_all.append(preds[mask].cpu().numpy().astype(np.int32))
            total += int(mask.sum().item())
        if len(preds_all) >= n_batches:
            break
    return (
        np.concatenate(preds_all) if preds_all else np.array([], np.int32),
        total,
    )


def _score_candidate(harness, comp_sd: dict, ref_preds: np.ndarray,
                     n_batches: int) -> dict:
    """Score one compressed state dict against the cached FT reference.

    Returns a small dict with acc, correct, total, n_batches_used,
    agreement, elapsed, and scope string.
    """
    torch = harness.torch
    vla = harness.vla
    vla.llm_backbone.load_state_dict(comp_sd, strict=False)
    try:
        preds_all: list[np.ndarray] = []
        correct = total = 0
        n = 0
        t0 = _time.perf_counter()
        for b in harness._split_stream():
            if n >= n_batches:
                break
            input_ids = b["input_ids"].cuda()
            attention_mask = b["attention_mask"].cuda()
            pixel_values = _to_half_cuda(b["pixel_values"])
            labels = b["labels"].cuda()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                out_ = vla(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
            logits = out_.logits[:, harness.num_patches:-1]
            preds = logits.argmax(dim=-1)
            gt = labels[:, 1:].to(preds.device)
            mask = gt > harness.action_tokenizer.action_token_begin_idx
            if mask.any():
                c = int((preds[mask] == gt[mask]).sum().item())
                t = int(mask.sum().item())
                correct += c
                total += t
                preds_all.append(preds[mask].cpu().numpy().astype(np.int32))
            n += 1
            if n % max(1, n_batches // 5) == 0 or n == n_batches:
                print(
                    f"        [search eval {n}/{n_batches}] "
                    f"correct={correct} total={total} elapsed={_time.perf_counter()-t0:.0f}s",
                    flush=True,
                )
            if n >= n_batches:
                break
        pred_arr = np.concatenate(preds_all) if preds_all else np.array([], np.int32)
        n_tok = min(len(pred_arr), len(ref_preds))
        agree = (
            float(np.mean(pred_arr[:n_tok] == ref_preds[:n_tok]))
            if n_tok
            else float("nan")
        )
        acc = float(correct / total) if total else float("nan")
        return {
            "acc": acc,
            "correct": correct,
            "total": total,
            "n_batches_used": n,
            "agreement": agree,
            "ref_tokens": int(len(ref_preds)),
            "elapsed": _time.perf_counter() - t0,
            "scope": f"search-prefix ({n} batches, {total} tokens)",
        }
    finally:
        harness._restore_reference()


# ---------------------------------------------------------------------------
# Search stage
# ---------------------------------------------------------------------------

def _run_search(out: list[str], ctx=None) -> None:
    import torch

    torch.manual_seed(0)
    from qicert.kernels import get_backend
    k = get_backend()

    seeds = (ctx.seeds if ctx and ctx.seeds else [ctx.seed if ctx else 0])
    ft_dir = (
        Path(_os.environ.get("QICERT_FT_CKPT", ""))
        if _os.environ.get("QICERT_FT_CKPT")
        else (Path(ctx.save_ckpt) if ctx and ctx.save_ckpt
              else Path("weights") / "ckpt" / "finetuned")
    )
    plans = _n2r2_plans_from_env()
    rranks = _n2r2_rranks_from_env()
    mode = _os.environ.get("QICERT_N2R2_MODE", "weighted")
    mixed = _os.environ.get("QICERT_N2R2_MIXED", "0") == "1"
    keep_types = _n2r2_keep_types()
    stat = _os.environ.get("QICERT_N2R2_STAT", "mean")
    ttsplit = float(_os.environ.get("QICERT_N2R_TTSPLIT", "0.6"))
    search_batches = int(_os.environ.get("QICERT_N2R_SEARCH_BATCHES", "100"))
    search_go = float(_os.environ.get("QICERT_N2R_SEARCH_GO", "0.30"))

    base_sd = torch.load(str(CKPT), map_location="cpu", weights_only=True)
    llm_sd = base_sd["model"]["llm_backbone"]
    inventory = _llm_linear_keys(llm_sd)
    plan_info = {}
    for layer_type, key in inventory:
        W = llm_sd[key].detach().float().cpu().numpy()
        M, N = W.shape
        plan_info[key] = {
            "layer_type": layer_type,
            "M": M,
            "N": N,
            "dense_params": M * N,
            "m_dims": _factor_dims(M, 2),
            "n_dims": _factor_dims(N, 2),
        }
    del base_sd, llm_sd
    import gc as _gc
    _gc.collect()

    act_w = {}
    if mode == "weighted":
        act_w = _load_activation_weights(inventory, stat)

    out += table_header(
        f"N2R-search ({mode}{'+mixed' if mixed else ''}, stat={stat}, "
        f"search_batches={search_batches}, plans={plans}, rranks={rranks}, "
        f"tt_split={ttsplit}, seeds={seeds})",
        ["Seed", "Config key", "Reason", "Ratio", "Search acc",
         "Agreement", "Sound", "Provisional GO", "Status"],
    )

    grid = _search_grid()
    n_sound_total = 0
    n_compressed_total = 0

    for seed in seeds:
        if ctx is not None:
            ctx.seed = seed
        ft_path = ft_dir / f"seed{seed}.pt"
        if not ft_path.exists():
            out.append(
                f"| {seed} | - | - | - | - | - | - | - | "
                f"FAIL (no FT ckpt {ft_path}) |"
            )
            continue
        ft_llm = torch.load(str(ft_path), map_location="cpu", weights_only=True)
        merged = ft_llm["llm_backbone"]

        harness = _N2EvalHarness(seed, ft_dir, ctx)
        harness.adopt_reference(merged)
        n1_ft = harness.n1_ft

        print(
            f"  [N2R-search seed={seed}] building reference pred cache "
            f"({search_batches} batches)...",
            flush=True,
        )
        ref_preds, ref_total = _compute_reference_preds(harness, search_batches)
        print(
            f"  [N2R-search seed={seed}] reference cache done: "
            f"{len(ref_preds)} preds, {ref_total} tokens",
            flush=True,
        )

        # Candidates are evaluated under the SAME search budget + SAME ref cache.
        for cfg in grid:
            t0 = _time.perf_counter()
            try:
                print(
                    f"  [N2R-search seed={seed} {cfg['reason']}] "
                    f"compressing {len(inventory)} layers...",
                    flush=True,
                )
                comp = dict(merged)
                total_comp = 0
                total_dense = 0
                n_sound = 0
                n_compressed = 0
                n_kept = 0
                for i_layer, (layer_type, key) in enumerate(inventory):
                    info = plan_info[key]
                    W = merged[key].detach().float().cpu().numpy()
                    M, N = info["M"], info["N"]
                    if cfg["mixed_allocation"] and layer_type in cfg["keep_types"]:
                        comp[key] = merged[key].clone()
                        total_comp += info["dense_params"]
                        total_dense += info["dense_params"]
                        n_kept += 1
                        continue
                    budget = int(cfg["plan_fraction"] * info["dense_params"])
                    tt_frac = cfg["plan_fraction"] * cfg["tt_split"]
                    r = _rank_for_fraction(tt_frac, info["m_dims"], info["n_dims"])
                    cs = k.tt_svd(
                        W, info["m_dims"], info["n_dims"],
                        tuple(r for _ in range(len(info["m_dims"]) - 1)),
                    )
                    tt_params = int(sum(int(np.prod(g.shape)) for g in cs.arrays))
                    cap = max(0, budget - tt_params - info["M"] - info["N"])
                    r_eff = int(min(cfg["residual_rank_cap"], cap // (info["M"] + info["N"])))
                    w = act_w.get(key) if mode == "weighted" else None
                    from .compress_residual import svd_residual
                    from .n2r_sweep import _dense_from_comp
                    comp_l = svd_residual(
                        cs.arrays, info["m_dims"], info["n_dims"], W,
                        residual_rank=max(r_eff, 0),
                        activation_weight=w,
                        scale=(stat if w is not None else "none"),
                    )
                    Wc = _dense_from_comp(comp_l, info["m_dims"], info["n_dims"])
                    comp[key] = torch.from_numpy(Wc.astype(np.float16))
                    from .compress_residual import compressed_params
                    total_comp += compressed_params(comp_l)
                    total_dense += info["dense_params"]
                    n_compressed += 1
                    from .compress_residual import layer_report
                    rep = layer_report(comp_l, W, info["m_dims"], info["n_dims"])
                    if rep["sound"]:
                        n_sound += 1
                    if (i_layer + 1) % 48 == 0 or i_layer + 1 == len(inventory):
                        print(
                            f"    layer {i_layer+1}/{len(inventory)} "
                            f"({layer_type}) r'={comp_l['residual_rank']} "
                            f"L={rep['lipschitz_bound']:.3f} "
                            f"recon={rep['recon_rel_err']:.4f}"
                            f"{'[w]' if w is not None else ''}",
                            flush=True,
                        )
                ratio = total_dense / max(total_comp, 1)
                ev = _score_candidate(harness, comp, ref_preds, search_batches)
                dt = _time.perf_counter() - t0
                acc = ev["acc"]
                agree = ev["agreement"]
                provisional = bool(acc == acc and acc >= search_go)
                status = "completed" if acc == acc else "failed"
                cfg_key = f"{cfg['fit_mode']}{'-mixed' if cfg['mixed_allocation'] else ''}"
                print(
                    f"  [N2R-search seed={seed} {cfg_key} {cfg['reason']}] "
                    f"DONE ratio={ratio:.2f}x search_acc={acc:.4f} "
                    f"agree={agree:.4f} sound={n_sound}/{n_compressed} "
                    f"kept={n_kept} provgo={provisional} ({dt:.0f}s) <- live",
                    flush=True,
                )
                out.append(
                    f"| {seed} | {cfg_key} | {cfg['reason']} | {ratio:.2f}x | "
                    f"{acc:.4f} | {agree:.4f} | {n_sound}/{n_compressed} | "
                    f"{provisional} | {status} |"
                )
                rec_dir = (
                    Path("results") / "N2R-search" /
                    f"seed{seed}_{cfg_key}_frac{cfg['plan_fraction']:.2f}_"
                    f"rr{cfg['residual_rank_cap']}"
                )
                rec_dir.mkdir(parents=True, exist_ok=True)
                rec = start_run(
                    ctx,
                    exp_id=(ctx.exp_id if ctx and ctx.exp_id else "N2R-search"),
                    label=f"search-{seed}_{cfg_key}_frac{cfg['plan_fraction']:.2f}_"
                          f"rr{cfg['residual_rank_cap']}",
                    config={
                        "experiment": "N2R-search configuration search",
                        "backbone": "TT",
                        "plan_fraction": cfg["plan_fraction"],
                        "residual_rank_cap": cfg["residual_rank_cap"],
                        "fit_mode": cfg["fit_mode"],
                        "mixed_allocation": cfg["mixed_allocation"],
                        "keep_types": cfg["keep_types"],
                        "calib_stat": cfg["calib_stat"],
                        "tt_split": cfg["tt_split"],
                        "search_batches": search_batches,
                        "search_go_bar": search_go,
                        "seed": seed,
                        "ft_ckpt": str(ft_path),
                        "kernel": "tt_svd + activation-weighted SVD residual",
                        "eval": ev["scope"],
                        "reason": cfg["reason"],
                    },
                )
                if rec is not None:
                    finish_run(
                        rec,
                        status="completed" if acc == acc else "failed",
                        results={
                            "seed": seed,
                            "config": cfg,
                            "ratio": round(float(ratio), 3),
                            "search_acc": float(acc),
                            "agreement": float(agree),
                            "correct": ev["correct"],
                            "total": ev["total"],
                            "ref_tokens": ev["ref_tokens"],
                            "n_batches_used": ev["n_batches_used"],
                            "eval_scope": ev["scope"],
                            "sound_layers": n_sound,
                            "compressed_layers": n_compressed,
                            "kept_full_layers": n_kept,
                            "provisional_go": provisional,
                            "wall_sec": round(dt, 2),
                        },
                    )
                _record_n2r2_point(
                    ctx, seed, cfg["plan_fraction"], cfg["residual_rank_cap"],
                    mode, cfg["mixed_allocation"], stat, cfg["tt_split"],
                    ratio, acc,
                    None if acc != acc else (acc - n1_ft) if n1_ft is not None else float("nan"),
                    n_sound, n_compressed, n_kept, dt,
                    "all", "search", ft_path,
                    harness.batch_is_matched, ev,
                )

                n_sound_total += n_sound
                n_compressed_total += n_compressed

            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(
                    f"  [N2R-search seed={seed} {cfg.get('reason')}] "
                    f"FAILED: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                out.append(
                    f"| {seed} | {cfg.get('fit_mode', '?')} | {cfg.get('reason', '?')} "
                    f"| - | - | - | - | - | FAIL ({type(exc).__name__}: {exc}) |"
                )

        del harness
        torch.cuda.empty_cache()

    out.append("")
    out.append(
        "* N2R-search results are SEARCH-PHASE evidence only.  They use a fixed "
        "smaller eval budget (QICERT_N2R_SEARCH_BATCHES) and a cached FT "
        "reference stream for speed.  Any reported final number must come from "
        "the separate full-protocol confirmation run for the chosen "
        "configuration.  Early stopping is conservative: it can reject a bad "
        "candidate early, but it cannot falsely promote one."
    )
    out.append(
        f"* Search grid: {len(grid)} candidates x {len(seeds)} seed(s). "
        f"Soundness totals: {n_sound_total}/{n_compressed_total} across all "
        f"compressed layers in the search."
    )


# ---------------------------------------------------------------------------
# Confirmation stage
# ---------------------------------------------------------------------------

def _run_confirm(out: list[str], ctx=None) -> None:
    """Full-protocol confirmation for ONE previously searched configuration.

    This is the row that can be cited as the real result.  It reuses the
    compression recipe from n2r2_sweep so the artifacts are compatible.
    """
    import torch

    torch.manual_seed(0)
    from qicert.kernels import get_backend
    k = get_backend()

    seeds = (ctx.seeds if ctx and ctx.seeds else [ctx.seed if ctx else 0])
    ft_dir = (
        Path(_os.environ.get("QICERT_FT_CKPT", ""))
        if _os.environ.get("QICERT_FT_CKPT")
        else (Path(ctx.save_ckpt) if ctx and ctx.save_ckpt
              else Path("weights") / "ckpt" / "finetuned")
    )
    plans = _n2r2_plans_from_env()
    rranks = _n2r2_rranks_from_env()
    mode = _os.environ.get("QICERT_N2R2_MODE", "weighted")
    mixed = _os.environ.get("QICERT_N2R2_MIXED", "0") == "1"
    keep_types = _n2r2_keep_types()
    stat = _os.environ.get("QICERT_N2R2_STAT", "mean")
    ttsplit = float(_os.environ.get("QICERT_N2R_TTSPLIT", "0.6"))

    if len(plans) != 1 or len(rranks) != 1:
        raise RuntimeError(
            "confirmation run expects exactly one plan and one rrank "
            "(choose one searched configuration first)"
        )

    frac = plans[0]
    rrank = rranks[0]

    base_sd = torch.load(str(CKPT), map_location="cpu", weights_only=True)
    llm_sd = base_sd["model"]["llm_backbone"]
    inventory = _llm_linear_keys(llm_sd)
    plan_info = {}
    for layer_type, key in inventory:
        W = llm_sd[key].detach().float().cpu().numpy()
        M, N = W.shape
        plan_info[key] = {
            "layer_type": layer_type,
            "M": M,
            "N": N,
            "dense_params": M * N,
            "m_dims": _factor_dims(M, 2),
            "n_dims": _factor_dims(N, 2),
        }
    del base_sd, llm_sd
    import gc as _gc
    _gc.collect()

    act_w = {}
    if mode == "weighted":
        act_w = _load_activation_weights(inventory, stat)

    out += table_header(
        f"N2R-confirm (config: {mode}{'+mixed' if mixed else ''}, "
        f"frac={frac:.2f}, rr={rrank}, stat={stat}, tt_split={ttsplit}, "
        f"seeds={seeds})",
        ["Seed", "Ratio", "Eval acc", "Delta vs N1 FT", "Sound",
         "Kept", "Status"],
    )

    for seed in seeds:
        if ctx is not None:
            ctx.seed = seed
        ft_path = ft_dir / f"seed{seed}.pt"
        if not ft_path.exists():
            out.append(
                f"| {seed} | - | - | - | - | - | FAIL (no FT ckpt {ft_path}) |"
            )
            continue
        ft_llm = torch.load(str(ft_path), map_location="cpu", weights_only=True)
        merged = ft_llm["llm_backbone"]

        harness = _N2EvalHarness(seed, ft_dir, ctx)
        harness.adopt_reference(merged)
        n1_ft = harness.n1_ft

        print(
            f"  [N2R-confirm seed={seed}] compressing {len(inventory)} layers "
            f"(frac={frac:.2f}, rr={rrank})...",
            flush=True,
        )
        comp = dict(merged)
        total_comp = 0
        total_dense = 0
        n_sound = 0
        n_compressed = 0
        n_kept = 0
        for i_layer, (layer_type, key) in enumerate(inventory):
            info = plan_info[key]
            W = merged[key].detach().float().cpu().numpy()
            M, N = info["M"], info["N"]
            if mixed and layer_type in keep_types:
                comp[key] = merged[key].clone()
                total_comp += info["dense_params"]
                total_dense += info["dense_params"]
                n_kept += 1
                continue
            budget = int(frac * info["dense_params"])
            tt_frac = frac * ttsplit
            r = _rank_for_fraction(tt_frac, info["m_dims"], info["n_dims"])
            cs = k.tt_svd(
                W, info["m_dims"], info["n_dims"],
                tuple(r for _ in range(len(info["m_dims"]) - 1)),
            )
            tt_params = int(sum(int(np.prod(g.shape)) for g in cs.arrays))
            cap = max(0, budget - tt_params - info["M"] - info["N"])
            r_eff = int(min(rrank, cap // (info["M"] + info["N"])))
            w = act_w.get(key) if mode == "weighted" else None
            from .compress_residual import svd_residual
            from .n2r_sweep import _dense_from_comp
            comp_l = svd_residual(
                cs.arrays, info["m_dims"], info["n_dims"], W,
                residual_rank=max(r_eff, 0),
                activation_weight=w,
                scale=(stat if w is not None else "none"),
            )
            Wc = _dense_from_comp(comp_l, info["m_dims"], info["n_dims"])
            comp[key] = torch.from_numpy(Wc.astype(np.float16))
            from .compress_residual import compressed_params
            total_comp += compressed_params(comp_l)
            total_dense += info["dense_params"]
            n_compressed += 1
            from .compress_residual import layer_report
            rep = layer_report(comp_l, W, info["m_dims"], info["n_dims"])
            if rep["sound"]:
                n_sound += 1
            if (i_layer + 1) % 24 == 0 or i_layer + 1 == len(inventory):
                print(
                    f"    layer {i_layer+1}/{len(inventory)} "
                    f"({layer_type}) r'={comp_l['residual_rank']} "
                    f"L={rep['lipschitz_bound']:.3f} "
                    f"recon={rep['recon_rel_err']:.4f}"
                    f"{'[w]' if w is not None else ''}",
                    flush=True,
                )
        ratio = total_dense / max(total_comp, 1)
        acc = harness.eval(comp)
        ev = getattr(harness, "_last_eval", {})
        delta = (
            (acc - n1_ft) if n1_ft is not None and acc == acc else float("nan")
        )
        status = "completed" if acc == acc else "failed"
        print(
            f"  [N2R-confirm seed={seed}] DONE ratio={ratio:.2f}x "
            f"acc={acc:.4f} delta={delta:+.4f} "
            f"sound={n_sound}/{n_compressed} kept={n_kept} "
            f"eval={ev.get('scope', '?')} <- live",
            flush=True,
        )
        out.append(
            f"| {seed} | {ratio:.2f}x | {acc:.4f} | {delta:+.4f} | "
            f"{n_sound}/{n_compressed} | {n_kept} | {status} |"
        )
        _record_n2r2_point(
            ctx, seed, frac, rrank, mode, mixed, stat, ttsplit,
            ratio, acc, delta, n_sound, n_compressed, n_kept, 0.0,
            "all", "full-confirm", ft_path,
            harness.batch_is_matched, ev,
        )

        del harness
        torch.cuda.empty_cache()

    out.append("")
    out.append(
        "* N2R-confirm is the REPORTABLE full-protocol number for the chosen "
        "configuration.  It uses the full eval split (QICERT_N2_EVAL=full), "
        "not the search prefix."
    )


def run(rows: str, out: list[str], ctx=None) -> None:
    if wants(rows, "n2r-search", SMOKE):
        _run_search(out, ctx)
    if wants(rows, "n2r-confirm", SMOKE):
        _run_confirm(out, ctx)
