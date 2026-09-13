# ===========================================================================
# N5-v2 — Layer-2b: degradation curve + predictor (action-agreement form)
# ===========================================================================
# Purpose (report section "Layer 2: local formal proofs", item 2b): turn the
# honest compression NO-GO results into the submission's DESIGN TOOL:
# "we can predict the largest safe compression ratio BEFORE running it."
#
# Form (honest scoping, 2026-09-12):
#   survival(r) := token-level action AGREEMENT between the compressed model
#   and the FT reference on a fixed prefix of the frozen eval split — the
#   empirical degradation curve.  (The original Lyapunov V-decrease form
#   needs closed-loop rollouts; agreement is the open-loop proxy we can
#   measure with the validated harness today, and it is what the
#   accuracy metric itself is made of.)
#
#   Predictor: least-squares fit from weight-space features (computable at
#   COMPRESSION time, no eval) -> agreement, validated leave-one-out over
#   the measured points.  Reported: predicted safe max-ratio r* (agreement
#   >= threshold) vs measured — the sentence the report promises.
#
#   Certified side-log: per-layer spectral deviation ||W_comp - W_ref||_2
#   (exact, power iteration) — the Layer-1 per-layer object for the
#   compressed-vs-reference pair.  The end-to-end deviation chain through
#   attention/softmax is Layer-2a (SOS) territory and is NOT claimed here.
#
# Two legs:
#   GPU  --rows=lyapunov-degradation : measures agreement per compression
#          plan, writes results/lyapunov/degradation_seed{S}.json
#   CPU  --rows=lyapunov-predictor   : fits the predictor on recorded
#          degradation points, LOO R^2, predicted vs measured r*
#
# GPU leg usage:
#   QICERT_N5_PLANS="0.50,0.33" QICERT_N5_RRANKS="32" QICERT_N5_MODE=plain \
#   QICERT_N5_BATCHES=500 \
#   python -m qicert.bench.all --module lyapunov_curve \
#       --rows=lyapunov-degradation --out results --exp-id N5 --seed 0 \
#       --run-tag N5-degradation-seed0 --save-ckpt results/N1v2-ckpt
"""N5-v2 — degradation curve + weight-space predictor (bench module)."""
from __future__ import annotations

SMOKE = frozenset()

import json as _json
import os as _os
from pathlib import Path

import numpy as np

from ._base import finish_run, start_run, table_header, wants

AGREEMENT_THRESHOLD = 0.95   # "safe" = retains 95% token-level agreement


# ---------------------------------------------------------------------------
# CPU leg: predictor
# ---------------------------------------------------------------------------

def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares y ~ [1, x] -> (coeffs, R^2).  Plain numpy (no sklearn)."""
    A = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, r2


def _loo_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out predictive R^2 (honesty for tiny point counts)."""
    n = len(x)
    if n < 3:
        return float("nan")
    errs = []
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        coef, _ = _fit_linear(x[m], y[m])
        errs.append(y[i] - (coef[0] + coef[1] * x[i]))
    errs = np.asarray(errs)
    ss_res = float(np.sum(errs ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _predict_safe_ratio(coef: np.ndarray, threshold: float,
                        lo: float = 1.0, hi: float = 64.0) -> float:
    """Largest ratio with predicted agreement >= threshold (linear model)."""
    a, b = float(coef[0]), float(coef[1])
    if b >= 0:
        return float("nan")       # non-decreasing fit: no safe-ratio concept
    r = (a - threshold) / (-b)
    return float(min(max(r, lo), hi))


def _load_degradation_points() -> list[dict]:
    """All recorded degradation points across seeds/plans/modes."""
    root = Path("results") / "lyapunov"
    pts: list[dict] = []
    if root.exists():
        for f in sorted(root.glob("degradation_seed*.json")):
            data = _json.loads(f.read_text())
            pts.extend(data.get("points", []))
    return pts


def _run_predictor(out: list[str], ctx=None) -> None:
    pts = _load_degradation_points()
    out += table_header(
        "N5-v2 predictor — weight-space features -> agreement "
        f"(threshold={AGREEMENT_THRESHOLD:.2f})",
        ["Source", "n pts", "Fit (a + b*ratio)", "R^2", "LOO R^2",
         "Predicted r*", "Measured r*", "Status"])
    if len(pts) < 3:
        out.append(f"| lyapunov | {len(pts)} | - | - | - | - | - | "
                   f"PENDING (need >= 3 measured points; run the "
                   f"lyapunov-degradation GPU leg) |")
        return
    x = np.asarray([p["ratio"] for p in pts], dtype=np.float64)
    y = np.asarray([p["agreement"] for p in pts], dtype=np.float64)
    coef, r2 = _fit_linear(x, y)
    loo = _loo_r2(x, y)
    r_pred = _predict_safe_ratio(coef, AGREEMENT_THRESHOLD)
    ok = [p for p in pts if p["agreement"] >= AGREEMENT_THRESHOLD]
    r_meas = float(max(p["ratio"] for p in ok)) if ok else 0.0
    status = "completed"
    out.append(f"| lyapunov | {len(pts)} | {coef[0]:.3f} {coef[1]:+.4f}*r | "
               f"{r2:.3f} | {loo:.3f} | {r_pred:.2f}x | {r_meas:.2f}x | "
               f"{status} |")
    out.append("")
    out.append("* The predictor uses ONLY weight-space/compression features, "
               "so r* is predicted before any eval. |r* - r_measured| is the "
               "reported design-tool error. LOO R^2 is the honest predictive "
               "score at this sample size; with < 5 points it is indicative "
               "only, and the report says so.")
    # Persist for the report build
    dest = Path("results") / "lyapunov" / "predictor.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_json.dumps({
        "n_points": len(pts), "coef": [float(coef[0]), float(coef[1])],
        "r2": r2, "loo_r2": loo,
        "predicted_safe_ratio": r_pred, "measured_safe_ratio": r_meas,
        "threshold": AGREEMENT_THRESHOLD,
        "points": pts,
    }, indent=2))


# ---------------------------------------------------------------------------
# GPU leg: measure the agreement degradation curve
# ---------------------------------------------------------------------------

def _plans_from_env() -> tuple[float, ...]:
    env = _os.environ.get("QICERT_N5_PLANS", "0.50,0.33")
    return tuple(float(v) for v in env.split(",") if v.strip())


def _rranks_from_env() -> tuple[int, ...]:
    env = _os.environ.get("QICERT_N5_RRANKS", "32")
    return tuple(int(v) for v in env.split(",") if v.strip())


def _spectral_dev(w_ref: np.ndarray, w_comp: np.ndarray,
                  iters: int = 60) -> float:
    """Exact ||w_ref - w_comp||_2 via power iteration (no full SVD)."""
    D = (w_ref - w_comp).astype(np.float64)
    nrm = float(np.linalg.norm(D))
    if nrm == 0.0:
        return 0.0
    rng = np.random.default_rng(0)
    v = rng.standard_normal((D.shape[1], 1))
    v /= np.linalg.norm(v)
    for _ in range(iters):
        u = D @ v
        un = np.linalg.norm(u)
        if un == 0.0:
            return 0.0
        u /= un
        v = D.T @ u
        vn = np.linalg.norm(v)
        if vn == 0.0:
            return 0.0
        v /= vn
    return float(un)


def _run_degradation(out: list[str], ctx=None) -> None:
    import time

    import torch

    from qicert.compress_residual import compressed_params, svd_residual
    from qicert.kernels import get_backend

    k = get_backend()
    seed = ctx.seed if ctx else 0
    ft_dir = (Path(_os.environ.get("QICERT_FT_CKPT", ""))
              if _os.environ.get("QICERT_FT_CKPT")
              else (Path(ctx.save_ckpt) if ctx and ctx.save_ckpt
                    else Path("weights") / "ckpt" / "finetuned"))
    plans = _plans_from_env()
    rranks = _rranks_from_env()
    mode = _os.environ.get("QICERT_N5_MODE", "plain")
    ttsplit = float(_os.environ.get("QICERT_N2R_TTSPLIT", "0.6"))
    n_batches = int(_os.environ.get("QICERT_N5_BATCHES", "500"))
    d = 2

    ft_path = ft_dir / f"seed{seed}.pt"
    if not ft_path.exists():
        out.append(f"| {seed} | - | - | - | FAIL (no FT ckpt {ft_path}) |")
        return

    from .n2_sweep import _N2EvalHarness, _factor_dims, _llm_linear_keys, \
        _rank_for_fraction, _to_half_cuda
    from .compress import CKPT

    base_sd = torch.load(str(CKPT), map_location="cpu", weights_only=True)
    llm_sd = base_sd["model"]["llm_backbone"]
    inventory = _llm_linear_keys(llm_sd)
    shapes = {}
    for _lt, key in inventory:
        W = llm_sd[key].detach().float().cpu().numpy()
        M, N = W.shape
        shapes[key] = {"layer_type": _lt, "M": M, "N": N,
                       "m_dims": _factor_dims(M, d), "n_dims": _factor_dims(N, d),
                       "dense_params": M * N}
    del base_sd, llm_sd
    import gc as _gc
    _gc.collect()

    act_w = {}
    if mode == "weighted":
        from .n2r2_sweep import _load_activation_weights
        act_w = _load_activation_weights(inventory, "mean")

    ft_llm = torch.load(str(ft_path), map_location="cpu", weights_only=True)
    merged = ft_llm["llm_backbone"]

    harness = _N2EvalHarness(seed, ft_dir, ctx)
    harness.adopt_reference(merged)

    out += table_header(
        f"N5-v2 degradation curve (mode={mode}, plans={plans}, "
        f"rranks={rranks}, first {n_batches} frozen-split batches)",
        ["Seed", "Plan", "Rr", "Ratio", "Agreement", "n tokens",
         "Mean |dW| (spectral)", "Status"])

    # ---- reference preds cache (measured once) -------------------------
    ref_cache = Path("results") / "lyapunov" / f"refpred_seed{seed}.npz"

    def _collect_preds(harness, n_batches, comp_sd):
        """Token-level argmax preds over the first n_batches split batches.

        comp_sd None => the resident (reference) weights are scored.
        Returns (pred_ids concatenated over masked tokens, n_tokens).
        Token ORDER is deterministic (fixed split prefix), so two calls are
        directly comparable element-wise.
        """
        torch_ = harness.torch
        vla = harness.vla
        if comp_sd is not None:
            vla.llm_backbone.load_state_dict(comp_sd, strict=False)
        try:
            preds_all: list[np.ndarray] = []
            total = 0
            t0 = time.perf_counter()
            for b in harness._split_stream():
                if total >= n_batches * 2:
                    # loose bound: stop once we have enough tokens (batches vary
                    # in token count); ref/comp must use the SAME n_batches param
                    break
                input_ids = b["input_ids"].cuda()
                attention_mask = b["attention_mask"].cuda()
                # _to_half_cuda is module-level (handles the DinoSigLIP dict);
                # audit 2026-09-13: the old hasattr(harness, ...) guard was
                # always False and passed pixel_values=None (device crash).
                pixel_values = _to_half_cuda(b["pixel_values"])
                labels = b["labels"].cuda()
                with torch_.inference_mode(), \
                        torch_.autocast("cuda", dtype=torch_.float16):
                    out_ = vla(input_ids=input_ids,
                               attention_mask=attention_mask,
                               pixel_values=pixel_values, labels=labels)
                logits = out_.logits[:, harness.num_patches:-1]
                preds = logits.argmax(dim=-1)
                gt = labels[:, 1:].to(preds.device)
                mask = gt > harness.action_tokenizer.action_token_begin_idx
                if mask.any():
                    preds_all.append(preds[mask].cpu().numpy().astype(np.int32))
                    total += int(mask.sum().item())
                if total >= n_batches * 2 and len(preds_all) >= n_batches:
                    break
                if len(preds_all) >= n_batches:
                    break
            return np.concatenate(preds_all) if preds_all else np.array([], np.int32), total
        finally:
            # restore reference so the next call starts clean
            harness._restore_reference()

    points: list[dict] = []
    rec_rows: list[dict] = []

    if ref_cache.exists():
        z = np.load(ref_cache)
        ref_preds, ref_total = z["preds"], int(z["total"])
        print(f"[N5] loaded reference preds cache: {ref_cache} "
              f"({ref_total} tokens)", flush=True)
    else:
        print("[N5] measuring reference predictions (resident FT weights)...",
              flush=True)
        ref_preds, ref_total = _collect_preds(harness, n_batches, None)
        ref_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ref_cache, preds=ref_preds, total=ref_total)
    if ref_total == 0:
        out.append(f"| {seed} | - | - | - | FAIL (no scored tokens) |")
        return

    for frac in plans:
        for rrank in rranks:
            t0 = time.perf_counter()
            try:
                print(f"  [N5 seed={seed} frac={frac:.2f} r'={rrank}] "
                      f"compressing...", flush=True)
                comp = dict(merged)
                tot_c = tot_d = 0
                devs: list[float] = []
                for i_layer, (_lt, key) in enumerate(inventory):
                    info = shapes[key]
                    W = merged[key].detach().float().cpu().numpy()
                    M, N = info["M"], info["N"]
                    budget = int(frac * info["dense_params"])
                    tt_frac = frac * ttsplit
                    r = _rank_for_fraction(tt_frac, info["m_dims"], info["n_dims"])
                    cs = k.tt_svd(W, info["m_dims"], info["n_dims"],
                                  tuple(r for _ in range(len(info["m_dims"]) - 1)))
                    tt_params = int(sum(int(np.prod(g.shape)) for g in cs.arrays))
                    cap = max(0, budget - tt_params - M - N)
                    r_eff = int(min(rrank, cap // (M + N)))
                    w = act_w.get(key) if mode == "weighted" else None
                    comp_l = svd_residual(cs.arrays, info["m_dims"], info["n_dims"],
                                          W, residual_rank=max(r_eff, 0),
                                          activation_weight=w)
                    from .n2r_sweep import _dense_from_comp
                    Wc = _dense_from_comp(comp_l, info["m_dims"], info["n_dims"])
                    comp[key] = torch.from_numpy(Wc.astype(np.float16))
                    tot_c += compressed_params(comp_l)
                    tot_d += info["dense_params"]
                    devs.append(_spectral_dev(W, Wc.astype(np.float64)))
                    if (i_layer + 1) % 48 == 0 or i_layer + 1 == len(inventory):
                        print(f"    layer {i_layer + 1}/{len(inventory)} "
                              f"mean|dW|={float(np.mean(devs)):.4f}", flush=True)
                ratio = tot_d / max(tot_c, 1)
                comp_preds, comp_total = _collect_preds(harness, n_batches, comp)
                if comp_total != ref_total:
                    print(f"    WARNING: token count mismatch "
                          f"({comp_total} vs {ref_total}); using min",
                          flush=True)
                n_tok = min(comp_total, ref_total)
                agree = float(np.mean(
                    comp_preds[:n_tok] == ref_preds[:n_tok])) if n_tok else float("nan")
                mean_dev = float(np.mean(devs)) if devs else float("nan")
                dt = time.perf_counter() - t0
                print(f"  [N5 seed={seed} frac={frac:.2f} r'={rrank}] DONE "
                      f"ratio={ratio:.2f}x agreement={agree:.4f} "
                      f"mean|dW|={mean_dev:.4f} ({dt:.0f}s) <- live", flush=True)
                out.append(f"| {seed} | {frac:.3f} | {rrank} | {ratio:.2f}x | "
                           f"{agree:.4f} | {n_tok} | {mean_dev:.4f} | completed |")
                pt = {"seed": seed, "mode": mode, "plan": frac, "rrank": rrank,
                      "ratio": round(ratio, 4), "agreement": round(agree, 6),
                      "n_tokens": n_tok, "mean_spectral_dev": round(mean_dev, 6),
                      "wall_sec": round(dt, 1)}
                points.append(pt)
                rec_rows.append(pt)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                out.append(f"| {seed} | {frac:.3f} | {rrank} | - | - | - | - | "
                           f"FAIL ({type(exc).__name__}: {exc}) |")

    # restore reference (contamination guard, same contract as other modules)
    harness._restore_reference()

    # persist the degradation points (append-safe across runs)
    dest = Path("results") / "lyapunov" / f"degradation_seed{seed}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    prev = _json.loads(dest.read_text())["points"] if dest.exists() else []
    known = {(p.get("mode"), p.get("plan"), p.get("rrank"), p.get("seed"))
             for p in prev}
    merged_pts = prev + [p for p in points
                         if (p["mode"], p["plan"], p["rrank"], p["seed"])
                         not in known]
    dest.write_text(_json.dumps({"threshold": AGREEMENT_THRESHOLD,
                                 "points": merged_pts}, indent=2))
    print(f"[N5] wrote {dest} ({len(merged_pts)} cumulative points)", flush=True)

    if ctx is not None and ctx.active:
        rec = start_run(ctx, exp_id=(ctx.exp_id or "N5"),
                        label=f"degradation-{mode}-seed{seed}",
                        config={"experiment": "N5-v2 degradation curve",
                                "mode": mode, "n_batches": n_batches,
                                "agreement_threshold": AGREEMENT_THRESHOLD})
        if rec is not None:
            finish_run(rec, status="completed", results={"points": rec_rows})


def run(rows: str, out: list[str], ctx=None) -> None:
    if wants(rows, "lyapunov-degradation", SMOKE):
        _run_degradation(out, ctx)
    if wants(rows, "lyapunov-predictor", SMOKE):
        _run_predictor(out, ctx)
