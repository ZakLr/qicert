# ========================================================================
# N2 - {TT, QTT} x TT-cross compression sweep: Pareto curve + certificates
# ========================================================================
# Pre-registered (submission/08-experiments.md): "{TT, QTT} x TT-cross sweep,
# 6 bond plans x 2 backbones", 3 seeds, 24 GPU-h. Output: Pareto curve
# (compression ratio vs action accuracy, measured against the N1 fine-tuned
# baseline at matched budget) + exact Lipschitz table per point.
#
# Design decisions (2026-08-16, honest-scrutiny pass):
#   * Bond plans are TARGET PARAMETER FRACTIONS of the compressed layers
#     {0.5, 1, 2, 4, 8, 16}%; the per-layer rank solves params(r) = fraction
#     x dense params. This is the uniform-ratio allocator that N5 compares
#     its safety-budgeted allocator against — the correct baseline.
#   * Kernel: tt_svd (deterministic reference). tt_cross is the
#     query-based approximation for the hardware story (Q19); it crashed on
#     deep splits (fixed in this commit) and is validated separately in
#     Step 0. Documented substitution, not a silent one (AGENTS.md).
#   * Both backbones use the N2' winner (bit-reversed) bit ordering:
#     TT = d=2 near-square factors, QTT = d=4, factors descending
#     (least-significant-first semantics).
#   * Compression targets: all linear projections of the LLM transformer
#     (24 layers x q/k/v/o/gate/up/down). The action head is excluded in v1
#     (tiny; the Q17 ordering puts it first — future work).
#   * Eval: the SAME held-out batch and action-accuracy protocol as N1
#     (matched budget), on the FINE-TUNED weights (N1 --save-ckpt).
#   * Certificate per point: per-layer exact Lipschitz product L and tight
#     operator norm (power iteration); sound iff L >= tight (N3 machinery).
"""N2 - {TT, QTT} compression Pareto sweep (bench module).

Usage:
    python -m qicert.bench.all --module n2_sweep --rows=compression-pareto
    python -m qicert.bench.all --module n2_sweep --rows=recon-sweep \\
        --out results --exp-id N2pre --run-tag 5060-recon
"""
from __future__ import annotations

# Target param fractions of the compressed layers, spanning 2x..50x
# compression and including the 50% (2x) point where R1's kill criterion
# ("accuracy drop >5% at 2x") is evaluated.
BOND_PLANS = (0.02, 0.04, 0.08, 0.16, 0.33, 0.50)
BACKBONES = ("TT", "QTT")  # d=2 vs d=4, both bit-reversed

# N2 needs the N1 fine-tuned handoff (--save-ckpt), so it is NOT part of the
# no-weights --rows=smoke set; it runs via --module n2_sweep --rows=compression-pareto.
SMOKE = frozenset()

import os as _os
from pathlib import Path

import numpy as np

from ._base import finish_run, start_run, table_header, wants
from .compress import CKPT, DATA_ROOT

def run(rows: str, out: list[str], ctx=None) -> None:
    if wants(rows, "compression-pareto", SMOKE):
        _run_n2(out, ctx)
    if wants(rows, "recon-sweep", frozenset()):
        _run_recon_sweep(out, ctx)


# ========================================================================
# N2pre (E4) - TT-SVD reconstruction sweep: per (layer-type x depth x
# target param fraction), record rel err + L_raw / L_min(gauge) / tight /
# tt-param count / wall time. No dataset, no eval leg — the pure
# reconstruction-and-certificate surface that sizes the N2 bond allocator.
# ========================================================================
RECON_DEPTHS = (0, 7, 15, 23)
RECON_FRACTIONS = (0.005, 0.01, 0.02, 0.04)


def _run_recon_sweep(out: list[str], ctx=None) -> None:
    import re
    import time

    import torch

    from qicert.kernels import get_backend

    k = get_backend()
    base_sd = torch.load(str(CKPT), map_location="cpu", weights_only=True)
    llm_sd = base_sd["model"]["llm_backbone"]

    inventory = []
    for layer_type, key in _llm_linear_keys(llm_sd):
        depth = int(re.search(r"layers\.(\d+)\.", key).group(1))
        if depth in RECON_DEPTHS:
            inventory.append((layer_type, key, depth))

    rec = start_run(ctx,
                    exp_id=(ctx.exp_id if ctx and ctx.exp_id else "N2pre"),
                    label="recon-sweep",
                    config={"experiment": "N2pre TT-SVD reconstruction sweep",
                            "backbone": "TT d=2 bit-reversed",
                            "depths": list(RECON_DEPTHS),
                            "fractions": list(RECON_FRACTIONS),
                            "checkpoint": str(CKPT),
                            "note": "per-cell recon rel-err + certificates "
                                    "(E1 gauge descent); CPU numpy kernels"})

    out += table_header(
        f"N2pre - reconstruction sweep ({len(inventory)} layers x "
        f"{len(RECON_FRACTIONS)} fractions, TT-SVD d=2 bit-reversed)",
        ["Layer", "Depth", "M x N", "Frac", "r", "TT params", "Rel err",
         "L_raw", "L_min", "Tight", "kappa_min", "Sec", "Status"])

    n_fail = 0
    n_mono_warn = 0
    for layer_type, key, depth in inventory:
        W = llm_sd[key].detach().float().cpu().numpy()
        M, N = W.shape
        m_dims = _factor_dims(M, 2)
        n_dims = _factor_dims(N, 2)
        prev_err = None
        for frac in RECON_FRACTIONS:
            r = _rank_for_fraction(frac, m_dims, n_dims)
            t0 = time.perf_counter()
            try:
                cs = k.tt_svd(W, m_dims, n_dims,
                              tuple(r for _ in range(len(m_dims) - 1)))
                Wr = k.contract_cores(cs.arrays, m_dims, n_dims)
                rel = float(np.linalg.norm(Wr - W) / np.linalg.norm(W))
                L = float(k.lipschitz_product(cs.arrays))
                # Gauge descent capped at 10 sweeps here (wall-clock: the
                # full 50-sweep budget only buys marginal extra tightening
                # on the biggest cells; N3's committed table keeps the full
                # budget).
                L_min, gst = k.min_gauge_product(cs.arrays, sweeps=10)
                tight = float(k.operator_norm_tight(cs.arrays, m_dims, n_dims))
                dt = time.perf_counter() - t0
                cparams = int(sum(np.prod(g.shape) for g in cs.arrays))
                kappa_min = L_min / max(tight, 1e-12)
                ok = bool(np.isfinite(rel) and np.isfinite(L_min)
                          and L_min >= tight - 1e-9)
                out.append(f"| {layer_type} | {depth} | {M}x{N} | "
                           f"{frac:.3f} | {r} | {cparams} | {rel:.5f} | "
                           f"{L:.4f} | {L_min:.4f} | {tight:.4f} | "
                           f"{kappa_min:.2f} | {dt:.1f} | "
                           f"{'PASS' if ok else 'FAIL'} |")
                if rec:
                    rec.metric(layer=layer_type, depth=depth, fraction=frac,
                               rank=int(r), tt_params=cparams,
                               recon_rel_err=rel, lipschitz=float(L),
                               lipschitz_gauge_min=float(L_min),
                               tight_norm=float(tight),
                               kappa_min=float(kappa_min), sound=bool(ok),
                               wall_sec=round(dt, 2),
                               gauge_converged=bool(gst["converged"]))
                # Gate sanity: rel err must be monotone non-increasing in
                # the fraction (more params cannot hurt optimal truncation).
                if prev_err is not None and rel > prev_err * (1 + 1e-6):
                    n_mono_warn += 1
                    print(f"  [mono-warn] {layer_type} depth {depth}: rel err "
                          f"rose {prev_err:.5f} -> {rel:.5f} at frac {frac}",
                          flush=True)
                prev_err = rel
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                out.append(f"| {layer_type} | {depth} | {M}x{N} | "
                           f"{frac:.3f} | {r} | - | - | - | - | - | - | - | "
                           f"FAIL ({type(exc).__name__}: {exc}) |")

    n_cells = len(inventory) * len(RECON_FRACTIONS)
    out.append("")
    verdict = (f"N2pre sweep complete: {n_cells - n_fail}/{n_cells} cells PASS, "
               f"{n_mono_warn} monotonicity warnings.")
    out.append(f"* {verdict}")
    if rec:
        rec.sample_power()
        finish_run(rec, status="completed" if n_fail == 0 else "failed",
                   results={"cells": n_cells, "failed": n_fail,
                            "monotonicity_warnings": n_mono_warn,
                            "verdict": verdict},
                   tolerance_note="TT-SVD is exact-rank adaptive; rel err is "
                                  "vs the dense weight; kappa = L_min/tight")




def _factor_dims(n: int, d: int, bit_reversed: bool = True) -> tuple[int, ...]:
    """Split n into d factors of n as evenly as possible.

    bit_reversed=True (N2' winner): largest factors first, i.e. the
    least-significant index digits are grouped first (Oseledets-style).
    """
    factors = []
    cur = n
    for i in range(d - 1):
        a = int(round(cur ** (1.0 / (d - i))))
        while cur % a != 0 and a > 1:
            a -= 1
        if a <= 1:
            a = cur
        factors.append(a)
        cur //= a
    factors.append(cur)
    return tuple(reversed(factors)) if bit_reversed else tuple(factors)


def _core_params(r: int, m_dims: tuple[int, ...], n_dims: tuple[int, ...],
                 d: int) -> int:
    """Parameter count of a uniform-rank TT-matrix (d modes, one bond r)."""
    total = 0
    for k in range(d):
        r_prev = 1 if k == 0 else r
        r_next = 1 if k == d - 1 else r
        total += r_prev * m_dims[k] * n_dims[k] * r_next
    return total


def _rank_for_fraction(fraction: float, m_dims, n_dims) -> int:
    """Uniform rank whose core params hit `fraction` of the dense matrix."""
    d = len(m_dims)
    m, n = int(np.prod(m_dims)), int(np.prod(n_dims))
    target = fraction * m * n
    # params(r) = a*r^2 + b*r with a = sum of middle m_k n_k, b = edge terms
    a = sum(m_dims[k] * n_dims[k] for k in range(1, d - 1))
    b = m_dims[0] * n_dims[0] + m_dims[-1] * n_dims[-1]
    if a <= 0:
        r = int(round(target / max(b, 1)))
    else:
        r = int(round((-b + (b * b + 4 * a * target) ** 0.5) / (2 * a)))
    return max(1, r)


def _llm_linear_keys(sd) -> list[tuple[str, str]]:
    """(layer_type, state-dict key) for every linear projection in the LLM."""
    import re
    out = []
    pat = re.compile(
        r"llm\.model\.layers\.\d+\.(self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight$")
    for key in sd:
        m = pat.match(key)
        if m:
            out.append((m.group(1).replace(".", "-"), key))
    return sorted(out)


def _run_n2(out: list[str], ctx=None) -> None:
    import os
    import time

    import numpy as np
    import torch

    from qicert.kernels import get_backend
    k = get_backend()

    seeds = (ctx.seeds if ctx and ctx.seeds else [ctx.seed if ctx else 0])
    # Fine-tuned weights handoff from N1 (--save-ckpt). Env override for the
    # Kaggle kernel layout.
    ft_dir = Path(os.environ.get("QICERT_FT_CKPT", "")) if os.environ.get(
        "QICERT_FT_CKPT") else (Path(ctx.save_ckpt) if ctx and ctx.save_ckpt
                                else Path("weights") / "ckpt" / "finetuned")
    layers_scope = os.environ.get("QICERT_N2_LAYERS", "all")  # 'all' | 'layer0'
    # Plan filter for fast local smoke (e.g. "0.16,0.50"); default = all.
    plans_env = os.environ.get("QICERT_N2_PLANS", "")
    plans = tuple(float(x) for x in plans_env.split(",") if x.strip()) \
        if plans_env else BOND_PLANS

    # Load the BASE checkpoint once: we need the flat key inventory and, for
    # each layer, the dense weight to compute per-layer fractions/ranks.
    base_sd = torch.load(str(CKPT), map_location="cpu", weights_only=True)
    llm_sd = base_sd["model"]["llm_backbone"]

    inventory = _llm_linear_keys(llm_sd)
    if layers_scope == "layer0":
        inventory = [(t, key) for t, key in inventory
                     if ".layers.0." in key]
    if not inventory:
        raise RuntimeError("no LLM linear layers found in checkpoint")

    # Pre-compute mode dims + dense params per layer (shared across plans).
    plan_info = {}
    for layer_type, key in inventory:
        W = llm_sd[key].detach().float().cpu().numpy()
        M, N = W.shape
        dims = {}
        for bb in BACKBONES:
            d = 2 if bb == "TT" else 4
            dims[bb] = (_factor_dims(M, d), _factor_dims(N, d))
        plan_info[key] = {"layer_type": layer_type, "M": M, "N": N,
                          "dense_params": M * N, "dims": dims}

    # The base checkpoint is loaded ONLY for the key inventory + shapes
    # (compression reads the FT weights, not the base weights). 5.3 GB must
    # NOT stay resident for the whole run — release it before the sweep
    # (host-freeze lesson 2026-09-11: two parallel N1v2 runs + resident
    # checkpoints locked the machine).
    del base_sd, llm_sd
    import gc as _gc
    _gc.collect()

    out += table_header(
        f"N2 - {layers_scope} compression Pareto sweep "
        f"({len(inventory)} layers, plans={plans}, backbones={BACKBONES}, "
        f"seeds={seeds})",
        ["Seed", "Backbone", "Plan (frac)", "Ratio", "Eval acc",
         "Delta vs N1 FT", "Cert sound", "Status"])

    for seed in seeds:
        # The recorder reads ctx.seed; the worker is launched with --seeds N,
        # so make ctx.seed follow the seed being run (ledger provenance).
        if ctx is not None:
            ctx.seed = seed
        # Load the fine-tuned backbone for this seed (N1 handoff).
        ft_path = ft_dir / f"seed{seed}.pt"
        if not ft_path.exists():
            out.append(f"| {seed} | - | - | - | - | - | - | "
                       f"FAIL (no fine-tuned ckpt {ft_path}; run N1 --save-ckpt) |")
            continue
        ft_llm = torch.load(str(ft_path), map_location="cpu", weights_only=True)
        merged = ft_llm["llm_backbone"]

        # One VLA load per seed; every Pareto point reuses the same model and
        # the same N1 eval batch (matched budget). This turns 36 model loads
        # (per point) into 3 (per seed) — the dominant wall-clock cost.
        harness = _N2EvalHarness(seed, ft_dir, ctx)
        n1_ft = harness.n1_ft
        if not harness.batch_is_matched:
            out.append(f"* seed {seed}: N1 eval-batch sidecar missing — delta is "
                       "approximate (fresh batch, same slice/protocol)")

        for bb in BACKBONES:
            for frac in plans:
                t0 = time.perf_counter()
                try:
                    # Compress each layer: TT-SVD -> cores -> contract ->
                    # write the reconstruction into a fresh merged dict so
                    # every plan is independent (no mutation between plans).
                    # NOTE: all.py prints the `out` table only at the END, so
                    # all per-point progress MUST go through direct prints
                    # (flush=True) to keep the log live during long runs.
                    print(f"  [N2 seed={seed} {bb} frac={frac:.2f}] "
                          f"compressing {len(inventory)} layers...",
                          flush=True)
                    comp = dict(merged)
                    cert_rows = []
                    total_compressed_params = 0
                    total_dense_params = 0
                    for i_layer, (layer_type, key) in enumerate(inventory):
                        W = merged[key].detach().float().cpu().numpy()
                        m_dims, n_dims = plan_info[key]["dims"][bb]
                        r = _rank_for_fraction(frac, m_dims, n_dims)
                        cs = k.tt_svd(W, m_dims, n_dims, tuple(
                            r for _ in range(len(m_dims) - 1)))
                        Wr = k.contract_cores(cs.arrays, m_dims, n_dims)
                        comp[key] = torch.from_numpy(Wr)
                        cparams = int(sum(np.prod(g.shape) for g in cs.arrays))
                        total_compressed_params += cparams
                        total_dense_params += plan_info[key]["dense_params"]
                        # certificate: exact Lipschitz product vs tight norm
                        L = float(k.lipschitz_product(cs.arrays))
                        tight = float(k.operator_norm_tight(
                            cs.arrays, m_dims, n_dims))
                        cert_rows.append((layer_type, L, tight,
                                          bool(L >= tight - 1e-9)))
                        if (i_layer + 1) % 24 == 0 or i_layer + 1 == len(inventory):
                            print(f"    layer {i_layer + 1}/{len(inventory)} "
                                  f"({layer_type}) L={L:.3f} tight={tight:.3f} "
                                  f"sound={L >= tight - 1e-9}", flush=True)
                    ratio = total_dense_params / max(total_compressed_params, 1)
                    n_sound = sum(1 for _, _, _, ok in cert_rows if ok)

                    acc = harness.eval(comp)
                    delta = (acc - n1_ft) if n1_ft is not None and acc == acc \
                        else float("nan")
                    dt = time.perf_counter() - t0
                    status = "completed" if acc == acc else "failed"
                    print(f"  [N2 seed={seed} {bb} frac={frac:.2f}] DONE "
                          f"ratio={ratio:.2f}x acc={acc:.4f} "
                          f"delta={delta:+.4f} sound={n_sound}/{len(cert_rows)} "
                          f"({dt:.0f}s) <- live", flush=True)
                    out.append(f"| {seed} | {bb} | {frac:.3f} | {ratio:.2f}x | "
                               f"{acc:.4f} | {delta:+.4f} | {n_sound}/"
                               f"{len(cert_rows)} | {status} | ({dt:.0f}s) |")

                    _record_n2_point(ctx, seed, bb, frac, ratio, acc, delta,
                                     cert_rows, dt, layers_scope, ft_path,
                                     matched_batch=harness.batch_is_matched)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    print(f"  [N2 seed={seed} {bb} frac={frac:.2f}] FAILED: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    out.append(f"| {seed} | {bb} | {frac:.3f} | - | - | - | "
                               f"- | FAIL ({type(exc).__name__}: {exc}) |")

    out.append("")
    out.append("* N2's Pareto points are measured against the N1 fine-tuned "
               "baseline at matched budget (same eval batch, same protocol). "
               "R1 kill criterion: if TT loses to INT8 on accuracy AND no "
               "certificate advantage exists at any ratio, the claim demotes "
               "to 'accuracy-parity at compression + certificate story'.")


def _record_n2_point(ctx, seed, bb, frac, ratio, acc, delta, cert_rows,
                     dt, layers_scope, ft_path, matched_batch: bool = True) -> None:
    """Record one Pareto point into the ledger via the run recorder."""
    rec = start_run(ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "N2"),
                    label=f"pareto-{bb}-{frac:.3f}-seed{seed}",
                    config={
                        "experiment": "N2 compression Pareto sweep",
                        "backbone": bb, "plan_fraction": frac,
                        "layers_scope": layers_scope,
                        "seed": seed, "ft_ckpt": str(ft_path),
                        "kernel": "tt_svd (reference; tt_cross = Q19)",
                        "ordering": "bit-reversed (N2' winner)",
                        "eval": "N1 matched-budget protocol",
                        "matched_batch": matched_batch,
                    })
    if rec is None:
        return
    for layer_type, L, tight, ok in cert_rows:
        rec.metric(layer=layer_type, lipschitz=float(L),
                   tight_norm=float(tight), sound=bool(ok))
    finish_run(rec, status="completed" if acc == acc else "failed",
               results={"seed": seed, "backbone": bb, "plan_fraction": frac,
                        "ratio": round(float(ratio), 3),
                        "eval_acc": float(acc),
                        "delta_vs_n1_ft": None if delta != delta else round(float(delta), 4),
                        "cert_sound_layers": sum(1 for *_, ok in cert_rows if ok),
                        "matched_batch": bool(matched_batch),
                        "wall_sec": round(float(dt), 2),
                        "note": "uniform-ratio bond plan (N5 baseline allocator)"})


class _N2EvalHarness:
    """Loads the VLA once per seed + the EXACT N1 eval batch, then scores any
    number of compressed state dicts on that fixed batch (matched budget: the
    same inputs the N1 baseline row for this seed was measured on).

    Falls back to drawing a fresh batch from the dataset only when the N1
    sidecar (eval_batch_seed{N}.pt) is missing, in which case the delta is
    flagged as approximate (different random batch, same slice/protocol).
    """

    def __init__(self, seed: int, ft_dir: Path, ctx=None):
        import sys
        import json

        import torch

        repo = Path(__file__).resolve().parents[1]
        fork = Path(_os.environ.get("QICERT_FORK", "")) if _os.environ.get(
            "QICERT_FORK") else repo / "weights" / "code"
        if str(fork) not in sys.path:
            sys.path.insert(0, str(fork))
        # Same requirement as N1: prismatic's datasets conf reads this env var
        # at import time. Set it before the first prismatic import.
        _os.environ.setdefault("PRISMATIC_DATA_ROOT",
                               str(repo / "weights" / "data"))
        # transformers 5.x alias shim (same as N1) before prismatic imports.
        from qicert.transformers5_compat import install as _tf5_install
        _tf5_install()

        self.torch = torch
        torch.manual_seed(seed)
        from prismatic.models.load import load_vla
        from prismatic.vla.action_tokenizer import ActionTokenizer

        vla = load_vla(str(CKPT), hf_token=None, load_for_training=False)
        vla = vla.to(dtype=torch.float16, device="cuda")
        vla.llm_backbone.eval()
        self.vla = vla
        # FT reference weights, snapshotted to CPU at load time (BEFORE any
        # compressed weights are ever loaded). Used by eval() to restore the
        # exact FT state after every point.
        self._ft_sd_cpu = {kk: vv.detach().to("cpu")
                           for kk, vv in vla.llm_backbone.state_dict().items()}
        self.num_patches = int(vla.vision_backbone.num_patches)
        self.action_tokenizer = ActionTokenizer(vla.llm_backbone.tokenizer)

        # N1 handoff sidecars: exact eval batch + this seed's FT baseline acc.
        batch_path = ft_dir / f"eval_batch_seed{seed}.pt"
        baseline_path = ft_dir / f"baseline_seed{seed}.json"
        self.n1_ft = None
        self.batch_is_matched = batch_path.exists()
        if batch_path.exists():
            b = torch.load(str(batch_path), map_location="cpu",
                           weights_only=True)
            self.batch = b
        else:
            self.batch = self._draw_batch(vla)
        if baseline_path.exists():
            try:
                self.n1_ft = float(json.loads(baseline_path.read_text())[
                    "eval_acc_finetuned"])
            except Exception:
                pass

    def _draw_batch(self, vla):
        """Fresh single batch from the fork's RLDS pipeline (fallback path)."""
        from torch.utils.data import DataLoader
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
        from prismatic.util.data_utils import PaddedCollatorForActionPrediction

        tokenizer = vla.llm_backbone.tokenizer
        prompt_builder_fn = vla.llm_backbone.prompt_builder_fn
        atok = ActionTokenizer(tokenizer)
        batch_transform = RLDSBatchTransform(
            atok, tokenizer,
            image_transform=vla.vision_backbone.get_image_transform(),
            prompt_builder_fn=prompt_builder_fn)
        ds = RLDSDataset(DATA_ROOT, "libero_spatial_no_noops", batch_transform,
                         resize_resolution=vla.vision_backbone.default_image_resolution[1:],
                         shuffle_buffer_size=1000, image_aug=False)
        collator = PaddedCollatorForActionPrediction(
            tokenizer.model_max_length, tokenizer.pad_token_id,
            padding_side="right")
        dl = DataLoader(ds, batch_size=1, collate_fn=collator, num_workers=0)
        return next(iter(dl))

    def eval(self, comp_llm_sd) -> float:
        """Swap compressed weights in, score on the fixed eval batch, restore.

        The restore is a CORRECTNESS guarantee, not hygiene: callers build
        `comp` as a shallow copy of the merged FT dict, so without an exact
        restore the first eval would leave COMPRESSED weights resident in the
        live model and every subsequent Pareto point would be measured on a
        mixture of plans — silently corrupting the whole curve.
        """
        torch = self.torch
        vla = self.vla
        try:
            vla.llm_backbone.load_state_dict(comp_llm_sd, strict=False)

            def _to_half_cuda(x):
                if isinstance(x, dict):
                    return {kk: vv.to(torch.float16).cuda() for kk, vv in x.items()}
                return x.to(torch.float16).cuda()

            with torch.inference_mode():
                b = self.batch
                input_ids = b["input_ids"].cuda()
                attention_mask = b["attention_mask"].cuda()
                pixel_values = _to_half_cuda(b["pixel_values"])
                labels = b["labels"].cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    out_ = vla(input_ids=input_ids, attention_mask=attention_mask,
                               pixel_values=pixel_values, labels=labels)
                logits = out_.logits[:, self.num_patches:-1]
                preds = logits.argmax(dim=-1)
                gt = labels[:, 1:].to(preds.device)
                mask = gt > self.action_tokenizer.action_token_begin_idx
                if not mask.any():
                    return float("nan")
                return float((preds[mask] == gt[mask]).float().mean().item())
        finally:
            # Exact FT restore for the next point (see docstring).
            vla.llm_backbone.load_state_dict(self._ft_sd_cpu, strict=False)
            torch.cuda.empty_cache()
