# ============================================================================
# Post-compression LoRA repair training (the "trained repair" lever)
# ============================================================================
# Purpose: the closed-form residual is a one-shot algebraic fit — it
# reconstructs the ORIGINAL weight matrix (recon err 0.6-0.8 at 2.5x)
# rather than task performance. A small trained adapter on the
# ALREADY-COMPRESSED model can spend the same parameter budget on the
# task instead. This is the literature-standard post-compression recovery
# step.
#
# Pipeline (one bench module, one process):
#   1. compress the seed's FINE-TUNED weights once (TT-SVD + absmax-weighted
#      residual, frac=0.50, rr=64 — the exact N2R2-confirmed config,
#      ratio ~2.54x, full-split acc 0.1640) — same source and math as N2R2;
#   2. load the VLA, install the COMPRESSED weights;
#   3. eval the collapsed model on the frozen split PREFIX -> acc_pre
#      (expected ~ the modal base rate 0.12);
#   4. LoRA-train r=8 (the SAME budget that lifted the base model
#      0.11 -> 0.4468 in N1) on TRAIN episodes only — the frozen eval
#      episodes are excluded from the optimizer (contamination guard);
#   5. merge the adapter, eval on the same frozen prefix -> acc_post;
#   6. certificate bookkeeping: per-layer soundness at compression time,
#      the LoRA delta's spectral norm (pre-merge), and the saved merged
#      weights from which the deployed ball re-derives (one SVD/layer).
#
# Honest protocol notes:
#   * Optimization sees ONLY train episodes (eval_split.json complement).
#   * acc_pre/acc_post use a FIXED PREFIX of the frozen eval split
#     (QICERT_N9_EVAL_BATCHES, default 600) — deterministic, same scorer as
#     every other experiment, but NOT the full 6496-batch protocol. A
#     reported final number requires the full-protocol confirm on the saved
#     repaired checkpoint (results/N9-repaired/).
#   * Soundness is preserved by construction (bounds are computed FROM the
#     deployed weights); what may change is the certified ball width.
"""N9 - post-compression LoRA repair training (bench module).

Usage:
    python -m bench.all --module n9_repair --rows=repair --out results \\
        --exp-id N9 --run-tag repair-frac0.50-rr64
    # smoke (~10 min total): --rows=repair-smoke
"""
from __future__ import annotations

import os as _os
import time
from pathlib import Path

import numpy as np

from ._base import finish_run, start_run, table_header, wants
from .compress import DATA_ROOT

# Needs the N1 fine-tuned handoff + VLA weights + NPZ dataset, so it is NOT
# part of the no-weights --rows=smoke set (same convention as n2_sweep).
SMOKE = frozenset()


def run(rows: str, out: list[str], ctx=None) -> None:
    if wants(rows, "repair", SMOKE):
        _run_repair(out, ctx)
    if wants(rows, "repair-smoke", SMOKE):
        _run_repair_smoke(out, ctx)
    if wants(rows, "repair-confirm", SMOKE):
        _run_repair_confirm(out, ctx)


def _run_repair_smoke(out: list[str], ctx=None) -> None:
    """Smoke: 40 steps, 60 eval batches — pipeline correctness only."""
    _os.environ.setdefault("QICERT_N9_STEPS", "40")
    _os.environ.setdefault("QICERT_N9_EVAL_BATCHES", "60")
    _run_repair(out, ctx)


def _run_repair(out: list[str], ctx=None) -> None:
    import gc as _gc

    import torch

    from qicert.compress_residual import (
        compressed_params,
        layer_report,
        svd_residual,
    )
    from qicert.kernels import get_backend

    k = get_backend()

    seed = ctx.seed if ctx is not None else 0
    if ctx is not None:
        ctx.seed = seed

    # --- fixed, pre-registered configuration --------------------------------
    frac = float(_os.environ.get("QICERT_N9_FRAC", "0.50"))
    rrank = int(_os.environ.get("QICERT_N9_RRANK", "64"))
    stat = _os.environ.get("QICERT_N9_STAT", "absmax")
    ttsplit = float(_os.environ.get("QICERT_N9_TTSPLIT", "0.6"))
    steps = int(_os.environ.get("QICERT_N9_STEPS", "1200"))
    batch = int(_os.environ.get("QICERT_N9_BATCH", "2"))
    lora_r = int(_os.environ.get("QICERT_N9_LORA_R", "8"))
    lr = float(_os.environ.get("QICERT_N9_LR", "5e-4"))
    eval_batches = int(_os.environ.get("QICERT_N9_EVAL_BATCHES", "600"))
    mixed = _os.environ.get("QICERT_N9_MIXED", "0") == "1"

    ft_dir = (Path(_os.environ.get("QICERT_FT_CKPT", ""))
              if _os.environ.get("QICERT_FT_CKPT")
              else (Path(ctx.save_ckpt) if ctx and ctx.save_ckpt
                    else Path("weights") / "ckpt" / "finetuned"))
    ft_path = ft_dir / f"seed{seed}.pt"
    if not ft_path.exists():
        raise RuntimeError(f"no fine-tuned ckpt {ft_path}; run N1 --save-ckpt")

    rec = start_run(ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "N9"),
                    label=f"repair-frac{frac:.3f}-rr{rrank}-seed{seed}",
                    config={
                        "experiment": "N9 post-compression LoRA repair",
                        "frac": frac, "residual_rank_cap": rrank,
                        "calib_stat": stat, "tt_split": ttsplit,
                        "steps": steps, "batch": batch, "lora_r": lora_r,
                        "lr": lr, "eval_batches": eval_batches,
                        "mixed": mixed, "seed": seed,
                        "ft_ckpt": str(ft_path),
                    })

    out += table_header(
        f"N9 repair (compress FT frac={frac} rr={rrank} stat={stat}, then "
        f"LoRA r={lora_r} x {steps} steps, lr={lr}, eval prefix="
        f"{eval_batches} batches)",
        ["Stage", "Value", "Note"])

    # ============ 1. compress the FINE-TUNED weights (same math as N2R2) ====
    # N2R2 compresses the FT llm_backbone (merged), NOT the base checkpoint —
    # same here, otherwise the eval comparison is meaningless.
    from .n2_sweep import (_factor_dims, _llm_linear_keys, _rank_for_fraction,
                           _to_half_cuda)
    from .n2r2_sweep import _is_kept, _keep_types, _load_activation_weights
    from .n2r_sweep import _dense_from_comp

    ft_llm = torch.load(str(ft_path), map_location="cpu", weights_only=True)
    merged_sd = ft_llm["llm_backbone"]
    inventory = _llm_linear_keys(merged_sd)
    keep_types = _keep_types() if mixed else []
    d = 2

    act_w = _load_activation_weights(inventory, stat)

    t0 = time.perf_counter()
    comp: dict[str, torch.Tensor] = {}
    total_comp_params = 0
    total_dense_params = 0
    n_sound = 0
    n_compressed = 0
    n_kept = 0
    for i_layer, (layer_type, key) in enumerate(inventory):
        W = merged_sd[key].detach().float().cpu().numpy()
        M, N = W.shape
        if mixed and _is_kept(layer_type, keep_types):
            comp[key] = merged_sd[key].clone()
            total_comp_params += M * N
            total_dense_params += M * N
            n_kept += 1
            continue
        m_dims = _factor_dims(M, d)
        n_dims = _factor_dims(N, d)
        budget = int(frac * (M * N))
        tt_frac = frac * ttsplit
        r = _rank_for_fraction(tt_frac, m_dims, n_dims)
        cs = k.tt_svd(W, m_dims, n_dims,
                      tuple(r for _ in range(len(m_dims) - 1)))
        tt_params = int(sum(int(np.prod(g.shape)) for g in cs.arrays))
        cap = max(0, budget - tt_params - M - N)
        r_eff = int(min(rrank, cap // (M + N)))
        w = act_w.get(key)
        comp_l = svd_residual(cs.arrays, m_dims, n_dims, W,
                              residual_rank=max(r_eff, 0),
                              activation_weight=w,
                              scale=(stat if w is not None else "none"))
        Wc = _dense_from_comp(comp_l, m_dims, n_dims)
        comp[key] = torch.from_numpy(Wc.astype(np.float16))
        total_comp_params += compressed_params(comp_l)
        total_dense_params += M * N
        n_compressed += 1
        rep = layer_report(comp_l, W, m_dims, n_dims)
        n_sound += 1 if rep["sound"] else 0
        if (i_layer + 1) % 48 == 0 or i_layer + 1 == len(inventory):
            print(f"    layer {i_layer + 1}/{len(inventory)} "
                  f"r'={comp_l['residual_rank']} L={rep['lipschitz_bound']:.3f} "
                  f"recon={rep['recon_rel_err']:.4f}", flush=True)
    del ft_llm, act_w
    _gc.collect()
    ratio = total_dense_params / max(total_comp_params, 1)
    comp_sec = time.perf_counter() - t0
    print(f"[N9] compressed: ratio={ratio:.3f}x sound={n_sound}/{n_compressed} "
          f"kept={n_kept} ({comp_sec:.0f}s)", flush=True)
    out.append(f"| compress | ratio={ratio:.3f}x | sound={n_sound}/"
               f"{n_compressed} kept={n_kept} |")
    rec.metric(event="compress", ratio=round(float(ratio), 4),
               sound_layers=n_sound, n_compressed=n_compressed,
               n_kept=n_kept, wall_sec=round(comp_sec, 1))

    # ============ 2-5. VLA: collapsed eval -> LoRA repair -> eval ==========
    import json as _json

    from peft import LoraConfig, get_peft_model

    from .n2_sweep import _N2EvalHarness

    harness = _N2EvalHarness(seed, ft_dir, ctx)
    n1_ft = harness.n1_ft
    vla = harness.vla
    torch_ = harness.torch

    def _prefix_eval(nb: int, tag: str) -> dict:
        """Fixed-prefix eval over the frozen split (deterministic, fast)."""
        correct = total = n = 0
        t0e = time.perf_counter()
        for b in harness._split_stream():
            c, t = harness._score_batch(b)
            correct += c
            total += t
            n += 1
            if n % 100 == 0 or n == nb:
                el = time.perf_counter() - t0e
                print(f"    [{tag} {n}/{nb}] acc={correct / max(total, 1):.4f} "
                      f"elapsed={el:.0f}s", flush=True)
            if n >= nb:
                break
        acc = float(correct / total) if total else float("nan")
        return {"acc": acc, "correct": correct, "total": total,
                "n_batches": n,
                "scope": f"frozen-split prefix ({n} batches, {total} tokens)"}

    # -- 2. install compressed weights, eval collapsed (acc_pre) --
    vla.llm_backbone.load_state_dict(comp, strict=False)
    del comp
    _gc.collect()
    torch_.cuda.empty_cache()
    print("[N9] eval PRE (collapsed) on frozen prefix...", flush=True)
    ev_pre = _prefix_eval(eval_batches, "eval-pre")
    acc_pre = ev_pre["acc"]
    out.append(f"| pre (collapsed) | {acc_pre:.4f} | {ev_pre['scope']} |")
    rec.metric(event="eval_pre", acc=round(acc_pre, 4),
               correct=ev_pre["correct"], total=ev_pre["total"],
               scope=ev_pre["scope"])

    # -- 4. LoRA on the compressed LLM (same targets/budget as N1) --
    QWEN2_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(r=lora_r, lora_alpha=min(lora_r, 16),
                          target_modules=QWEN2_LINEAR, lora_dropout=0.0,
                          bias="none")
    vla.llm_backbone = get_peft_model(vla.llm_backbone, lora_cfg)
    vla.llm_backbone.print_trainable_parameters()

    from qicert.data.npz_loader import NpzEpisodeDataset
    from qicert.data.vla_local import LocalEpisodeStream, LocalVLABatcher

    lb = LocalVLABatcher(vla)
    action_tokenizer = lb.action_tokenizer
    npz_root = (_os.environ.get("QICERT_NPZ_ROOT")
                or str(DATA_ROOT.parent / "libero_spatial_no_noops_npz"))
    ep_ds = NpzEpisodeDataset(npz_root)

    # CONTAMINATION GUARD: optimizer sees train episodes ONLY.
    split_path = (_os.environ.get("QICERT_EVAL_SPLIT")
                  or str(Path(__file__).resolve().parents[1] / "results"
                         / "eval_split.json"))
    train_episode_ids = None
    if Path(split_path).exists():
        _split = _json.loads(Path(split_path).read_text())
        name_to_idx = {p.name: i for i, p in enumerate(ep_ds.files)}
        eval_idx = {name_to_idx[e] for e in _split["eval"]
                    if e in name_to_idx}
        train_episode_ids = [i for i in range(len(ep_ds.files))
                             if i not in eval_idx]
        print(f"[N9] contamination guard: {len(train_episode_ids)}/"
              f"{len(ep_ds.files)} train episodes (eval excluded)",
              flush=True)
    else:
        print("[N9] WARNING: no eval_split.json; using all episodes "
              "(UNPROTECTED protocol — do not report)", flush=True)

    def _stream(s: int):
        return LocalEpisodeStream(ep_ds, lb, batch_size=batch, seed=s,
                                  episode_ids=train_episode_ids)

    it = iter(_stream(seed))
    num_patches = harness.num_patches

    def _action_acc(out_, labels_t) -> float:
        logits = out_.logits[:, num_patches:-1]
        preds = logits.argmax(dim=-1)
        gt = labels_t[:, 1:].to(preds.device)
        mask = gt > action_tokenizer.action_token_begin_idx
        if not mask.any():
            return float("nan")
        return float((preds[mask] == gt[mask]).float().mean().item())

    opt = torch_.optim.AdamW([p for p in vla.llm_backbone.parameters()
                              if p.requires_grad], lr=lr)
    vla.llm_backbone.train()

    losses, accs = [], []
    t_train = time.perf_counter()
    log_every = max(steps // 25, 1)
    for step in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(_stream(seed * 1000 + step + 1))
            b = next(it)

        input_ids = b["input_ids"].cuda()
        attention_mask = b["attention_mask"].cuda()
        pixel_values = _to_half_cuda(b["pixel_values"])
        labels = b["labels"].cuda()
        with torch_.autocast("cuda", dtype=torch_.float16):
            out_ = vla(input_ids=input_ids, attention_mask=attention_mask,
                       pixel_values=pixel_values, labels=labels)
        loss = out_.loss
        loss.backward()
        opt.step()
        opt.zero_grad()

        losses.append(float(loss.item()))
        accs.append(_action_acc(out_, labels))
        if (step + 1) % log_every == 0 or step + 1 == steps:
            el = time.perf_counter() - t_train
            print(f"  [N9 step {step + 1}/{steps}] loss={losses[-1]:.4f} "
                  f"acc={accs[-1]:.3f} elapsed={el:.0f}s", flush=True)
            rec.metric(event="train", step=step + 1,
                       loss=round(losses[-1], 4), acc=round(accs[-1], 4),
                       wall_sec=round(el, 1))
    train_sec = time.perf_counter() - t_train

    # -- 5. measure LoRA delta, merge adapter, eval post --
    with torch_.no_grad():
        lora_delta_norm = 0.0
        for mod in vla.llm_backbone.modules():
            lora_A = getattr(mod, "lora_A", None)
            lora_B = getattr(mod, "lora_B", None)
            if lora_A and "default" in lora_A and "default" in lora_B:
                try:
                    A = lora_A["default"].weight.detach().float()
                    B = lora_B["default"].weight.detach().float()
                    lora_delta_norm = max(lora_delta_norm,
                                          float(torch_.linalg.matrix_norm(
                                              B @ A, ord=2).item()))
                except Exception:
                    pass
    vla.llm_backbone.eval()
    vla.llm_backbone = vla.llm_backbone.merge_and_unload()
    torch_.cuda.empty_cache()
    print("[N9] eval POST (repaired, merged) on frozen prefix...", flush=True)
    ev_post = _prefix_eval(eval_batches, "eval-post")
    acc_post = ev_post["acc"]
    delta = acc_post - acc_pre
    out.append(f"| post (repaired) | {acc_post:.4f} | delta {delta:+.4f} vs "
               f"collapsed; {steps} steps in {train_sec:.0f}s |")
    out.append(f"| vs N1 FT | {acc_post - n1_ft:+.4f} | n1_ft={n1_ft:.4f} |")
    rec.metric(event="eval_post", acc=round(acc_post, 4),
               correct=ev_post["correct"], total=ev_post["total"],
               scope=ev_post["scope"],
               delta_repair=round(float(delta), 4))

    # -- 6. certificate bookkeeping --
    out.append(f"| certificate | sound {n_sound}/{n_compressed} at "
               f"compression | max LoRA delta spectral norm "
               f"{lora_delta_norm:.4f}; deployed ball re-derives from the "
               f"saved merged weights (one SVD/layer) |")
    rec.metric(event="certify", sound_layers=n_sound,
               n_compressed=n_compressed,
               lora_delta_spectral_max=round(lora_delta_norm, 6))

    # -- save the repaired (merged) handoff for a full-protocol confirm --
    save_dir = Path("results") / "N9-repaired"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch_.save({"llm_backbone": {k_: v_.detach().cpu()
                                  for k_, v_ in
                                  vla.llm_backbone.state_dict().items()},
                 "meta": {"frac": frac, "rrank": rrank, "stat": stat,
                          "steps": steps, "lora_r": lora_r, "seed": seed,
                          "acc_pre": acc_pre, "acc_post": acc_post,
                          "n1_ft": n1_ft, "ratio": ratio,
                          "eval_scope": ev_post["scope"]}},
                save_dir / f"repaired_seed{seed}.pt")
    print(f"[N9] saved repaired handoff: "
          f"{save_dir / f'repaired_seed{seed}.pt'}", flush=True)

    finish_run(rec, status="completed",
               results={"seed": seed, "frac": frac, "residual_rank_cap": rrank,
                        "calib_stat": stat, "ratio": round(float(ratio), 3),
                        "acc_pre": float(acc_pre), "acc_post": float(acc_post),
                        "delta_repair": round(float(delta), 4),
                        "n1_ft": n1_ft,
                        "delta_vs_n1_ft": round(float(acc_post - n1_ft), 4),
                        "eval_scope": ev_post["scope"],
                        "cert_sound_layers": n_sound,
                        "n_layers_compressed": n_compressed,
                        "n_layers_kept_full": n_kept,
                        "steps": steps, "lora_r": lora_r, "lr": lr,
                        "train_wall_sec": round(float(train_sec), 1),
                        "compress_wall_sec": round(float(comp_sec), 1),
                        "lora_delta_spectral_max": round(lora_delta_norm, 6)})

    del harness
    torch_.cuda.empty_cache()
    out.append("")
    out.append("* N9 acc_pre/acc_post are FIXED-PREFIX evidence "
               f"({eval_batches} frozen-split batches). A reported final "
               "number requires the full 6496-batch protocol on the saved "
               "repaired checkpoint (results/N9-repaired/). Optimization "
               "saw train episodes only (frozen split).")


# ============================================================================
# N9-confirm — full-protocol confirmation of the repaired checkpoint
# ============================================================================
# Why a separate stage: _run_repair reports PREFIX evidence (600 batches).
# The reportable number is the full 6496-batch frozen protocol on the SAVED
# merged weights, PLUS certificates re-derived FROM those merged weights
# (the compression-time core bounds do not cover the LoRA merge), PLUS an
# honest ratio that counts the adapter params. No training here — eval only.
def _run_repair_confirm(out: list[str], ctx=None) -> None:
    import gc as _gc
    import math as _math

    import torch

    from .compress import _wilson
    from .n2_sweep import _llm_linear_keys, _N2EvalHarness

    seeds = (ctx.seeds if ctx and ctx.seeds else [ctx.seed if ctx else 0])
    ft_dir = (Path(_os.environ.get("QICERT_FT_CKPT", ""))
              if _os.environ.get("QICERT_FT_CKPT")
              else (Path(ctx.save_ckpt) if ctx and ctx.save_ckpt
                    else Path("weights") / "ckpt" / "finetuned"))
    lora_r = int(_os.environ.get("QICERT_N9_LORA_R", "8"))
    # Ratio of the compression recipe the repair started from (recorded in
    # the N9 run; adapter cost added below for the honest figure).
    base_ratio = float(_os.environ.get("QICERT_N9_BASE_RATIO", "2.536"))
    targets = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")

    out += table_header(
        "N9 repair-confirm (full 6496-batch protocol on saved merged "
        f"weights, re-derived dense certs, seeds={seeds})",
        ["Seed", "Ratio", "Eval acc", "Delta vs N1 FT", "Cert",
         "Status"])

    for seed in seeds:
        if ctx is not None:
            ctx.seed = seed
        rep_default = (Path("results") / "N9-repaired"
                       / f"repaired_seed{seed}.pt")
        rep_path = Path(_os.environ.get("QICERT_N9_REPAIRED", str(rep_default)))
        ft_path = ft_dir / f"seed{seed}.pt"
        if not rep_path.exists():
            out.append(f"| {seed} | - | - | - | - | FAIL (no {rep_path}) |")
            continue
        if not ft_path.exists():
            out.append(f"| {seed} | - | - | - | - | FAIL (no FT {ft_path}) |")
            continue
        rec = start_run(
            ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "N9C"),
            label=f"repair-confirm-seed{seed}",
            config={"experiment": "N9 repair full-protocol confirm",
                    "seed": seed, "repaired_ckpt": str(rep_path),
                    "ft_ckpt": str(ft_path), "lora_r": lora_r,
                    "base_ratio": base_ratio})
        saved = torch.load(str(rep_path), map_location="cpu",
                           weights_only=True)
        repaired = saved["llm_backbone"]
        meta = saved.get("meta", {})
        ft_llm = torch.load(str(ft_path), map_location="cpu",
                            weights_only=True)
        merged = ft_llm["llm_backbone"]

        harness = _N2EvalHarness(seed, ft_dir, ctx)
        harness.adopt_reference(merged)
        n1_ft = harness.n1_ft
        print(f"  [N9-confirm seed={seed}] full-split eval of repaired "
              f"weights...", flush=True)
        t0 = time.perf_counter()
        acc = harness.eval(repaired)
        info = harness._last_eval
        eval_sec = time.perf_counter() - t0
        ci = _wilson(info["correct"], info["total"]) if info["total"] else None
        delta = acc - n1_ft if n1_ft is not None else float("nan")

        # -- re-derived certificates: exact dense spectral norm per layer --
        print(f"  [N9-confirm seed={seed}] re-deriving dense certs "
              f"(SVD/layer)...", flush=True)
        inv = _llm_linear_keys(repaired)
        norms = {}
        for _lt, key in inv:
            W = repaired[key].detach().float().cpu().numpy().astype(np.float64)
            norms[key] = float(np.linalg.norm(W, 2))
        log10L = float(sum(_math.log10(v) for v in norms.values() if v > 0))
        del repaired, merged, ft_llm, saved
        _gc.collect()

        # -- honest ratio: cores+residual (base_ratio) + LoRA adapter ------
        # Shapes re-read from the repaired file (keys alone carry no dims).
        dense_total = 0
        adapter_total = 0
        saved2 = torch.load(str(rep_path), map_location="cpu",
                            weights_only=True)["llm_backbone"]
        for layer_type, key in _llm_linear_keys(saved2):
            t = saved2[key]
            M, N = (t.shape[0], t.shape[1]) if len(t.shape) == 2 else (0, 0)
            dense_total += M * N
            suffix = layer_type.rsplit("-", 1)[-1]
            if suffix in targets and M and N:
                adapter_total += lora_r * (M + N)
        del saved2
        _gc.collect()
        adapter_frac = adapter_total / max(dense_total, 1)
        ratio_honest = 1.0 / (1.0 / base_ratio + adapter_frac)
        go = (ratio_honest >= 2.0 and acc >= 0.4468 - 0.05)
        status = "GO" if go else "NO-GO"
        out.append(f"| {seed} | {ratio_honest:.3f}x | {acc:.4f} "
                   f"{ci} | {delta:+.4f} | log10L={log10L:.1f} "
                   f"({len(norms)}/{len(inv)} exact) | {status} |")
        print(f"  [N9-confirm seed={seed}] DONE acc={acc:.4f} "
              f"ratio_honest={ratio_honest:.3f}x {status} ({eval_sec:.0f}s)",
              flush=True)
        finish_run(rec, status="completed",
                   results={"seed": seed, "ratio_base": base_ratio,
                            "ratio_honest": round(float(ratio_honest), 4),
                            "adapter_frac": round(float(adapter_frac), 6),
                            "eval_acc": float(acc), "eval_correct": info["correct"],
                            "eval_total": info["total"], "eval_ci": ci,
                            "eval_scope": info["scope"],
                            "n1_ft": n1_ft,
                            "delta_vs_n1_ft": round(float(delta), 4),
                            "cert_log10L": round(float(log10L), 3),
                            "cert_layers": len(norms),
                            "gate": status,
                            "bar": "ratio>=2.0 and acc>=0.3968"},
                   kill_criterion="N9-GO: ratio>=2x and acc>=FT-0.05",
                   kill_verdict=status)
        del harness
        torch.cuda.empty_cache()
    out.append("")
    out.append("* N9-confirm is the REPORTABLE row (full-split + re-derived "
               "dense certs + adapter-counted ratio). Prefix N9 numbers stay "
               "search-phase evidence.")
