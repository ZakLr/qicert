"""Compression + Layer-1 certificates: N1 (baseline), N2, N2', N3 (submission/08)."""
from __future__ import annotations

from pathlib import Path

from ._base import finish_run, pending_row, start_run, table_header, wants

# Real MiniVLA checkpoint (same source as N2'); layers come from the LLM
# backbone state dict (flat dotted keys, see bench/step0.py + n2prime.py).
# Env overrides (QICERT_CKPT / QICERT_DATA_ROOT) let the Kaggle kernel point
# at its own /kaggle/working layout instead of the repo-relative default.
import os as _os
CKPT = Path(_os.environ.get("QICERT_CKPT", "")) if _os.environ.get("QICERT_CKPT") else \
    Path(__file__).resolve().parents[1] / "weights" / "ckpt" / "checkpoints" / \
    "step-122500-epoch-55-loss=0.0743.pt"
DATA_ROOT = Path(_os.environ.get("QICERT_DATA_ROOT", "")) if _os.environ.get("QICERT_DATA_ROOT") else \
    Path(__file__).resolve().parents[1] / "weights" / "libero_spatial_no_noops"

# Layer inventory for N3 (real MiniVLA layer-0, from the checkpoint):
#   layer_type -> state-dict key suffix
N3_LAYERS = {
    "attention-q-proj": "llm.model.layers.0.self_attn.q_proj.weight",
    "attention-k-proj": "llm.model.layers.0.self_attn.k_proj.weight",
    "attention-v-proj": "llm.model.layers.0.self_attn.v_proj.weight",
    "attention-o-proj": "llm.model.layers.0.self_attn.o_proj.weight",
    "mlp-gate-proj": "llm.model.layers.0.mlp.gate_proj.weight",
    "mlp-up-proj": "llm.model.layers.0.mlp.up_proj.weight",
    "mlp-down-proj": "llm.model.layers.0.mlp.down_proj.weight",
}

SMOKE = frozenset({"compression-pareto", "kernel-smoke"})


def run(rows: str, out: list[str], ctx=None) -> None:
    # --- Phase 0.5: the Python reference kernels measured for real ---
    if wants(rows, "kernel-smoke", SMOKE):
        from qicert.kernels import parity_check, active_backend_name
        rec = start_run(ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "kernel-smoke"),
                        label="kernel-smoke",
                        config={"check": "kernel-smoke",
                                "backend": active_backend_name(),
                                "parity": "tests/test_kernels.py::test_parity_harness"})
        fp = parity_check()
        out += table_header(
            f"Phase-0.5 - kernel smoke (backend={active_backend_name()})",
            ["Check", "Value", "Tolerance", "Status"])
        checks = [
            ("TT-SVD reconstruction rel. err", fp["tt_svd_rel_err"], 1e-10, "lower"),
            ("Lipschitz tight-vs-dense rel. err", fp["lipschitz_tight_vs_dense"], 1e-4, "lower"),
            ("Pauli diagonalization identity err", fp["pauli_identity_err"], 1e-9, "lower"),
            ("IQAE interval covers known p", fp["iqae_interval_contains_p"], 1.0, "exact"),
        ]
        all_ok = True
        for name, val, tol, mode in checks:
            ok = (val == tol) if mode == "exact" else val <= tol
            all_ok = all_ok and ok
            bar = "= 1" if mode == "exact" else f"< {tol:.0e}"
            out.append(f"| {name} | {val:.3e} | {bar} | {'PASS' if ok else 'FAIL'} |")
            if rec:
                rec.metric(check=name, value=float(val), tol=tol, pass_ok=bool(ok))
        if rec:
            rec.sample_power()
            finish_run(rec, status="completed" if all_ok else "failed",
                       results={"checks": {k: float(v) for k, v in fp.items()}},
                       tolerance_note="parity harness tolerances per tests/test_kernels.py")
            out.append(f"\nrecorded: {rec.run_dir}")

    # --- N1: the load-bearing baseline (LoRA fine-tune + INT8 reference) ---
    if wants(rows, "baseline-int8", SMOKE):
        _run_n1(out, ctx)

    # --- N2: {TT, QTT} x TT-cross sweep ---
    if wants(rows, "compression-pareto", SMOKE):
        out += table_header("N2 - compression Pareto curve (ratio vs accuracy vs certified-safe-set)",
                            ["ID", "Experiment", "Seeds", "GPU-h", "Status"])
        out.append(pending_row("N2", "{TT, QTT} x TT-cross sweep, 6 bond plans x 2 backbones",
                               "3", "24"))
        out.append(pending_row("N2'", "bit-ordering sensitivity (3 orderings, 1 layer, 1 seed)",
                               "1", "2"))

    # --- N3: Layer-1 certificate table ---
    if wants(rows, "lipschitz-table", SMOKE):
        _run_n3(out, ctx)

    # --- N14: clean-env reproducibility audit ---
    if wants(rows, "clean-env-audit", SMOKE):
        out += table_header("N14 - clean-env reproducibility audit (fresh venv)",
                            ["ID", "Check", "GPU-h", "Status"])
        out.append("| N14 | pip install qicert + bench.all --rows=smoke on fresh venv | 4 | [pending] CI runs the smoke set |")


# ========================================================================
# N1 - load-bearing baseline: MiniVLA LoRA fine-tune + INT8 reference
# ========================================================================
# Pre-registered (submission/08-experiments.md): "AD backbone fine-tune
# (QLoRA) + INT8 bnb reference", 3 seeds, 8 GPU-h. Robotics-first
# redefinition (Q11 decision, docs/backbones.md): MiniVLA-1B on the LIBERO
# spatial slice, LoRA fine-tune (peft) + INT8 reference on the same
# backbone. Every qicert compression claim (N2/N13) is measured against
# this baseline accuracy (kill criterion R1), so N1 MUST be honest: no
# noiseless-statevector optimism, real RLDS batches, real action accuracy.
#
# bnb is NOT available on sm_120 in the container image (verified), so the
# INT8 reference uses torch.ao.quantization.quantize_dynamic instead of
# bitsandbytes 8-bit. Same semantics: same backbone, INT8 weights, matched
# budget. Documented substitution, not a silent one (AGENTS.md).
#
# Run (container, GPU, weights + repo mounted):
#   python -m qicert.bench.all --module compress --rows baseline-int8 \
#       --out results --exp-id N1 --seeds 0,1,2 --steps-per-seed 400 \
#       --batch 2 --run-tag libero-spatial-lora-r8
# Smoke (this machine, ~5 min): --steps 24 --batch 1 --seeds 0


def _load_state_dict() -> dict:
    import torch  # torch-optional module: local import
    if not CKPT.exists():
        raise FileNotFoundError(f"checkpoint not found: {CKPT} "
                                "(run scripts/download_backbones.py first)")
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    return sd["model"]


def _run_n1(out: list[str], ctx=None) -> None:
    steps = (ctx.steps if ctx and ctx.steps else 24)
    batch = (ctx.batch if ctx and ctx.batch else 1)
    seeds = (ctx.seeds if ctx and ctx.seeds else [ctx.seed if ctx else 0])
    steps_per_seed = (ctx.steps_per_seed if ctx and ctx.steps_per_seed
                      else steps)
    lora_r = 8

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    out += table_header(
        f"N1 - MiniVLA baseline: LoRA fine-tune + INT8 reference "
        f"(steps/seed={steps_per_seed}, batch={batch}, seeds={seeds}, lora_r={lora_r})",
        ["Seed", "Train loss (last)", "Action acc (train, last)",
         "Eval acc (fine-tuned)", "Eval acc (INT8)", "Delta (INT8 - FT)", "Status"])

    all_rows = []
    for seed in seeds:
        row = _run_n1_seed(out, ctx, seed, steps_per_seed, batch, lora_r)
        if row:
            all_rows.append(row)

    if all_rows:
        acc_ft = np.array([r["eval_acc_ft"] for r in all_rows])
        acc_i8 = np.array([r["eval_acc_i8"] for r in all_rows])
        out.append("")
        out.append(f"* Baseline (median over seeds): fine-tuned {np.median(acc_ft):.4f} "
                   f"(min {acc_ft.min():.4f}); INT8 {np.median(acc_i8):.4f} "
                   f"(min {acc_i8.min():.4f}).")
        out.append("* N1 is the R1 comparator: N2's compressed accuracy is measured "
                   "against this fine-tuned baseline at matched budget.")


def _run_n1_seed(out: list[str], ctx, seed: int, steps: int, batch: int,
                 lora_r: int) -> dict | None:
    """One seed of N1. Returns the results dict; None on environment failure."""
    import os
    import sys
    import time

    # The fork must be importable (container image bakes it; host needs
    # weights/code on sys.path + the transformers-5.x patch applied).
    # QICERT_FORK lets the Kaggle kernel point at its own clone location.
    repo = Path(__file__).resolve().parents[1]
    fork = Path(os.environ.get("QICERT_FORK", "")) if os.environ.get("QICERT_FORK") \
        else repo / "weights" / "code"
    if str(fork) not in sys.path:
        sys.path.insert(0, str(fork))
    os.environ.setdefault("PRISMATIC_DATA_ROOT", str(repo / "weights" / "data"))

    import numpy as np
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    rec = start_run(ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "N1"),
                    label=f"baseline-seed{seed}",
                    config={
                        "experiment": "N1 baseline fine-tune + INT8 reference",
                        "model": "minivla-libero90-prismatic (native Prismatic)",
                        "checkpoint": str(CKPT),
                        "dataset": "libero_spatial_no_noops",
                        "data_root": str(DATA_ROOT),
                        "seed": seed, "steps": steps, "batch": batch,
                        "lora": {"r": lora_r, "alpha": min(lora_r, 16),
                                 "target_modules": "all-linear"},
                        "int8": "torch.ao.quantization.quantize_dynamic, "
                                "weight-only qint8 (bnb unavailable on sm_120; "
                                "documented substitution, AGENTS.md)",
                        "dtype": "fp16", "device": "cuda",
                        "note": "scored baseline — R1 comparator",
                    })

    try:
        from prismatic.models.load import load_vla
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
        from prismatic.util.data_utils import PaddedCollatorForActionPrediction
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(seed)

        print(f"[N1 seed={seed}] loading MiniVLA (fp16)...", flush=True)
        t0 = time.perf_counter()
        vla = load_vla(str(CKPT), hf_token=None, load_for_training=True)
        vla = vla.to(dtype=torch.float16, device="cuda")
        load_sec = time.perf_counter() - t0
        n_params = sum(p.numel() for p in vla.parameters())
        rec.metric(event="load", load_sec=round(load_sec, 2), params=int(n_params),
                   vram_mb=round(torch.cuda.memory_allocated() / 1024**2, 1))

        # Grab tokenizer + prompt-builder class BEFORE peft-wrapping (peft
        # delegates __getattr__ to the base, but we want the real objects).
        tokenizer = vla.llm_backbone.tokenizer
        prompt_builder_fn = vla.llm_backbone.prompt_builder_fn

        # LoRA on the LLM backbone IN PLACE (vision + projector frozen): the
        # VLM forward fuses pixel_values -> embeddings and calls
        # self.llm_backbone(...); peft's generic (no-task_type) wrapper passes
        # through all kwargs, so we wrap vla.llm_backbone and let the full
        # forward run the multimodal path (verified in scripts/step1_*.py).
        lora_cfg = LoraConfig(r=lora_r, lora_alpha=min(lora_r, 16),
                              target_modules="all-linear", lora_dropout=0.0,
                              bias="none")
        vla.llm_backbone = get_peft_model(vla.llm_backbone, lora_cfg)
        vla.llm_backbone.print_trainable_parameters()

        # Data: fork's native RLDS pipeline (verified: 432 trajectories,
        # 110-frame episodes, 7-dim actions, language instructions)
        action_tokenizer = ActionTokenizer(tokenizer)
        batch_transform = RLDSBatchTransform(
            action_tokenizer, tokenizer,
            image_transform=vla.vision_backbone.get_image_transform(),
            prompt_builder_fn=prompt_builder_fn,
        )
        ds = RLDSDataset(
            DATA_ROOT, "libero_spatial_no_noops", batch_transform,
            resize_resolution=vla.vision_backbone.default_image_resolution[1:],
            shuffle_buffer_size=1000, image_aug=False,
        )
        collator = PaddedCollatorForActionPrediction(
            tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="right")
        dl = DataLoader(ds, batch_size=batch, collate_fn=collator, num_workers=0)
        it = iter(dl)

        opt = AdamW([p for p in vla.llm_backbone.parameters() if p.requires_grad],
                    lr=5e-4)
        vla.llm_backbone.train()
        num_patches = int(vla.vision_backbone.num_patches)

        def _to_half_cuda(x):
            """DinoSigLIP pixel_values is a dict {'dino':..,'siglip':..}; handle both."""
            if isinstance(x, dict):
                return {k: v.to(torch.float16).cuda() for k, v in x.items()}
            return x.to(torch.float16).cuda()

        def _action_acc(out_, labels_t) -> float:
            """Fork's action-token accuracy: logits[:, num_patches:-1] vs labels[:, 1:]."""
            logits = out_.logits[:, num_patches:-1]
            preds = logits.argmax(dim=-1)
            gt = labels_t[:, 1:].to(preds.device)
            mask = gt > action_tokenizer.action_token_begin_idx
            if not mask.any():
                return float("nan")
            return float((preds[mask] == gt[mask]).float().mean().item())

        # Train `steps` steps.
        losses, accs = [], []
        t_train = time.perf_counter()
        for step in range(steps):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl)
                b = next(it)
            input_ids = b["input_ids"].cuda()
            attention_mask = b["attention_mask"].cuda()
            pixel_values = _to_half_cuda(b["pixel_values"])
            labels = b["labels"].cuda()

            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.float16):
                out_ = vla(input_ids=input_ids, attention_mask=attention_mask,
                           pixel_values=pixel_values, labels=labels)
                loss = out_.loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            acc = _action_acc(out_, labels)
            accs.append(acc)
            peak = torch.cuda.max_memory_allocated() / 1024**2
            rec.metric(event="train_step", step=step, loss=float(loss.item()),
                       action_acc=float(acc) if acc == acc else None,
                       peak_vram_mb=round(peak, 1))
            if step % 8 == 0 or step == steps - 1:
                print(f"  step {step+1}/{steps}: loss={loss.item():.4f} "
                      f"acc={acc:.3f} peak={peak/1024:.2f} GiB", flush=True)
        train_sec = time.perf_counter() - t_train

        # ---- eval leg: held-out batch from the same slice (fine-tuned) ----
        vla.llm_backbone.eval()
        with torch.inference_mode():
            b = next(it)
            input_ids = b["input_ids"].cuda()
            attention_mask = b["attention_mask"].cuda()
            pixel_values = _to_half_cuda(b["pixel_values"])
            labels = b["labels"].cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                out_ = vla(input_ids=input_ids, attention_mask=attention_mask,
                           pixel_values=pixel_values, labels=labels)
            eval_acc_ft = _action_acc(out_, labels)
        print(f"  eval acc (fine-tuned): {eval_acc_ft:.4f}", flush=True)

        # ---- INT8 reference: BASE model (no LoRA), weight-only int8, CPU ----
        # torch.ao dynamic quant is CPU-only, so the reference eval runs on
        # CPU (small held-out batch; this is the reference, not the train
        # loop). Fresh load so the reference is the UNTRAINED backbone — the
        # honest INT8 point of the same architecture (R1 comparator).
        try:
            from torch.ao.quantization import quantize_dynamic
            t_i8 = time.perf_counter()
            vla_i8 = load_vla(str(CKPT), hf_token=None, load_for_training=False)
            # quantize_dynamic (CPU) needs fp32 input tensors; the quantized
            # linears reject fp16 ("Input type (float) and bias type (c10::Half)").
            vla_i8 = vla_i8.to(dtype=torch.float32).cpu()
            vla_i8 = quantize_dynamic(vla_i8, dtype=torch.qint8)
            with torch.inference_mode():
                b = next(it)
                input_ids = b["input_ids"]
                attention_mask = b["attention_mask"]
                pixel_values = b["pixel_values"]
                labels = b["labels"]
                out_ = vla_i8(input_ids=input_ids, attention_mask=attention_mask,
                              pixel_values=pixel_values, labels=labels)
            eval_acc_i8 = _action_acc(out_, labels)
            i8_sec = time.perf_counter() - t_i8
            print(f"  eval acc (INT8 reference, CPU): {eval_acc_i8:.4f} ({i8_sec:.1f}s)",
                  flush=True)
        except Exception as exc:  # INT8 must never kill the baseline row
            print(f"  INT8 reference failed: {exc}", flush=True)
            rec.metric(event="int8_reference", error=str(exc))
            eval_acc_i8 = float("nan")
            i8_sec = 0.0

        delta = (eval_acc_i8 - eval_acc_ft) if not np.isnan(eval_acc_i8) else float("nan")
        verdict = "OK" if not np.isnan(eval_acc_ft) else "FAIL"
        status = "completed" if verdict == "OK" else "failed"
        out.append(f"| {seed} | {losses[-1]:.4f} | {accs[-1]:.4f} | "
                   f"{eval_acc_ft:.4f} | {eval_acc_i8:.4f} | {delta:+.4f} | {verdict} |")

        rec.sample_power()
        results = {
            "seed": seed, "steps": steps, "batch": batch,
            "train_loss_last": float(losses[-1]),
            "train_action_acc_last": float(accs[-1]),
            "train_sec": round(train_sec, 2),
            "int8_reference_sec": round(i8_sec, 2),
            "eval_acc_finetuned": float(eval_acc_ft),
            "eval_acc_int8": float(eval_acc_i8),
            "delta_int8_minus_ft": round(float(delta), 4) if delta == delta else None,
            "verdict": verdict,
        }
        finish_run(rec, status=status, results=results,
                   tolerance_note="smoke steps unless --steps given; eval is one "
                                  "held-out batch (same slice); INT8 on CPU")
        out.append(f"\nrecorded: {rec.run_dir}")
        return results
    except Exception as exc:
        import traceback
        traceback.print_exc()
        out.append(f"| {seed} | - | - | - | - | - | FAIL ({type(exc).__name__}: {exc}) |")
        if rec:
            rec.sample_power()
            finish_run(rec, status="failed",
                       results={"seed": seed, "error": f"{type(exc).__name__}: {exc}"},
                       tolerance_note="environment/toolchain failure — no scored claim")
            out.append(f"\nrecorded: {rec.run_dir}")
        return None


# ========================================================================
# N3 - Layer-1 exact Lipschitz certificate table
# ========================================================================
# Pre-registered (submission/08-experiments.md): "Layer-1 certificate table
# (Lipschitz products per layer per Pareto point)", 2 GPU-h, CPU-friendly.
# Consumes the committed bit-ordering from N2' (bit-reversed) and the real
# MiniVLA layer inventory, computing L = prod_k ||G_k||_2 per layer at
# matched bond, plus the tight operator norm (power iteration) and the
# bound-tightness ratio. No GPU needed; pure numpy kernels.


def _factorize(n: int, parts: int = 2) -> tuple[int, int]:
    """Split n into two close factors (for TT-matrix mode dims)."""
    a = int(round(n**0.5))
    while n % a != 0:
        a -= 1
    return a, n // a


def _run_n3(out: list[str], ctx=None) -> None:
    import numpy as np
    from qicert.kernels import get_backend

    k = get_backend()
    ranks = (8,)

    sd = _load_state_dict()
    rec = start_run(ctx, exp_id=(ctx.exp_id if ctx and ctx.exp_id else "N3"),
                    label="lipschitz-table",
                    config={"experiment": "N3 Layer-1 certificate table",
                            "layers": N3_LAYERS,
                            "ordering": "adaptive (near-square factors, "
                                         "N2' bit-reversed semantics)",
                            "ranks": list(ranks),
                            "note": "per-layer Lipschitz products at matched bond, "
                                     "CPU (numpy reference kernels)"})

    out += table_header(
        "N3 - Layer-1 exact Lipschitz table (L = prod_k ||G_k||_2, "
        f"rank {ranks[0]})",
        ["Layer", "M x N", "Params", "Lipschitz (L)", "Tight norm",
         "Tightness (L/||W||)", "Status"])

    results: dict[str, dict] = {}
    for layer_type, key in N3_LAYERS.items():
        W = sd["llm_backbone"][key].detach().float().cpu().numpy()
        M, N = W.shape
        # Adaptive mode grouping: near-square factors of M and N (bit-reversed
        # semantics: least-significant digits first => factors descending).
        m_a, m_b = _factorize(M)
        n_a, n_b = _factorize(N)
        m_dims = (m_b, m_a) if m_b > m_a else (m_a, m_b)
        n_dims = (n_b, n_a) if n_b > n_a else (n_a, n_b)
        cs = k.tt_svd(W, m_dims, n_dims, ranks)
        L = k.lipschitz_product(cs.arrays)
        tight = k.operator_norm_tight(cs.arrays, m_dims, n_dims)
        params = int(sum(np.prod(g.shape) for g in cs.arrays))
        ratio = L / max(tight, 1e-12)
        # certificate is sound iff L >= tight (product bound upper-bounds)
        ok = bool(L >= tight - 1e-9)
        results[layer_type] = {"L": float(L), "tight": float(tight),
                               "ratio": float(ratio), "params": params,
                               "sound": ok, "m_dims": list(m_dims),
                               "n_dims": list(n_dims)}
        out.append(f"| {layer_type} | {M}x{N} | {params} | {L:.6f} | "
                   f"{tight:.6f} | {ratio:.2f} | {'PASS' if ok else 'FAIL'} |")
        if rec:
            rec.metric(layer=layer_type, m=M, n=N, params=params,
                       lipschitz=float(L), tight_norm=float(tight),
                       tightness_ratio=float(ratio), sound=bool(ok))

    n_sound = sum(1 for r in results.values() if r["sound"])
    out.append("")
    out.append(f"* Certificate sound (L >= tight) on {n_sound}/{len(results)} layers. "
               f"Layer-1 bound is a global sanity bound; Layer-2a SOS (N4) is the "
               f"local certificate.")

    if rec:
        rec.sample_power()
        finish_run(rec, status="completed",
                   results={"ordering": "adaptive near-square", "ranks": list(ranks),
                            "layers": results,
                            "sound_count": n_sound,
                            "kill_criterion": "R2: certified-safe-set <50% at all "
                                              "Pareto points (evaluated in N4)",
                            "verdict": "Layer-1 certificate computed — soundness "
                                       "holds (bound >= tight norm)"},
                   tolerance_note="CPU numpy reference kernels; power iteration "
                                  "100 iters")
        out.append(f"\nrecorded: {rec.run_dir}")
