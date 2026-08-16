"""Step 1 — MiniVLA-1B robotics smoke on the RTX 5060 (8 GB, sm_120).

Warm-up ladder (docs/backbones.md §6, step 2): prove the *toolchain* end to
end on the real scored backbone before any N1/N2/N3 GPU-h is spent:

    1. load `minivla-libero90-prismatic` in fp16 via the fork's native
       Prismatic loader (`load_vla`) — the ONLY supported MiniVLA format
       (no transformers-converted MiniVLA exists; verified 2026-08-16)
    2. one LIBERO-style prompt -> action decode (`predict_action`)
    3. one tiny LoRA step (peft, r=8, one forward/backward/optimizer step)

Kill criterion (backbones.md §3): if fp16 doesn't fit at batch 1 with
gradient checkpointing, the robotics-first plan needs revision (it will fit).

This is a TOOLCHAIN smoke, not a scored run (AGENTS.md: noiseless/quick runs
are proofs of concept, not evidence of anything). All numbers are recorded
with qicert.record.RunRecorder — VRAM peak, per-step latency, power/energy,
config hash, git commit — so nothing needs re-running for missing data.

Run (container, repo + weights mounted):
    python scripts/step1_minivla_smoke.py --out /workspace/qicert/results \\
        --exp-id STEP1 --run-tag minivla-smoke

Requires the Prismatic fork at weights/code with the transformers-5.x patch
applied (scripts/patches/prismatic-transformers5.patch) and its pure-python
deps installed (see Dockerfile).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# The fork lives in weights/code (host-mounted); put it on sys.path BEFORE
# importing prismatic so we use the mounted (patched) code, not any baked copy.
REPO = Path(__file__).resolve().parents[1]
FORK = REPO / "weights" / "code"
if str(FORK) not in sys.path:
    sys.path.insert(0, str(FORK))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from qicert.record import RunRecorder  # noqa: E402

CKPT = REPO / "weights" / "ckpt" / "checkpoints" / "step-122500-epoch-55-loss=0.0743.pt"
INSTRUCTION = "pick up the red block and place it on the plate"
SYNTH_IMG = (224, 224)


def _dummy_image(seed: int = 0) -> "object":
    """Deterministic synthetic image (toolchain smoke — not a real scene)."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = (rng.random((SYNTH_IMG[0], SYNTH_IMG[1], 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _apply_lora_step(vla, rec, seed: int = 0) -> dict:
    """One peft LoRA step on the LLM backbone (r=8, lora_alpha=16)."""
    from peft import LoraConfig, get_peft_model

    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    cfg = LoraConfig(
        r=8, lora_alpha=16, target_modules=targets,
        lora_dropout=0.0, bias="none")
    # no task_type: the Prismatic LLM wrapper is a plain nn.Module without
    # prepare_inputs_for_generation; we only run one fwd/bwd step here.
    model = get_peft_model(vla.llm_backbone, cfg)

    # one synthetic training step: random token ids + labels
    tokenizer = vla.llm_backbone.tokenizer
    prompt = vla.get_prompt_builder()
    prompt.add_turn(role="human",
                    message=f"What action should the robot take to {INSTRUCTION.lower()}?")
    text = prompt.get_prompt()
    ids = tokenizer(text, truncation=True, return_tensors="pt").input_ids.to(vla.device)
    labels = ids.clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = model(input_ids=ids, labels=labels)
    loss = out.loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    dt = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    grad_norm = sum(
        float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
    rec.metric(event="lora_step", loss=float(loss.item()),
               grad_norm=float(grad_norm), step_sec=round(dt, 4),
               peak_vram_mb=round(peak_mb, 1))
    return {"lora_loss": float(loss.item()), "lora_grad_norm": float(grad_norm),
            "lora_step_sec": round(dt, 4), "lora_peak_vram_mb": round(peak_mb, 1),
            "lora_config": {"r": 8, "lora_alpha": 16, "target_modules": targets}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="results root (recorder)")
    ap.add_argument("--exp-id", default="STEP1")
    ap.add_argument("--run-tag", default="minivla-smoke")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert CKPT.exists(), f"checkpoint missing: {CKPT} (run scripts/download_backbones.py)"

    # The Prismatic fork needs PRISMATIC_DATA_ROOT at import (conf/datasets.py
    # reads it as a dataclass default); it is unused by the load path.
    import os
    os.environ.setdefault("PRISMATIC_DATA_ROOT", str(REPO / "weights" / "data"))

    # Sanity: the transformers-5.x patch must be present in the mounted fork.
    # (Applied once via `git apply scripts/patches/prismatic-transformers5.patch`
    # after checkout; see scripts/download_backbones.py + docs/backbones.md.)
    sentinel = FORK / "prismatic" / "models" / "backbones" / "llm" / ".qicert_patched"
    if not sentinel.exists():
        import subprocess
        patch = REPO / "scripts" / "patches" / "prismatic-transformers5.patch"
        if subprocess.run(["git", "-C", str(FORK), "apply", "--check", str(patch)],
                          capture_output=True).returncode == 0:
            subprocess.run(["git", "-C", str(FORK), "apply", str(patch)], check=True)
            sentinel.write_text("applied by qicert step1 smoke\n")
        else:
            print("error: Prismatic fork missing the transformers-5.x patch "
                  "(weights/code) — run:", file=sys.stderr)
            print(f"  git -C {FORK} apply scripts/patches/prismatic-transformers5.patch",
                  file=sys.stderr)
            return 2

    config = {
        "checkpoint": str(CKPT), "instruction": INSTRUCTION,
        "dtype": "fp16", "device": "cuda",
        "image": {"size": list(SYNTH_IMG), "source": "synthetic (toolchain)"},
        "lora": {"r": 8, "lora_alpha": 16, "target_modules": ["q/k/v/o_proj"]},
        "note": "TOOLCHAIN smoke — not a scored claim (AGENTS.md)",
    }
    rec = RunRecorder(args.exp_id, args.seed, config, out_root=args.out or "results",
                      run_tag=args.run_tag)

    from prismatic.models.load import load_vla
    print("loading MiniVLA (native Prismatic, fp16)...", flush=True)
    t_load = time.perf_counter()
    vla = load_vla(str(CKPT), hf_token=None, load_for_training=True)
    vla = vla.to(dtype=torch.float16, device="cuda")
    t_load = time.perf_counter() - t_load
    n_params = sum(p.numel() for p in vla.parameters())
    vram_mb = torch.cuda.memory_allocated() / 1024**2
    rec.metric(event="load", load_sec=round(t_load, 2),
               params=int(n_params), vram_mb=round(vram_mb, 1))
    print(f"  loaded {n_params/1e6:.1f}M params in {t_load:.1f}s, "
          f"{vram_mb/1024:.1f} GiB VRAM", flush=True)

    # transformers>=5 generate() calls _optimize_model_for_decode() which
    # assumes a transformers PreTrainedModel (get_experts_implementation); the
    # Prismatic VLM is a plain nn.Module. Bypass that decode path — dropout is
    # already off in eval/inference mode.
    vla._optimize_model_for_decode = lambda: __import__("contextlib").nullcontext()

    # one decode
    print("decode: one LIBERO-style prompt -> action chunk", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t_dec = time.perf_counter()
    with torch.inference_mode():
        action = vla.predict_action(_dummy_image(args.seed), INSTRUCTION)
    t_dec = time.perf_counter() - t_dec
    dec_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    rec.metric(event="decode", decode_sec=round(t_dec, 4),
               peak_vram_mb=round(dec_peak_mb, 1),
               action_shape=list(np.asarray(action).shape),
               action_sample=np.asarray(action).ravel()[:8].tolist())
    print(f"  decode {t_dec*1000:.0f} ms, peak {dec_peak_mb/1024:.2f} GiB, "
          f"action[:8]={np.asarray(action).ravel()[:8].round(3)}", flush=True)

    # one LoRA step
    print("LoRA: one r=8 step (synthetic batch)", flush=True)
    lora_results = _apply_lora_step(vla, rec, args.seed)
    print(f"  loss={lora_results['lora_loss']:.4f} "
          f"step={lora_results['lora_step_sec']*1000:.0f} ms "
          f"peak={lora_results['lora_peak_vram_mb']/1024:.2f} GiB", flush=True)

    rec.sample_power()
    rec.finalize(
        status="completed",
        results={
            "params": int(n_params),
            "load_sec": round(t_load, 2), "load_vram_gb": round(vram_mb / 1024, 2),
            "decode_sec": round(t_dec, 4), "decode_peak_vram_gb": round(dec_peak_mb / 1024, 2),
            **lora_results,
            "kill_criterion": "fp16 at batch 1 must fit 8 GB with gradient checkpointing",
            "verdict": "fit" if (dec_peak_mb / 1024) < 8.0 else "DID NOT FIT",
        },
        tolerance_note="toolchain validation, not a scored claim")
    print(f"\nrecorded: {rec.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
