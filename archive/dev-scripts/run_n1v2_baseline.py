#!/usr/bin/env python3
"""N1v2 baseline revalidation (host python, py3.12 venv).

Reproduces the load-bearing N1 baseline — MiniVLA-1B on the LIBERO spatial
slice, scored on the FROZEN 108-episode eval split — using the environment
actually available on this machine (torchao Int8WeightOnlyConfig, no bnb,
no flash-attn, no Docker).

Run:
  .venv312/Scripts/python.exe scripts/run_n1v2_baseline.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---- env setup mirroring bench/compress._run_n1_seed ---- #
REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("PRISMATIC_DATA_ROOT", str(REPO / "weights" / "data"))
sys.path.insert(0, str(REPO / "weights" / "code"))

from qicert.transformers5_compat import install as _tf5_install
_tf5_install()

import json
import time
import traceback

import numpy as np
import torch

from qicert.data.npz_loader import NpzEpisodeDataset
from qicert.data.vla_local import LocalEvalSplitStream, LocalVLABatcher

CKPT = REPO / "weights" / "ckpt" / "checkpoints" / "step-122500-epoch-55-loss=0.0743.pt"
SPLIT = REPO / "results" / "eval_split.json"
NPZ_ROOT = str(REPO / "weights" / "libero_spatial_no_noops_npz")

EVAL_K = 27  # 27 batches at batch=1 = 108 eval episodes exactly
SEED = 0


def main() -> None:
    t0 = time.perf_counter()
    print(f"[N1v2 host] loading MiniVLA from {CKPT}", flush=True)
    from prismatic.models.load import load_vla

    vla = load_vla(str(CKPT), hf_token=None, load_for_training=False)
    vla = vla.to(dtype=torch.float32).cpu()
    load_sec = time.perf_counter() - t0
    n_params = sum(p.numel() for p in vla.parameters())
    print(
        f"[N1v2 host] loaded in {load_sec:.1f}s, params={n_params:,}",
        flush=True,
    )

    lb = LocalVLABatcher(vla)
    action_tokenizer = lb.action_tokenizer
    num_patches = int(vla.vision_backbone.num_patches)

    ep_ds = NpzEpisodeDataset(NPZ_ROOT)
    with open(SPLIT) as f:
        split = json.load(f)
    assert len(split["eval"]) == 108, f"expected 108 eval episodes, got {len(split['eval'])}"

    print("[N1v2 host] building frozen eval stream (108 episodes, batch=1)", flush=True)
    eval_stream = LocalEvalSplitStream(ep_ds, lb, split["eval"], batch_size=1)
    batches = []
    for b in eval_stream:
        batches.append(b)
        if len(batches) >= EVAL_K:
            break
    assert len(batches) == EVAL_K, f"expected {EVAL_K} batches, got {len(batches)}"
    print(f"[N1v2 host] {len(batches)} held-out batches ready", flush=True)

    def eval_counts(model, label: str) -> tuple[int, int, float]:
        correct = total = 0
        t = time.perf_counter()
        for b in batches:
            out = model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                pixel_values=b["pixel_values"],
                labels=b["labels"],
            )
            logits = out.logits[:, num_patches:-1]
            preds = logits.argmax(dim=-1)
            gt = b["labels"][:, 1:]
            mask = gt > action_tokenizer.action_token_begin_idx
            correct += int((preds[mask] == gt[mask]).sum().item())
            total += int(mask.sum().item())
        sec = time.perf_counter() - t
        acc = correct / max(total, 1)
        print(
            f"[N1v2 host] {label}: correct={correct:,} total={total:,} "
            f"acc={acc:.5f} time={sec:.1f}s",
            flush=True,
        )
        return correct, total, acc

    # ---- FP32 reference (unquantized pretrained; the "baseline" accuracy) ----
    fp_correct, fp_total, fp = eval_counts(vla, "FP32 reference")
    fp_ci = None
    z = 1.959963984540054
    if EVAL_K > 0:
        p = fp_correct / fp_total
        d = 1.0 + z * z / fp_total
        centre = (p + z * z / (2 * fp_total)) / d
        half = z * ((p * (1 - p) / fp_total + z * z / (4 * fp_total * fp_total)) ** 0.5) / d
        fp_ci = (float(centre - half), float(centre + half))

    # ---- torchao Int8 weight-only reference on the SAME frozen batch ----
    import torchao
    import torchao.quantization as q

    print("[N1v2 host] applying torchao Int8WeightOnlyConfig to llm_backbone", flush=True)
    torchao.quantize_(
        vla.llm_backbone,
        q.Int8WeightOnlyConfig(),
    )
    i8_correct, i8_total, i8 = eval_counts(vla, "torchao INT8 reference")
    i8_ci = None
    if EVAL_K > 0:
        p = i8_correct / i8_total
        d = 1.0 + z * z / i8_total
        centre = (p + z * z / (2 * i8_total)) / d
        half = z * ((p * (1 - p) / i8_total + z * z / (4 * i8_total * i8_total)) ** 0.5) / d
        i8_ci = (float(centre - half), float(centre + half))

    delta = i8 - fp
    print("=" * 70, flush=True)
    print(
        f"[N1v2 host] FINAL: baseline_fp={fp:.5f}"
        + (f" [{fp_ci[0]:.5f},{fp_ci[1]:.5f}]" if fp_ci else "")
        + f"  INT8={i8:.5f}"
        + (f" [{i8_ci[0]:.5f},{i8_ci[1]:.5f}]" if i8_ci else "")
        + f"  delta={delta:+.5f}",
        flush=True,
    )
    print("=" * 70, flush=True)

    rec = {
        "exp_id": "N1v2",
        "kind": "baseline (pretrained MiniVLA-1B, no LoRA)",
        "seed": SEED,
        "eval_episodes": 108,
        "eval_batches": EVAL_K,
        "fp32_acc": float(fp),
        "fp32_ci95": fp_ci,
        "int8_acc": float(i8),
        "int8_ci95": i8_ci,
        "int8_delta": float(delta),
        "int8_method": "torchao Int8WeightOnlyConfig (per-tensor, dynamic-act/int8-weight)",
        "note": "host revalidation before torchao is folded into bench/compress; "
        "LN/vision frozen, only LLM backbone quantized. Low precision here is "
        "expected for an un-fine-tuned 1B on this slice — the load-bearing "
        "claim is the fine-tuned LoRA baseline, run separately.",
    }
    out_path = REPO / "results" / "N1v2-baseline-host.json"
    out_path.write_text(json.dumps(rec, indent=2))
    print(f"[N1v2 host] wrote {out_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
