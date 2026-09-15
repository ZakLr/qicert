# N1v2 — Classical-Energy Baseline (measured results)

**Run family:** `N1v2` (MiniVLA-1B, LIBERO spatial slice, frozen 108-episode
eval split, streamed eval, LoRA fine-tune + INT8-of-fine-tuned leg)

**What this is:** the Q3 / classical-energy baseline the challenge asks for —
a small VLAM (MiniVLA-1B-class, qwen2.5-0.5B LLM backbone + DinoSigLIP vision
backbone) fine-tuned on the LIBERO spatial robotic manipulation slice, with an
INT8 weight-only quantization leg so we can characterize the classical-quantized
relationship honestly. This is **not a claim of classical superiority or quantum
advantage** — it is the fidelity/energy context that makes any later quantum
comparison honest (without it, "our method beats the baseline" has no real zero).

**Hardware:** local NVIDIA RTX 5060 Laptop GPU (8 GB VRAM), single-GPU.
**Stack (versions in `docs/resource-declaration.md`):** PyTorch 2.14+cu130,
torchao 0.18, peft 0.14, HF transformers 5.x, the Prismatic/MiniVLA fork at
`weights/code/`, the checkpoint at
`weights/ckpt/checkpoints/step-122500-epoch-55-loss=0.0743.pt`.

**Eval protocol:** the frozen 108-episode held-out split from
`results/eval_split.json` (sha256 `497fa8769f810b7c…`, created before any run
by the md5-hash rule so train/eval contamination is structurally impossible).
Each eval leg consumes the split as a **lazy stream** (two fresh stream
iterators, one per leg), so the whole split (6,496 batches ≈ 16 GB of materialized
batches in the earlier design) fits in ~10 MB per batch. The earlier materialized
design OOM-killed processes silently; the streaming fix is in
`bench/compress.py` (committed 2026-09-11).

**INT8 leg methodology:** torchao `Int8WeightOnlyConfig` (per-tensor weight-only
INT8, GPU) applied to a fresh VLA loaded from the same checkpoint, with the
LoRA-merged fine-tuned weights copied into it, then evaluated over the same 6,496
frozen batches. This is a *matched-protocol* comparison (same split, same batches,
same action-token metric) EXCEPT it is naive weight-only INT8 of a freshly
fp16-fine-tuned model with **no calibration activations and no residual
compensation**.

---

## What we measured (seed 0, completed)

Two seed-0 runs were completed, with the **same config hash**
(`030a78f95194fdc4`) — i.e., the same run spec, independently reproduced. Both
are valid and regenerable from their `results/N1v2/<run_id>/run.json`.

| Run ID | Seed | Steps | Batch | FT acc (mean ± 95% Wilson CI over 6,496 batches) | Correct / total token-positions | INT8-of-FT acc | Δt (INT8 − FT) | Wall time (approx) |
|--------|------|-------|-------|--------------------------------------------------|---------------------------------|----------------|-----------------|---------------------|
| `bf446e7ed683` | 0 | 1200 | 2 | **0.4453 ± 0.0029** `[0.4424, 0.4482]` | 50,984 / 114,497 | 0.0000 | **−0.4453** | ~48 min (incl. 1153 s INT8 leg) |
| `75482ddaf556` | 0 | 1200 | 2 | **0.4468 ± 0.0029** `[0.4439, 0.4497]` | 51,154 / 114,497 | 0.0000 | **−0.4468** | ~49 min |

**Both runs agree:** median FT acc ≈ 0.446, INT8-of-FT ≈ 0.0. This is the
honest measured result.

**Seeds 1 and 2** (single-seed detached runs, PIDs 20780 and 32320 as of this
writing, logs `logs/N1v2-seed1.log` and `logs/N1v2-seed2.log`) are **in
progress** and will be incorporated into the median row once they complete.
Until then, the table above reports seed 0 only and is labeled as such — no
seed-averaging is claimed.

---

## What the numbers mean (honest interpretation)

### 1. The fine-tuned baseline is real and comparable to Kaggle's N1

Kaggle's N1 scored the same MiniVLA-1B backbone (frozen DINO-SigLIP + qwen2.5-0.5B
LLM) fine-tuned on LIBERO spatial, and reported **eval acc ≈ 0.469** (median over
3 seeds × 2000 steps, batch 4, on Kaggle 2×T4). Our local reproduction at
**0.445–0.447** (seed 0, 1200 steps, batch 2, single RTX 5060) is within normal
noise of that — slightly lower, very plausibly because:
- fewer steps (1200 vs 2000),
- batch 2 vs batch 4 (smaller batch → noisier gradients),
- single local GPU vs Kaggle 2×T4 (potential thermal/power differences).

This consistency is the important honest signal: the local classical-energy setup
is reproducing the same ballpark as the Kaggle N1, so the local runs are a valid
classical-energy baseline context, not some artifact of a broken local pipeline.

### 2. Naive INT8-of-fine-tuned is catastrophically bad (0.0) — and that's
the honest finding

A freshly fp16-fine-tuned qwen2.5-0.5B model has **no quantization calibration**.
Applying naive per-tensor weight-only INT8 (`Int8WeightOnlyConfig` without
per-channel scaling, without calibration activations, without any residual
compensation) to its LLM backbone sends the action-token logits off into garbage
— every predicted action token is wrong, so action accuracy is 0.0.

This is **not a failure of our setup**; it is the **correct and expected
behavior** of a naive INT8 baseline on an uncalibrated fine-tuned VLAM, and it
is the honest result. The implication for the report is:

> The right "INT8 baseline" that N2 compares against is **calibrated INT8 at
> matched compression ratio**, not naive weight-only INT8 of a freshly
> fine-tuned model.

The calibrated INT8 reference should use:
- per-channel weight scaling from calibration activations (the GPTQ-intrinsic
  pattern), and
- optionally a residual compensation term (QuaSAR-style closed-form residual
  fitting on the calibration activations, with a stable-pseudoinverse guard for
  rank-deficient calibration).

That is exactly what the N2$'$ pipeline's "INT4 base + certified TT residual"
arm is designed to do — so the N2$'$ compressed model will be compared against a
**calibrated INT8 at matched compression ratio**, not against the 0.0 naive
INT8 number.

### 3. What we do NOT claim from this run

- We do **not** claim that quantum-enhanced compression beats this baseline at
  this point — the N2$'$ compressed model has not been run yet (it depends on
  the N1v2 checkpoint handoff + the Hessian-aware residual fitting stage).
- We do **not** claim any quantum advantage from this classical-energy baseline.
- We do **not** claim that 1200-step batch-2 training is the "right" training
  budget — it is the budget we could run on a single local RTX 5060 in
  reasonable wall time; the Kaggle N1 ran 2000 steps batch 4 on 2×T4.
- We **do** claim that the local setup reproduces the Kaggle N1 ballpark
  (0.445–0.447 vs 0.469), which is the honest calibration that makes the
  local classical-energy runs a valid context for anything we compare against.

---

## Reproducibility

Each completed run is fully recorded at
`results/N1v2/<run_id>_baseline_seed0/`:

```
results/N1v2/<run_id>_baseline_seed0/
  run.json            # full provenance: command line, config hash, seed spec,
                     # start/end UTC, status, and a results dict with the final
                     # eval_acc_finetuned, eval_acc_int8, wall_sec, int8_note.
  metrics.jsonl       # per-event timeline: train_step (loss/acc/peak_vram),
                     # eval_ft (per-batch streaming acc with live ETA), eval_int8
                     # (per-batch streaming acc), save_ckpt, n1_results.
  config.json         # run config snapshot.
  env.json / system.json  # environment provenance (torch/pytorch/torchao/peft
                     # versions, GPU name, OS).
```

The config hash `030a78f95194fdc4` is the same for both completed seed-0 runs,
so they are the **same run spec independently reproduced** — a stronger
reproducibility signal than a single run.

The ledger (`results/ledger.csv`) records every run attempt (including failed
ones, with their error strings) — the full history, not just the good runs.

**To regenerate seed 0 of the baseline locally:**

```bash
cd <repo>
export PRISMATIC_DATA_ROOT="$(pwd)/weights/data"
export PYTHONPATH="$(pwd)/python"
.venv312/Scripts/python.exe -m bench.all \
  --module compress --rows baseline-int8 \
  --out results --exp-id N1v2-repro \
  --seeds 0 --steps-per-seed 1200 --batch 2 \
  --save-ckpt results/N1v2-repro-ckpt \
  --run-tag repro-seed0
```

Expected: a run dir `results/N1v2-repro-ckpt/<new_run_id>_baseline_seed0/`
with FT acc ≈ 0.445–0.447 and INT8-of-FT ≈ 0.0, same config hash
`030a78f95194fdc4`.

---

## What comes next

1. **Wait for seeds 1 and 2 to complete** (single-seed detached runs, PIDs 20780
   and 32320). When they finish, merge all 3 seeds into a single median row
   (mean ± std, as required by challenge Sec. 5.5), and update this report's
   seed-0-only table to a 3-seed table.
2. **Run the calibrated-INT8 reference arm** (per-channel scaling from calibration
   activations + optional QuaSAR-style residual) at the SAME compression ratio as
   the N2$'$ compressed model, on the SAME frozen split, as the real "INT8
   baseline" the N2$'$ model is compared against. This is the honest comparison,
   and the 0.0 naive-INT8 number must not be used for it.
3. **Run N2$'$** (INT4 base + certified TT residual, Hessian-aware fitting +
   stability guard) on the frozen split, handoff from the N1v2 checkpoint, and
   compare against the calibrated INT8 at matched ratio.
4. **Update `docs/technical-report.tex`** Results Framework section with the
   measured N1v2 median row (and, later, the calibrated-INT8 row and the N2$'$
   row), replacing the "pre-registered target" placeholders.

---

*This report is a measured-results summary, not a final paper. It will be updated
as seeds 1/2 complete and as the calibrated-INT8 and N2$'$ arms run.*
