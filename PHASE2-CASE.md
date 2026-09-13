# PHASE2-CASE.md — why the accuracy failed, what would fix it, and what Phase 2 buys

**Created:** 2026-09-13, post-queue (N2R2 closed NO-GO, N5 closed with mode-collapse
diagnosis). This file is the single place where the failure analysis and the Phase-2
technical case live; the report and proposal cite it. Every number resolves to
`results/` (see `EXPERIMENT-LOG.md`).

---

## 1. What failed, precisely

Pre-registered gate (committed before the runs): **ratio ≥ 2.0× AND acc ≥ 0.3968**
(= FT 0.4468 − 0.05). Result: best certified point **0.1640 @ 2.536×** — NO-GO.

The mechanism (verified 2026-09-13): **mode collapse to the modal action token.**
The compressed model emits token `151515` (the stream's most frequent action token)
at every position. Evidence: two different N5 configs agree with the reference on
exactly 584/1007 = the modal token's frequency; three different search configs all
score exactly 86/704 = the same base rate in the search prefix. The model did not
degrade gracefully — it lost output diversity entirely and kept only the safest
statistical answer. A Lipschitz certificate on a collapsed model is sound but
vacuous: the guard never refuses because the model never strays.

## 2. Why it failed — three levers we could not pull on one laptop GPU

**L1 — No training-based repair.** Our repair is *closed-form* (one-shot algebraic
residual fit, no gradient steps) — chosen because it is cheap and keeps the
certification story clean. The literature standard after compression is
*post-compression fine-tuning*: re-training the compressed model (typically a
low-rank adapter on the compressed weights) against the task. This reliably
recovers large accuracy fractions in the literature and was never run here, purely
for compute cost.

**L2 — Rank budget spent on function reconstruction, not the task.** The closed-form
residual reproduces the *original weight matrix* (reconstruction error 0.6–0.8
relative at our ratios — the removed information was real signal on this backbone's
flat spectra). A *trained* residual with the same parameter budget would spend it on
task performance instead — a strictly better use of the same rank. This is testable
cheaply (see §4, arm A).

**L3 — The backbone is small and has no redundant spectra.** MiniVLA's LLM is a 0.5B
Qwen2.5 (498M params total, 4.4M trainable LoRA). Low-rank/TT compression is known
to bite hardest exactly here: small, LoRA-fine-tuned weights have flat spectra, so
every parameter carries unique signal. Larger instruction-tuned backbones are more
overparameterized; TT/SVD literature consistently reports better compression
tolerance with scale. We could not test scale: a 1.5B backbone barely trains on an
8 GB GPU, 3B+ does not train at all in fp16.

**Compute accounting:** total project budget was ~15 GPU-hours on one RTX 5060
(8 GB, batch size 2, peak 4.8 GiB in training). Each lever above needs more: L1/L2
~10–30 GPU-h each (feasible even now), L3 needs 24–80 GB cloud GPUs for 3–7B
backbones.

## 3. If we get a bigger model — real analysis of meeting the bar

**What "bigger" changes and what it doesn't.** Compression ratio is relative; what
matters for the task is *absolute remaining capacity*. At 2.5× a 0.5B model keeps
~200M params; a 3B model compressed 2.5× keeps ~1.2B — more than the *entire*
compressed-and-repaired small model, and drawn from weights with genuinely
redundant spectra. That is the mechanism by which scale plausibly closes a 0.28
accuracy gap. It is a literature-backed hypothesis, not a guarantee — our Phase-1
diagnosis (collapse at 2.5×) is exactly the risk that scale must retire.

**Arm ladder (cheapest first):**

| Arm | What | Cost | Bar prospects | Where |
|---|---|---|---|---|
| A | Repair training on the *current* compressed 0.5B (LoRA on TT+residual, 1200–2400 steps) | ~0.5–1.5 h on the 5060 | plausible: same 4.4M trainable budget that lifted 0.11→0.4468 in the original fine-tune; starts from 0.164 | Phase 1 if time, else Phase 2 week 1 |
| B | Activation-aware rank allocation (per-layer ranks from calibration, no training) | ~2 h | unlikely alone (search showed allocation is second-order vs training) | Phase 2 |
| C | 1.5B backbone (Qwen2.5-1.5B VLA): fine-tune → compress → repair-train | ~1–2 days single GPU | good | Phase 2, one A100-day |
| D | 3–7B backbone + full search + multi-seed | ~1 week cloud | best | Phase 2, cloud |

**Honest odds.** Arm A alone: moderate chance of clearing 0.3968 at ≥2× (the
information removed was real, but the trained residual re-learns task-relevant
structure the closed-form fit cannot). Arms C/D: literature strongly favors scale
for post-compression recovery; we would state "expected to clear, gate committed
in advance" rather than promise.

## 4. Does repair training break anything we built? (checked, not assumed)

**Certificates: no — they re-derive.** The Lipschitz bound is computed *from the
final deployed weights* (per-layer spectral norms chained into the output ball).
After repair training the deployed weights change (TT cores + residual + adapter
delta), so we simply re-run certification on the final matrices: one SVD per layer,
seconds-to-minutes of CPU. Soundness is preserved *by construction* — the bound is
always computed from whatever the deployed weights are, never assumed. What can
change is the *width* of the certified ball: if training inflates spectral norms,
the ball widens and the guard admits more. That is measurable, reportable, and the
guard still enforces exactly what was certified.

**What is lost, honestly:** the Phase-1 selling point "the bound is computable from
the compressed form without materializing the dense matrix" survives only for the
TT part; a trained residual/adapter is certified on its materialized matrix. The
certificate remains exact and enforced; its *cheap derivation story* weakens one
notch. We say this in the report if arm A ships.

**Compression ratio: ~2% cost.** A 4.4M-param adapter added to a ~200M-param
compressed model moves the ratio by ~0.05× (2.54× → ~2.49×). Negligible.

**Latency: no change.** The adapter is two low-rank matmuls on the same path we
already profile; edge-profile latency is dominated by TT contraction either way.

**Guard/runtime: no change.** Same certified set, re-derived numbers.

## 5. What Phase 2 claims, in submission language

Phase 1 proved the safety stack end-to-end and characterized the accuracy failure
mechanically (mode collapse), with three untested levers identified and costed.
Phase 2 pulls them in cost order — repair training (single GPU), activation-aware
allocation, then backbone scale (cloud) — against the same pre-registered gate, on
the same frozen protocol, with certificates re-derived at every step. If scale +
repair training clears the bar, the deliverable is a certified, guard-enforced
compressed VLA at ≥2× that INT8-based stacks structurally cannot match. If it does
not, the deliverable remains the certified sub-2× regime plus the collapse boundary
map — both already measured.
