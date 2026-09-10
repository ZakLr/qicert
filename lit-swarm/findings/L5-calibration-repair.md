# L5 — Calibration/activation-aware compression + repair training

**Swarm run:** 2026-09-10 · arXiv backend · 57 unique papers, 15 shortlisted
**Decision relevance: HIGH — directly upgrades our N2″ residual-fitting design**

## 🟢 Findings (this lane is almost pure adoption value)

### F1. AIR — Activation- and Influence-Aware Ranks (2026)
Function-preserving SVD compression with per-layer rank allocation from activation
influence. **Confirms B6 (activation-aware bond allocation) is the state of the art
direction** — uniform bond plans are officially the strawman. Our allocator experiment
should compare against AIR-style influence ranks, not just uniform.
- Adopt: influence-scored rank/bond allocation for the residual; cite AIR + SVD-LLM +
CoSpaDi as the lineage.

### F2. GPTQ-intrinsic LoRA (2606.01412) — already flagged in L1
The design we should steal: fold the low-rank correction directly into the quantization
pass via the calibration Hessian, **training-free**. If our N2″ residual can be fitted
Hessian-aware instead of plain least-squares, accuracy-per-rank improves and we inherit
their error-bound machinery. This is a better default than naive SVD-of-residual.
- Adopt for N2″: residual fitting = GPTQ-Hessian-augmented pass (with QuaSAR's stable
pseudoinverse guard for rank-deficient activations).

### F3. CoA-LoRA — configuration-aware adapters for quantized LLMs (2509.25214)
One adapter that adapts to arbitrary per-layer bit-width configurations without re-fitting.
- Adoptable idea at Phase-II: one TT-residual artifact serving multiple compression
configs; for the report, a one-line related-work mention.

### F4. SLiM (2410.09615) — one-shot quantization + sparsity + low-rank, mathematically
computed adapter values (saliency with invertible/additive features).
- Adoptable: their closed-form adapter computation is a third option for residual fitting
(cheaper than GPTQ-Hessian); note as fallback.

### F5. QuaSAR stability analysis — the failure-mode catalog
Rank-deficient calibration activations ⇒ singular Gram matrices ⇒ spurious rejections of
compensable layers. Our residual fitting must include the truncated-pseudoinverse guard.
- Adopt: guard code + cite their analysis as motivation.

### F6. KV-cache low-rank lane (PuzzleKV, SAKI — 2026)
Side finding: KV-cache compression via low-rank decomposition is its own crowded sub-lane.
Not our target, but one sentence in related work shows breadth (attention-memory
compression is the same math).

## Open questions
- Hessian-aware residual fitting cost on our 1B model: GPTQ pass needs ~128 calibration
  samples; feasibility on 5060 CPU should be measured before committing (fallback: SLiM-style
  closed form).
- Does the GPTQ-intrinsic error bound compose with our TT certificate? (The quantization
  bound + low-rank bound composition is B7's theorem-shaped paragraph — cite their bound
  as the quantization side.)
