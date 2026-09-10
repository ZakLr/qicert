# L1 — TT/QTT compression of transformers + residual/hybrid quantization

**Swarm run:** 2026-09-10 · arXiv backend · 52 unique papers found, top 15 shortlisted
**Decision relevance: HIGH — defines the competitive position of our P1 pivot**

## 🔴 Threats (must cite + differentiate in the report)

### T1. QuaSAR — Quantization Compensation via Stable Activation-Aware Rank Truncation (arXiv:2608.14149, Aug 2026)
- W4A4 PTQ + **closed-form residual compensation** (quantized base + low-rank residual) —
  this is the *plain-math* version of our P1 architecture, published one month ago.
- **Our delta survives:** QuaSAR's residual is a dense low-rank factor with NO error
  certificate, NO tensor-network structure. Our residual is TT-structured AND ships a
  gauge-minimized composable Lipschitz certificate. The mandatory QI-ablation also splits
  cleanly for us (drop TT structure → their setting).
- **Action:** cite prominently in the P1 motivation; position P1 as "QuaSAR-style
  compensation + certified TT structure." Their numerically-stable truncated-pseudoinverse
  solver is directly reusable for our residual fitting (adoption, not threat).

### T2. GPTQ-intrinsic LoRA (arXiv:2606.01412, Jun 2026)
- Studies exactly `W ≈ Q + LR` with calibration-Hessian-augmented GPTQ; proves
  **information-theoretic lower bounds** for the quantization-plus-low-rank problem.
- **Impact:** the `Q+LR` pattern is now theory-backed — strengthens our P1 legitimacy.
  **Threat:** they prove optimality results; our residual fitting must not look naive.
- **Action:** adopt their insight (fold residual into the GPTQ Hessian pass); cite the lower
  bounds when justifying why residual rank r suffices.

### T3. Saten (arXiv:2505.14871, May 2025)
- Sparse-augmented TT compression **during fine-tuning** — "high-rank nature of pre-trained
  LLMs" named as the blocker, matching our own N2pre finding. They claim full-model
  tensorized fine-tuning works.
- **Impact:** contradicts (or refines) our "post-training TT is dead" narrative — the
  distinction is *tensorized-training* (theirs, works) vs *post-training decomposition*
  (ours, dies). This is exactly the N11/B4 direction.
- **Action:** cite; our negative-result section must be precise about the boundary —
  TT survives only when training co-adapts to the structure. Supports the repair-training
  requirement in our N2″ and elevates B4 (TT-adapters) from "nice extra" to "the
  literature-confirmed mechanism."

### T4. KARIPAP (arXiv:2510.21844, Oct 2025)
- Claims iPEPS + TRG compression of LLaMA-2-7B: 93% memory / 70% param cut, 2–3% acc loss.
  If the numbers hold, the "2D tensor networks beat 1D TT" claim is in print.
- **Caution:** reads like a CompactifAI-lineage paper with bold claims; treat as
  [SECONDHAND-claimed] until verified. Even if true: no certificates anywhere.
- **Action:** cite in related work; do not contest; our certificate moat is untouched.

## 🟢 Adoptable techniques

- **QuaSAR's truncated-pseudoinverse solver** — numerically stable residual fitting under
  rank-deficient calibration activations (their failure-mode analysis of goodness-of-fit
  gates is directly relevant to our residual fitting code).
- **Minima (arXiv:2602.01613)** — confirmed details: conv sensitivity predictor + Tucker/TT/TR
  mixture + healing FT + Triton/CUDA kernels; 64→40 GiB VRAM on Qwen3-32B; 40→50 tok/s.
  Use as the production-realism citation for what custom kernels buy (supports our honest
  "Python TT kernels are 15× slower" framing and the B6 allocator direction).
- **SeeMPS (arXiv:2601.16734)** — maintained Python MPS/QTT library; potential replacement
  for parts of our reference backend in future work; not needed for submission.
- **TT-LoRA (arXiv:2410.09615-lineage, ACM 2025)** — confirms B4 (TT-parameterized adapters)
  is established enough to cite, still uncrowded enough to claim the certified variant.
- **CompactifAI (2401.14109)** — reconfirmed as the field's anchor; our bib already carries
  it (ESANN 2025 nuance noted in prior-art).

## Open questions
- Does KARIPAP's 2-3% claim replicate? (Phase-II verification item, not ours.)
- Is there a certified-residual paper we still missed? The swarm found none — the
  certificate-on-residual-compression niche appears open as of 2026-09-10.
