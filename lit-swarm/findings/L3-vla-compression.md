# L3 — VLA compression + deployment + latency on edge

**Swarm run:** 2026-09-10 · arXiv backend · 61 unique papers, 15 shortlisted
**Decision relevance: CRITICAL — this lane is now officially CROWDED; our positioning must
shift from "VLA compression" to "certified VLA compression"**

## 🔴 Threats (positioning-level)

### The VLA-compression stampede (2025–2026, all post-dating our plan's design)
- **BitVLA (2506.07530):** native 1-bit (ternary) VLA trained from BitNet b1.58-2B4T;
  Quantize-then-Distill for the vision encoder; matches full-precision OpenVLA-OFT with
  **11.0× memory reduction**. Deployment-first training, not post-hoc compression.
- **QVLA (2602.03782):** channel-aware low-bit PTQ — already in our register; confirmed.
- **ActQuant (2605.24011):** sub-4-bit action-guided quantization on π0.5/LIBERO.
- **QuantVLA (2602.20309):** scale-calibrated PTQ for VLAs.
- Plus SQAP-VLA (quantization-aware pruning), HBVLA (1-bit PTQ), HoloQ-VLA (uniform W4A4),
  DyQ-VLA (temporal-dynamic-aware), CogVLA (routing/sparsification), QuoVLA, X-Tokenizer,
  and Just-Noticeable-Difference token compression.
- **Verdict:** a bare "we compress VLAs" claim is now dead on arrival — at least 10 papers
  in 14 months. **Every one of them ships no certificate object.** The crowded lane *is*
  the argument for our moat: compression claims are commoditized; certified compression
  is not. Our report must say this explicitly — the crowd is our contrast class.

### BitVLA's "deployment-oriented view" framing (T-level: high)
- BitVLA argues post-hoc compression is the wrong paradigm (train-native instead). A judge
  could ask "why compress post-hoc at all?"
- **Answer in report:** our contribution is not the compression leaderboard spot; it is
  the *certificate* the deployment pipeline consumes for ISO 26262/10218 evidence.
  A 1-bit native model still needs a runtime-verifiable error bound against its teacher —
  certificates are orthogonal to (and compatible with) their pipeline. Note as future-work
  bridge: "certified BitVLA."

## 🟢 Adoptable

- **ActQuant's action-guided sensitivity** — action-head-aware quantization granularity.
  Cheap idea-transfer: our residual allocator (B6) could weight layers by downstream
  action-token influence; ActQuant validates that this matters on LIBERO-class benchmarks.
- **SQAP-VLA's synergistic pruning+quantization** — same synergy shape as our
  quantization+TT-residual; cite as parallel-direction evidence.
- **JND token compression (2608.21247)** — tolerable-deviation framing at the token level;
  conceptually adjacent to our certified-margin framing; cite in motivation (already
  flagged in prior-art).

## Open questions
- Do any of the 10 VLA-compression papers report *any* formal error bound? (Spot-check 3
  during report writing; expected: none — that sentence becomes a report strength.)
- LIBERO success-rate protocols vary across these papers; our action-accuracy metric needs
  a one-paragraph reconciliation note.
