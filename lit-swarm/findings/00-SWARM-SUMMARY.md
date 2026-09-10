# Literature Swarm — Synthesis (2026-09-10)

**Run:** 6 lanes × 5 queries = 30 arXiv queries (Semantic Scholar was 429-rate-limited;
arXiv Atom API used as primary backend — full abstracts, zero auth). 325 unique papers
harvested, 90 shortlisted, abstracts read for all decision-critical items.
**Raw:** `papers/L{1-6}.json` · **Per-lane findings:** `findings/L{1-6}-*.md`

---

## The five things that change what we do

### 1. Our P1 pivot is now a *crowded pattern* — but the certified version is unclaimed
`W ≈ quantized + low-rank residual` has three fresh papers: **QuaSAR** (2608.14149,
W4A4 closed-form residual), **GPTQ-intrinsic LoRA** (2606.01412, with information-theoretic
lower bounds), **SLiM** (2410.09615). Plain low-rank residual = dead as a novelty claim.
**What nobody has: a residual with a composable, gauge-minimized Lipschitz certificate and
TT structure.** Our report must position P1 as "the certified end of a pattern the field is
converging on" — citing all three, not discovering the pattern.

### 2. The VLA-compression lane is a stampede — 10+ papers in 14 months
BitVLA (1-bit, 11× smaller, matches OpenVLA-OFT), QVLA, ActQuant, QuantVLA, SQAP-VLA,
HBVLA, HoloQ-VLA, DyQ-VLA, CogVLA… **Every single one ships zero certificates.**
This converts our weakest-looking position (not winning compression) into the thesis:
*compression is commoditized; verifiable compression is not.* The report's contrast class
is now concrete and citable.

### 3. "Post-training TT is dead" needs a precise boundary — and Saten marks it
Saten (2505.14871) makes TT work **when fine-tuning co-adapts** to the tensor structure.
Together with our N2pre/N2local failures, the field's actual lesson is: *TT survives when
training co-adapts, dies when applied post-hoc.* This sharpens our negative-result section
AND elevates repair-training from implementation detail to literature-confirmed necessity
(B4 TT-adapters rises in priority: it IS the co-adaptation mechanism).

### 4. Our gauge-certificate idea has a formal foundation — and must cite it
**The minimal canonical form of a tensor network** (2209.14358) proved the gauge-group
structure our P2 trick implicitly uses. Good news: our application (certificate
optimization for NN compression) remains unclaimed and now rests on a 2022 theorem.
Obligation: cite it, state our delta as *application + measured κ improvement*.
Also: HiTaB (2605.10621) means we must scope the certificate claim to
"compression-shipped certificates, printed for every method" — never "best verifier."

### 5. Residual-fitting machinery is there to steal
GPTQ-intrinsic's Hessian-augmented fitting + QuaSAR's stable truncated-pseudoinverse
guard = the right N2″ residual-fitting default (better than naive SVD-of-residual,
guarded against the exact failure mode QuaSAR catalogs). SLiM's closed-form adapter is
the fallback. AIR (2606.19993-lane) confirms activation-aware allocation (B6) is SOTA
direction; uniform bonds are the strawman arm.

## New ideas worth recording (not in any prior repo doc)

| # | Idea | Source | Where it could land |
|---|---|---|---|
| N-1 | **Certified BitVLA bridge**: certificates are orthogonal to train-native compression; propose certifying a 1-bit VLA against its teacher as Phase-II | BitVLA L3 | report future-work |
| N-2 | **One TT-residual serving multiple bit-configs** (configuration-aware residuals) | CoA-LoRA L5 | Phase-II |
| N-3 | **KV-cache low-rank compression** as related-work breadth (same math, crowded lane) | PuzzleKV/SAKI L5 | related work |
| N-4 | **Silent-failures framing**: independent 2026 survey (2606.00090) states the runtime-authorization problem our monitor targets — free motivation citation | L4 | report §1 |
| N-5 | **Dequantization paragraph**: TNs as the classical substrate of QML models — sharpens the "quantum-inspired" identity honestly | L6 | report §2 |
| N-6 | **Compression-degraded region-of-attraction**: neural-Lyapunov community never studies compressed policies; N5 would fill a real gap | L4 | N5 justification / future work |
| N-7 | **Privacy side-benefit of TT structure** — one breadth sentence | L2 | related work |

## Threat register updates (merge into research/prior-art.md)

| ID | Item | Severity | Action |
|---|---|---|---|
| T-S1 | QuaSAR (2608.14149) — quantized base + residual, Aug 2026 | HIGH | cite + differentiate (certified TT residual); adopt their solver |
| T-S2 | GPTQ-intrinsic LoRA (2606.01412) — Q+LR theory + bounds | HIGH | cite; adopt Hessian fitting |
| T-S3 | Saten (2505.14871) — tensorized fine-tuning works | MEDIUM | cite; sharpen negative-result boundary |
| T-S4 | KARIPAP (2510.21844) — iPEPS 93%/70% claims [SECONDHAND] | MEDIUM | cite cautiously; verify later |
| T-S5 | Minimal canonical form (2209.14358) — gauge theory exists | OBLIGATION | cite as P2's foundation |
| T-S6 | HiTaB (2605.10621) — stronger general verifiers exist | SCOPING | never claim "best verifier" |
| T-S7 | BitVLA (2506.07530) + VLA stampede — crowded lane | POSITIONING | contrast class; certificate moat sentence |
| T-S8 | S2 API unauthenticated = unusable for swarms (429) | TOOLING | arXiv backend is default; S2 only with API key |

## Immediate actions feeding Plan A (submission)

1. **Report §2 (positioning):** add the commoditized-vs-certified paragraph with the
   VLA-stampede citation block (all 10, one sentence each max).
2. **N2″ residual fitting:** switch default to Hessian-augmented + stable-pseudoinverse
   guard; cite GPTQ-intrinsic + QuaSAR in the method section.
3. **P2 gauge certificates:** add 2209.14358 citation + "gauge-optimal certificate"
   terminology; run the 30-min follow-up search (canonical form × Lipschitz) before
   claiming application novelty.
4. **Negative-result section:** add Saten to draw the co-adaptation boundary precisely.
5. **Bib update:** ~14 new entries (IDs in findings files); mark all [abstract-verified].

## Swarm coverage gaps (re-run before final report)
- L2: "canonical form AND Lipschitz", "gauge AND certificate" (novelty check for P2)
- L4: "specification AND repair", "STL AND gradient" (differentiable-STL novelty check)
- L1: "tensor ring AND compression" (KARIPAP family verification)
- L6: "amplitude estimation AND robotics" (IQAE revival check)
