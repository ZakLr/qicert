# L4 — Formal safety: STL repair, conformal risk control, runtime assurance

**Swarm run:** 2026-09-10 · arXiv backend · 69 unique papers, 15 shortlisted
**Decision relevance: MEDIUM-HIGH — shapes the safety section's citations and one new idea**

## 🟢 Findings

### F1. "Silent Failures in Physical AI" literature review (arXiv:2606.00090, Jun 2026)
- A 2026 literature review of runtime action authorization for autonomous systems: black-box
  VLA/world-model actions can be physically consequential while appearing confident —
  silent failures from sensor drift, distribution shift, hallucinated affordances.
- **Gift for the motivation section:** an independent 2026 survey stating exactly the
  problem our monitor + certificate stack addresses. Cite it as the framing reference for
  why compression-without-guarantees is a deployment blocker.

### F2. Conformal prediction for robotics is blooming (5 hits)
- Interaction-aware CP for crowd navigation; learnable CP with context-aware nonconformity;
  egocentric CP for cluttered navigation; **Formal Verification and Control with Conformal
  Prediction (Lindemann survey, 2024)** — already in our register via P5; the swarm
  confirms the lane is hot and our conformal-risk-control upgrade (P5/B10) is
  well-trodden-adjacent: we must cite the robotics-CP cluster and keep our delta
  (CRC on certificate-margin predicates, fed by compressed-model certificates) explicit.

### F3. Neural-Lyapunov certification cluster (5 hits)
- Lyapunov NN (2018) + region-of-attraction search (2024); stability certificates for RL
  policies in the real world (2025); learned Lyapunov shielding for adaptive control (2026);
  certified set convergence for piecewise-affine systems via neural Lyapunov functions (2026).
- **Impact:** the Layer-2b (Lyapunov-margin degradation) story has a rich citation bed and
  an active community; nobody in it works on *compressed* policies — the "how does
  compression degrade the certified region of attraction" question remains ours.
- **Action:** one report paragraph + citations; position N5 (if run) as filling exactly
  that gap; else future work.

### F4. Safe-and-stable neural dynamical systems for robot motion (2025)
- Combines stability certificates with motion planning end-to-end.
- Adoptable framing for the robotics-track safety paragraph.

## Open questions
- No direct hit for "differentiable STL repair" in this lane's top — the pdSTL/RM-for-STL
  papers (2606.19561, 2608.13625) from IMPROVEMENTS.md remain the best citations; the lane
  query may need a targeted re-run ("repair" AND "specification") before claiming
  anything about novelty there.
- Runtime-assurance (Simplex) framing got only one indirect hit — the SOTIF/Simplex
  citations from P4 remain the anchor; treat monitor-as-RA as presentation, not novelty.
