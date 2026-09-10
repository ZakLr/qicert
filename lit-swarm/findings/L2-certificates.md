# L2 — Certificates: Lipschitz, gauge optimization, canonical forms

**Swarm run:** 2026-09-10 · arXiv backend · 48 unique papers, 15 shortlisted
**Decision relevance: HIGH — our certificate moat's validity depends on this lane**

## 🔴 Threats

### T1. The minimal canonical form of a tensor network (arXiv:2209.14358, Sep 2022)
- Defines and proves a **minimal canonical form** over the full gauge symmetry of tensor
  networks (MPS and PEPS), with a fundamental theorem: same minimal canonical form ⟺
  gauge equivalent (up to limits) ⟺ same physical state.
- **This is the theoretical foundation our gauge-minimized certificates (P2/B1) rest on.**
  It is *good news and a citation obligation*: our "minimize ∏‖G_k‖ over the gauge group"
  is an application of gauge-fixing to *certificate optimization* — a use case the TN
  literature has not made (they use canonical forms for numerical stability/entanglement
  theory, not for shipping tight spectral bounds with NN weights).
- **Action:** cite as the formal basis; state our contribution as "gauge-optimal
  certificates: choosing the gauge that minimizes the composable Lipschitz product, and
  printing it per artifact." The 2022 paper does the gauge theory; we do the NN-compression
  application + measured κ improvements. This is exactly the differentiation that survives
  a reviewer.

### T2. HiTaB — Hierarchical End-to-End Taylor Bounds (arXiv:2605.10621, May 2026)
- Complete-NN verification via 0th/1st/2nd-order Taylor bounds with compositional Hessian
  Lipschitz propagation. A stronger general-purpose NN verifier exists.
- **Impact:** for *full-network* verification, HiTaB-class methods beat layer-wise
  product bounds — we cannot claim "best verifier." We never should: our claim is the
  **certificate surface across compression levels and methods** (a compression-paper
  artifact), not a better verifier.
- **Action:** cite in the certificate section; scope claim to "certificates that ship with
  compression, computed identically for every method" — not "tightest possible NN bound."

## 🟢 Adoptable / supportive

- **Quasioptimal alternating projections (arXiv:2305.xxxx lane hit, 2023)** — alternating
  projection theory for low-rank approximation; supports the correctness argument of our
  alternating bond-rescaling optimizer.
- **Lipschitz-based robustness certification for RNNs via convex relaxation (2025)** and
  the neural-Lyapunov cluster (Lyapunov NN 2018/2024, stability certificates for RL
  policies 2025, certified set convergence for piecewise-affine systems 2026) — a healthy
  active lane to cite in the Layer-2b narrative; no one combines these with
  compressed-weight certificates.
- **Introduction to MPS/TN (2026), Parallelized contraction of TT/MPO (2026)** — textbook
  and tooling citations for the report's methods section.
- **Privacy-preserving ML with tensor networks (2022)** — sideways surprise: TT structure
  as a privacy mechanism. A one-line "adjacent benefits of TT structure" mention is a cheap
  bonus signal of breadth.

## Open questions
- Does a *minimal-canonical-form-based* Lipschitz certificate exist anywhere? Swarm found
  none — searching "canonical form AND Lipschitz" and "gauge AND certificate" directly on
  arXiv listing pages is a 30-minute follow-up before the report claims novelty.
- Our gauge minimization currently targets ∏‖G_k‖₂ (spectral product). The minimal
  canonical form literature optimizes different objectives (local tensors isometric).
  Confirm our objective is well-posed under their gauge-group characterization — likely
  yes (the product is gauge-non-invariant, which is precisely why minimizing it works).
