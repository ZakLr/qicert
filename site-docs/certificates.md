# Certificates

## Per-layer exact bound

For a linear layer $W \in \mathbb{R}^{M \times N}$ factorized into
tensor-train cores $G_1, \dots, G_d$, the spectral norm satisfies

$$\|W\|_2 \le \prod_{k=1}^{d} \|G_k\|_3$$

where $\|G_k\|_3$ is the spectral norm of core $k$ as a 3-way operator.
This is exact algebra on the TT structure — no data, no bootstrap, no
noise model. The layer reports two numbers: the bound $L$ and the tight
norm $\|W\|_2$ via power iteration through the contraction; the run
passes iff $L \ge \|W\|_2$.

## Chaining (what the guard actually enforces)

If every layer satisfies $\|W_i\|_2 \le L_i$, then the end-to-end Lipschitz
constant satisfies $\|\text{model}\|_2 \le \prod L_i$. The deployment
certificate used by the guard is the resulting ball: under an input
perturbation of diameter $\delta$, the output action stays within
$\prod L_i \cdot \delta$ of the reference action. In a 168-layer network
that product is vacuous (~$10^{71}$ on the 0.5B backbone) — expected for
any deep model including the dense baseline — so the **shippable** claim
is per-layer exactness on the deployed weights, not the chained ball as an
operational number.

## What proofs cover

- Python smoke: every recorded run checks bound $\ge$ tight norm per layer.
- Z3 falsification (`tests/test_certify_guard.py`): the guard's
  soundness/completeness properties are checked by the SMT solver finding no
  counterexample.
- Lean 4 (`lean/`): the same guard properties plus the composition step
  ($a_i \le b_i \implies \prod a_i \le \prod b_i$) are proved with
  `lake build` clean, zero `sorry` (foundation-only axioms).

## Limits

- No closed-loop trajectory guarantee (single-step action bound only).
- Not a hardware security proof (noiseless simulation; simulated amplification).
- Certificates are recomputed after repair training — the guard never serves
  stale pre-repair bounds.
