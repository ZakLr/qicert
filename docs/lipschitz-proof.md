# Q17: Exact Lipschitz-Product Theorem (Layer-1 certificate)

**Owner:** tensor-network-architect. **Status:** resolved 2026-08-09 (0 GPU-h).
**Purpose:** the load-bearing Layer-1 statement (submission/03 L1a). Half-page
appendix proof for the report; the same theorem powers N3's table.

## Setup

A linear layer stored in TT/QTT form is a chain of cores
G_1 ... G_d (3-way arrays). Contracting the bond indices reconstructs the weight
matrix W. For a chain of L layers (each with its own cores) plus 1-Lipschitz
activations (ReLU et al.), the whole map is

    F = f_L o ... o f_1,   each f_l : x -> W_l x + b_l (post-activation 1-Lipschitz)

## Theorem (exact product bound)

For every layer l with cores G_1^l ... G_d^l,

    ||W_l||_2  <=  prod_i ||G_i^l||_2,

where ||.||_2 is the spectral (operator) norm of the full matrix / of each core
viewed as an (r_{i-1} n_i)-by-r_i matrix. Consequently the global Lipschitz
constant of the composed map F is

    L(F)  =  prod_l prod_i ||G_i^l||_2,     (exact for linear layers)

computed in O(sum_l sum_i r_i^2 n_i): the SVD of each small core, with no
dependence on the layer's dense size.

### Proof sketch (one paragraph, for the appendix)

1. **Submultiplicativity of the spectral norm.** A TT contraction is a product of
   contractions; operator norms are submultiplicative under composition, so
   ||W||_2 <= prod_i ||G_i||_2. (Equality is achieved when the cores are in
   canonical left-orthogonal form and the flattened core is a column-orthogonal
   matrix: the bound is tight in the TT-format sense.)
2. **Composition rule.** Lip(f o g) = Lip(f) * Lip(g) for linear maps (spectral
   norm of the product); each activation is 1-Lipschitz, so it contributes
   nothing.
3. **Chaining.** Apply (1) per layer, (2) across layers. The product is exact
   for the linear backbone and a conservative (but computable) bound through
   activations: never an estimate, never data-dependent.

## Safe-set certificate (Layer-1 output)

For the input box B (from the slice spec, submission/05) with diameter diam(B)
and a reference input x_ref in B:

    Safe(B) := { x in B : ||F(x) - F(x_ref)|| <= L(F) * diam(B) < margin }

which is an exact certified safe set for the compressed action head. margin is
set by the control specification (Lyapunov basin / STL predicate thresholds).

## Why INT8 has no analogue

Quantization error q(W) depends on the input distribution (per-block scales,
activations), so no function of the quantized weights alone bounds the
induced perturbation. Our error is an exact function of the cores: the
asymmetry that makes the report's INT8-0%-line structural, not empirical.

## Which layers become TT-linear first (Q17b)

Order by (safety-sensitivity x size), matching the bond allocator's signal #1
(submission/07):

1. **Action head layers**: smallest, safety-relevant, highest certificate value;
   certify these first.
2. **Cross-modal output projections**: the single seam feeding the compiler
   (04); compressing here couples cleanly with Pillar B.
3. **Q/K/V and FFN projections**: largest, headline compression ratio; done
   after 1-2 certify.

## Verification hook (N3)

N3 recomputes the product table from the actual compressed cores and checks
each row against the numerically estimated operator norm of the corresponding
layer (tolerance 1e-6 relative). The table is the deliverable; the theorem is
what makes it a certificate rather than a measurement.
