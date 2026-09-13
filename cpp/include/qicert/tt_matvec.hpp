// ============================================================================
// tt_matvec.hpp — qicert C++ contraction kernel SKELETON (Phase-2 lever)
// ============================================================================
// Status: skeleton. Compiles, runs, one self-tested property, zero
// performance claims. The Phase-1 report's measured rows all come from the
// Python pipeline; this header only fixes the interface the port must honor.
//
// The primitive: y += TT(W) @ x for a TT-decomposed weight W with cores
// G_1..G_d in row-major layout, matching the Python kernel's core ordering
// (qicert.kernels.tt_svd output convention) so a Phase-2 A/B against the
// Python fixture is byte-comparable on the same cores.
//
// Skeleton contract:
//   * tt_matvec_accum: correct for identity cores (self-tested), loop
//     structure in place, allocation-free in the hot loop.
//   * dense_spectral_norm_hook: declaration only; Phase-2 fills the fused
//     SVD path that N9's re-certification loop needs (Python: NumPy SVD,
//     minutes; gate: whole backbone <= 30 s).
//   * NO timing calls, NO benchmarks, NO claimed speedups. Measurement
//     happens only through bench/latency_microbench.py against the shared
//     Python/C++ fixture, per cpp/README.md.

#pragma once

#include <cstddef>
#include <cstdint>

namespace qicert {

// Core tensor view: G[k] has shape (r_k, m_k, n_k), row-major, so that the
// unfolding G[k]^(2) used by the Python kernel is the (m_k x r_k*n_k) or
// (r_k*m_k x n_k) reshape depending on parity — kept identical to
// qicert.kernels.tt_svd to make cross-implementation A/B trivial.
struct TTCores {
    const float* data;      // concatenated cores, r_0*m_0*n_0 + ... floats
    const int32_t* ranks;   // r_0 .. r_d  (d+1 entries; r_0 = r_d = 1)
    const int32_t* m_dims;  // m_0 .. m_{d-1}
    const int32_t* n_dims;  // n_0 .. n_{d-1}
    int32_t order;          // d
};

// y += TT(W) @ x.  x has m_0*m_1*...*m_{d-1} entries, y has
// n_0*...*n_{d-1}; caller owns all buffers.  Skeleton implementation is
// plain loops — Phase-2 replaces the inner contraction with SIMD/blocked
// kernels and reorders the bond loop; the signature must not change.
void tt_matvec_accum(const TTCores& cores,
                     const float* x, float* y);

// Phase-2 hook (declaration only in the skeleton): exact spectral norm of
// the matrix represented by `cores`, to feed the per-layer certificate
// without materializing W.  Returns 0.0 and sets *ok=false in the skeleton.
double dense_spectral_norm_hook(const TTCores& cores, bool* ok);

// Self-test helper: builds identity-core TT for an (m x n) matrix and
// checks tt_matvec_accum(x) == x for the identity case.  Returns 0 on pass.
int tt_matvec_identity_selftest();

}  // namespace qicert
