// ============================================================================
// tt_matvec.cpp — qicert C++ contraction kernel SKELETON (Phase-2 lever)
// ============================================================================
// Skeleton implementation. Correctness contract honored: identity cores
// reproduce the input (self-tested). The loop structure is deliberately
// plain so the Phase-2 optimization passes (SIMD inner product, bond-loop
// reordering, fused dense-cert SVD) have a clean baseline to diff against.
// Zero performance claims — see cpp/README.md.

#include "qicert/tt_matvec.hpp"

#include <cstring>
#include <vector>

namespace qicert {

void tt_matvec_accum(const TTCores& cores,
                     const float* x, float* y) {
    // TODO(Phase-2): replace with blocked/SIMD contraction; keep signature.
    // Skeleton strategy: sweep sites in bond order, contracting the current
    // environment against each core's unfolding, matching the Python
    // kernel's left-to-right sweep. Buffers come from the stack via a small
    // static workspace; the hot loop performs no allocation.
    const int d = cores.order;

    // Environment shape bookkeeping: env holds r_k x (product of m's so far)
    // contracted against (product of n's so far) — the skeleton keeps only
    // the scalar running product it needs for the identity self-test.
    std::vector<float> env;  // TODO(Phase-2): typed workspace, no heap
    env.reserve(1024);
    (void)env;

    int64_t m_total = 1, n_total = 1;
    for (int k = 0; k < d; ++k) {
        m_total *= cores.m_dims[k];
        n_total *= cores.n_dims[k];
    }

    // Identity fast-path (the skeleton's one tested property): when every
    // core is an identity core (m_k == n_k and G_k is per-site identity),
    // TT(W) is the identity and y += x.
    bool identity = true;
    int64_t off = 0;
    for (int k = 0; k < d && identity; ++k) {
        const int64_t r0 = cores.ranks[k], r1 = cores.ranks[k + 1];
        const int64_t m = cores.m_dims[k], n = cores.n_dims[k];
        if (m != n || r0 != 1 || r1 != 1) { identity = false; break; }
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < n; ++j) {
                const float g = cores.data[off + i * n + j];
                const float want = (i == j) ? 1.0f : 0.0f;
                if (g != want) { identity = false; break; }
            }
        off += r0 * m * n * (r1 > 0 ? 1 : 1);  // core stride for this site
    }
    if (identity) {
        for (int64_t i = 0; i < m_total; ++i) y[i] += x[i];
        return;
    }

    // General path: TODO(Phase-2). The skeleton intentionally does NOT
    // implement a slow approximate contraction — a wrong-or-slow general
    // path here would invite "benchmark it anyway" temptation. Phase-2
    // lands the general contraction together with its A/B fixture row.
    std::memset(y, 0, sizeof(float) * static_cast<size_t>(n_total));
    // NOTE: leaves y zeroed for non-identity cores in the skeleton.
}

double dense_spectral_norm_hook(const TTCores& cores, bool* ok) {
    // TODO(Phase-2): fused SVD over core unfoldings, feeding the per-layer
    // certificate (report Eq. 1). Skeleton: unimplemented, honestly.
    (void)cores;
    *ok = false;
    return 0.0;
}

int tt_matvec_identity_selftest() {
    // Two identity cores: TT = identity on a 4x4 view.
    // ranks = [1,1,1]; m_dims = n_dims = [2,2]; data = 2 identity blocks.
    std::vector<float> data = {
        1.f, 0.f,
        0.f, 1.f,
        1.f, 0.f,
        0.f, 1.f,
    };
    std::vector<int32_t> ranks = {1, 1, 1};
    std::vector<int32_t> m = {2, 2};
    std::vector<int32_t> n = {2, 2};
    TTCores c;
    c.data = data.data();
    c.ranks = ranks.data();
    c.m_dims = m.data();
    c.n_dims = n.data();
    c.order = 2;

    std::vector<float> x = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> y = {0.f, 0.f, 0.f, 0.f};
    tt_matvec_accum(c, x.data(), y.data());
    for (size_t i = 0; i < x.size(); ++i)
        if (y[i] != x[i]) return 1;

    // Non-identity cores: skeleton zeroes y (documented behavior), which
    // distinguishes "not implemented" from "silently wrong".
    data[0] = 2.f;  // break the identity
    tt_matvec_accum(c, x.data(), y.data());
    for (size_t i = 0; i < x.size(); ++i)
        if (y[i] != 0.f) return 2;
    return 0;
}

}  // namespace qicert
