#pragma once
// Kernel 1 (T2): TT-cross + maxvol. SVD-free decomposition from entry
// evaluations — O(sum m_i n_i r_i^2) queries, never forming the dense W.
// Classical algorithm on a quantum-native data structure.
#include <vector>
#include <cstddef>

namespace qicert {

// Decompose an m x n matrix given via a query callback into TT cores.
// cores[i] is the i-th 3-way core (G_1..G_d); ranks follow mode_factors.
struct TTCrossResult {
  std::vector<std::vector<double>> cores;  // flattened 3-way cores
  double truncation_error_estimate = 0.0;  // free adaptive estimate (maxvol)
};

using MatrixQuery = double (*)(std::size_t row, std::size_t col, void* ctx);

TTCrossResult tt_cross(MatrixQuery query, void* ctx, std::size_t m,
                       std::size_t n, const std::vector<std::size_t>& mode_factors,
                       std::size_t rank);

// Exact Layer-1 Lipschitz constant of a core chain: L = prod_i ||G_i||_2.
double lipschitz_constant(const std::vector<std::vector<double>>& cores);

}  // namespace qicert
