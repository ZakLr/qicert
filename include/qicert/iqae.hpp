#pragma once
// Kernel 3 (Pillar C): iterative quantum amplitude estimation with a Bayesian
// posterior (Grinko-style IQAE). Simulated classically; the oracle is the
// seeded reversible scenario suite (qicert.oracle). Cost model printed per
// benchmark; the honest classical race is RESTART + GEV.
#include <vector>
#include <cstddef>

namespace qicert {

struct IQAEConfig {
  std::size_t shots_per_iter = 16;
  double epsilon = 0.01;   // target CI half-width
  std::size_t max_depth = 64;
  double noise_model = 0.0;  // per-gate depolarizing rate (0 = ideal sim)
};

struct IQAEInterval {
  double estimate = 0.0;       // p = Pr[rho < 0]
  double lower = 0.0;
  double upper = 0.0;
  std::size_t oracle_queries = 0;  // the number printed in the report
};

// oracle(seed) -> 1 if the seeded rollout violates the spec, else 0.
using OracleFn = int (*)(std::size_t seed, void* ctx);

IQAEInterval iqae_estimate(OracleFn oracle, void* ctx, const IQAEConfig& cfg);

}  // namespace qicert
