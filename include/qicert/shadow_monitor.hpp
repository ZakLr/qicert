#pragma once
// Kernel 4 (T4): syndrome-shadow runtime monitor.
// Layer 1: PDU hygiene on the Pauli manifold (syndrome state change =
//          syndromic interrupt, gray-list).
// Layer 2: median-of-means on M random projections (shadow concentration
//          theorem -> statistical false-alarm budget).
#include <vector>
#include <cstddef>

namespace qicert {

struct ShadowMonitorConfig {
  std::size_t num_shadows = 64;
  std::size_t sketch_size = 32;
  double conformal_fpr = 0.01;  // user-set false-alarm budget
};

struct Alarm {
  bool assert = false;    // hard interrupt (PDU hygiene)
  bool gray_list = false; // soft flag (shadow statistics)
};

// feed(x) -> alarm decision for the current inference step.
// prev_syndrome must be carried by the caller (0 on first step).
Alarm monitor_step(const ShadowMonitorConfig& cfg, const double* x,
                   std::size_t dim, int* prev_syndrome);

}  // namespace qicert
