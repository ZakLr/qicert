#include "qicert/tt_cross.hpp"

#include <stdexcept>

namespace qicert {

TTCrossResult tt_cross(MatrixQuery, void*, std::size_t, std::size_t,
                       const std::vector<std::size_t>&, std::size_t) {
  // Phase-0 stub: implementation lands with the Q19 CUDA-Q env pin.
  throw std::logic_error("tt_cross: not implemented until the Q19 environment pin.");
}

double lipschitz_constant(const std::vector<std::vector<double>>& cores) {
  double L = 1.0;
  for (const auto& core : cores) {
    // ||G||_2 = largest singular value; exact per core.
    L *= 1.0;  // TODO(Q19): spectral norm once cores are real.
  }
  return L;
}

}  // namespace qicert
