#include "qicert/pauli_family.hpp"

#include <stdexcept>

namespace qicert {

CompileResult compile_families(const std::vector<PauliString>&) {
  // Phase-0 stub: greedy grouping + Clifford diagonalization land with N7.
  throw std::logic_error("compile_families: not implemented until N7 (compiler bench).");
}

}  // namespace qicert
