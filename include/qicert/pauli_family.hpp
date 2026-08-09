#pragma once
// Kernel 2 (Pillar B): the commuting-Pauli interaction compiler.
// Group Pauli strings into commuting families, diagonalize each by a single
// Clifford (pass 3-4), rewrite T = sum_j C_j^dag D_j C_j exactly (pass 4),
// emit hardware table (pass 5) and pruning certificates (pass 6).
#include <string>
#include <vector>
#include <cstddef>

namespace qicert {

struct PauliString {
  std::vector<int> paulis;   // 0=I,1=X,2=Y,3=Z per qubit
  double coeff = 0.0;
};

struct Family {
  std::vector<std::size_t> members;  // indices into the interaction table
  std::vector<int> clifford;         // the diagonalizing Clifford (pattern)
  double prune_cost = 0.0;           // ||sum_{a in F} c_a||_1
};

struct CompileResult {
  std::vector<Family> families;
  double exactness = 1.0;  // |T_exact - T_compiled| / |T_exact|
  // Pass-5 hardware pathway: qubits, depth, shots.
  std::size_t qubits = 0;
  std::size_t depth = 0;
  std::size_t shots = 0;
};

CompileResult compile_families(const std::vector<PauliString>& table);

}  // namespace qicert
