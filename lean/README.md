# qicert-proofs — machine-checked core claims (Lean 4)

`lake build` (toolchain pinned in `lean-toolchain`, Batteries pinned in
`lake-manifest.json`). No other dependencies.

Checked in `QicertProofs/Qicert.lean`, zero `sorry`:
- `prod_le_prod_of_pointwise` — per-layer bounds compose through products
  (the report's Eq. 1 chaining step; formalized over ℕ in Lean core, the
  deployment reading over ℝ uses the same shape — see the file docstring
  for the exact scope boundary).
- `guard_sound`, `guard_complete`,
  `guard_negative_margin_rejects_all` — the deployed guard's accept
  condition (`python/qicert/certify/guard.py::action_in_certified_set`),
  mirroring the Z3 falsification in `tests/test_certify_guard.py`.
- `QicertProofs/Audit.lean` — `#print axioms` witnesses: only Lean's
  standard foundation axioms (`propext`, `Classical.choice`, `Quot.sound`).

Explicitly NOT proven here: bounds on specific weight tensors (those live
in `results/*/run.json`), closed-loop stability, hardware behavior.
