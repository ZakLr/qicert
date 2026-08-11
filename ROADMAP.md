# qicert — Refinement Roadmap

**Purpose:** the single living document for "what's next" on the qicert package.
Status of every item is tracked here; check boxes off as work lands. The research-side
plan (experiments N1–N14, kill criteria, budget) lives in the submission folder of the
research workspace (`submission/08-experiments.md`, `submission/09-risks-kills.md`,
`submission/11-package-spec.md`); this file is the package-side execution plan.

**Current state (2026-08-11):** Phase-0 skeleton complete and green — bench suite
(N1–N14 table modules, `--rows` contract, smoke mode), Python package
(`python/qicert/`), C++ kernel *interface* stubs (`include/qicert/`, `src/`), CI,
LICENSE, tests (3/3 pass), all pushed to `github.com/ZakLr/qicert`. The skeleton is a
**shell with real contracts and stub internals**: the C++ kernels are 12–22 line stubs,
there are **no pybind11 bindings**, and the heavy Python paths raise
`NotImplementedError` pending env pins and experiments.

---

## Phase 0.5 — Make the skeleton real (0 GPU-h)

- [ ] **Q19: Pin CUDA-Q** — verify the CUDA-Q install path (local Windows: WSL2 or CPU
      mode; Kaggle: `!pip install cudaq`), freeze exact versions (CUDA-Q, PennyLane,
      tntorch, PyTorch CUDA build) into `environment.yml`. Gates every C++ kernel that
      links simulators/noise models. (owner: reproducibility-auditor)
- [ ] **Implement the 4 C++ kernels for real** (pure C++ first; CUDA-Q hooks where the
      simulator helps; all CPU-testable, 0 GPU-h):
  - [ ] `tt_cross` — query-based TT-cross + maxvol, adaptive truncation-error estimates,
        **exact core spectral norms** (power iteration on cores). Replaces the current
        `L *= 1.0; // TODO(Q19)` in `src/tt_cross.cpp`.
  - [ ] `pauli_family` — greedy commutation grouping (graph coloring), single-Clifford
        simultaneous diagonalization, diagonal kernel application, pruning-certificate
        emission (`‖Σ c_a‖₁` per family).
  - [ ] `iqae` — iterative amplitude estimation with Bayesian posterior updating
        (arXiv:2607.18996 formulation); configurable shot budget + noise model.
  - [ ] `shadow_monitor` — median-of-means over random projections + PDU syndrome gate
        from the commuting-family table.
- [ ] **pybind11 bindings + CMake integration** — expose the four kernels to Python so
      `qicert.compress` / `qicert.certify` / `qicert.safety` / `qicert.monitor` actually
      call C++. Today the static lib builds but nothing calls it.
- [ ] **Per-kernel property tests** — TT-cross reconstruction error on known TT tensors;
      compiler exactness identity (`‖T_exact − T_compiled‖`); IQAE vs MC on a known p;
      shadow bound vs measured alarm rate. These are the unit floor under N14.

## Phase 1 — Make the bench honest (Week 1 on GPU)

- [ ] **N2′ bit-ordering sweep** (2 GPU-h, Q20) — 3 orderings × 1 layer × 1 seed; picks
      the ordering N2's 24-GPU-h sweep uses.
- [ ] **N1 baseline fine-tune + INT8 reference** (8 GPU-h) — backbone behaves on our
      slice (kill criterion R1).
- [ ] **N3 Layer-1 certificate table** (2 GPU-h) — exact Lipschitz products per layer per
      Pareto point; first real numbers for the report's certificate column.
- [ ] **Backbone download + license record** (0 GPU-h, parallel) — LLaVA-1.5-7B /
      MiniVLA-1B checkpoints, license terms recorded at download (Q11 evidence, §6).
      Unblocks N1/N13.

## Phase 2 — Heavy pieces (GPU or toolchain)

- [ ] **Q22: Julia SumOfSquares bridge** — small JSON-in/JSON-out subprocess CLI for SOS
      boxes (N4); committed `Project.toml`/`Manifest.toml`; package must not depend on
      Julia at runtime.
- [ ] **ALS/DMRG training engine** (N11, 8 GPU-h) — compressed-manifold fine-tuning, no
      dense round-trip.
- [ ] **Safety-critical last-pass** (N12, 3 GPU-h) — P3 fine-tune on the
      RESTART/IQAE-falsified slice; tail-shrinkage evidence.
- [ ] **STL spec library + `qicert.oracle`** — machine-readable φ₁–φ₄ / ψ₁–ψ₃ specs and
      the seeded reversible scenario oracle (Q21 specs exist on paper; the code is next).

## Phase 3 — Submission hardening

- [ ] **N14 clean-env audit** (4 GPU-h) — fresh venv: `pip install qicert` +
      `python -m qicert.bench.all` reproduces every table with no manual steps.
- [ ] **CI C++ build check** — CI currently runs pytest + smoke bench only; add the CMake
      kernel build (and pybind11 build once bindings exist) to the workflow.
- [ ] **Report auto-generation** — `qicert.bench` output drops into the report's tables
      (KS5: "every table: `pip install qicert && python -m qicert.bench.<table>`").
- [ ] **Latency profile** — `qicert.bench.latency` on the stated ≤100 ms profile with the
      real kernels (dense vs Performer vs compiled families vs certified-pruned).

---

## Critical path

```
Q19 CUDA-Q pin → real C++ kernels → pybind11 bindings → kernel tests
      → Week-1 trio (N2′ → N1 → N3) → report fill-in (measured values)
```

Everything else runs parallel to this spine. Kill/demote criteria for every experiment
are pre-registered in `submission/09-risks-kills.md` (R1–R12); the honesty contract is
that a failed criterion deletes the overclaim, not the schedule.

## Definition of "finished"

The package is finished when the Phase-3 checklist is all green **and** `N14` passes on a
fresh environment **and** every table in `docs/technical-report.pdf` is regenerated by
`qicert.bench` from measured values (no pre-registered placeholders remain).
