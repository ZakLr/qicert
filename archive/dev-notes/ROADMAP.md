# qicert — Refinement Roadmap

## Findings (2026-08-25) — RTX 5060 local campaign

**E3: `tt_cross` deep-split crash root-caused and fixed.** Reproduction
(`scripts/tt_cross_repro.py`): real layer-0 q_proj (896x896), d=2 completes,
d>=4 raised `LinAlgError` from `_maxvol`'s exact pivot solve. Mechanism: deep
splits shrink fused mode sizes (e.g. d=8 -> modes of 4-49 entries), so the
random initial skeletons collide; the frame matrix restricted to repeated
columns is rank-deficient, and `linalg.solve(B.T, A.T)` on a singular pivot
block raised. Not a reshape/index bug — the entry oracle was verified exact
(max abs err 0.0 vs dense). Fix (`python_backend._maxvol`): singular-safe
fallback to a least-squares interpolation via `pinv`; the cross loses a
little accuracy instead of crashing. Regression test:
`test_tt_cross_deep_split_survives_rank_deficient_pivots`. Separately noted:
on real weights with near-flat spectra (q_proj participation ratio ~452 of
896; decay slope -0.010 per `results/layer_spectra.csv`) any rank-8 TT has
optimal rel err ~0.99, so cross rel err ~1.05-1.09 at d in {2,4,8} is
close to the truncation bound, not "garbage" as previously assumed.

**E1: uniform scalar bond gauges provably cannot tighten the Layer-1 product.**
Scaling one bond by alpha multiplies both adjacent flattened core norms by
exactly alpha and 1/alpha (both flattenings scale uniformly), so the raw
product is invariant under the scalar gauge subgroup; the reduction capacity
lives in non-uniform per-channel diagonal gauges.
`PythonKernelSet.min_gauge_product` therefore descends over per-channel
diagonal bond scalings (monotone pairwise-product minimization, warm-bracket
golden section); reconstruction invariance <= 3e-16 across shapes, soundness
vs tight norm holds on all tested cases (xfail-with-report guard in place).

**RTX 5060 Laptop local campaign — measured results (all ledger-recorded,
`results/ledger.csv`; native Windows venv, torch 2.10.0+cu130, noiseless
fp16 GPU + numpy CPU kernels; NOT hardware evidence).**

- **E1/N3 gauge certificates**: `min_gauge_product` tightened L on 14/14
  N3 cells (e.g. q_proj r=16 kappa 1.39 -> 1.13), sound (L_min >= tight)
  on 14/14; kappa_min <= kappa_raw everywhere. Descent is monotone by
  construction; reconstruction invariance <= 3e-16.
- **E2 spectra**: `results/layer_spectra.csv` (168 rows). Headline: the
  backbone's singular spectra are NEAR-FLAT (q_proj participation ratio
  ~452/896, decay slope -0.010) — the root cause of everything below.
- **E4/N2pre recon sweep**: 112/112 cells (`results/N2pre/`). TT-SVD rel
  err at param fractions {0.005..0.04} is 0.96-1.00 across layer types —
  uniform-ratio TT truncation at these ratios does not preserve real
  weights with flat spectra.
- **E5 residual pilot**: INT4-per-channel base + matched-budget residual:
  TT correction within 1.00-1.12x of dense-SVD correction (weighted by E7
  activation Grams) => "INT4 base + certified TT residual" is
  budget-competitive IF small corrections are wanted at all.
- **E6 latency**: dense fp16 matmul 36-179 GFLOP/s-equivalent at these
  shapes; TT einsum contraction 5.4-12.7x SLOWER. The 19 s decode is a
  pipeline artifact, not a matmul limit; compressed weights are not a
  local latency lever.
- **E7 INT8 subset ablation**: per-channel weight-only int8 =>
  END-TO-END action drift 0.0000 at every subset scope; per-tensor =>
  1.0897 (actions destroyed). Scope (llm/+projector/all) is irrelevant;
  scale granularity is everything. N1's scored INT8 delta was an artifact
  of quantization granularity, not of quantizing "too much".
- **E9/E10a N1local**: 1000-step LoRA fine-tune at batch 4 fits 8 GB
  (peak 6.92 GiB); enlarged holdout (24 batches, Wilson 95% CI):
  eval action acc 0.4588 [0.4256, 0.4924]. INT8 CPU reference leg failed
  on Windows page-file exhaustion (documented in-run; superseded by E7).
- **E10b N2local** (first LOCAL accuracy-vs-compression points):
  FT baseline 0.4788 [0.4453, 0.5124] over 96 samples; TT-SVD arms at
  fractions {0.01, 0.02, 0.04} ALL collapse to 0.0000 — consistent with
  E2/E4 flat-spectrum rel errs. Verdict: **no advantage found** for
  naive uniform-ratio TT compression on this backbone locally; the only
  lossless local compression arm measured is per-channel INT8 (E7).
  Four earlier no-op arm rows (empty compression inventory from a key-
  prefix bug) were quarantined under `results/_invalid_N2local_prefix/`
  and pruned from the ledger.
- Toolchain deviations, all documented in-code: MiniVLA fork code comes
  from github.com/Stanford-ILIAD/openvla-mini (vanilla openvla does not
  carry the patchable tree); RLDS slice layout flattened to the legacy
  tfds format; training/eval feed via `qicert/data/vla_local.py`
  (fork-equivalent transform/collator) because dlimp->tensorflow cannot
  install on py3.14; TF used ONCE in a py3.12 side venv for the NPZ
  materialization (bridge check: max |mean diff| 0.0, |std diff| 8.9e-14).

---

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

**Status 2026-08-11: Python reference kernels DONE** — the four kernels now exist as a
pure-Python (numpy/scipy) conformance spec in `python/qicert/kernels/` with 17/17 tests
passing, including the parity harness every future backend must reproduce. `bench.all
--backend python` runs the smoke set end-to-end with measured (not placeholder) values.
What remains is the *ports* behind the same contracts + their parity gates.

- [ ] **Q19: Pin CUDA-Q** — verify the CUDA-Q install path (local Windows: WSL2 or CPU
      mode; Kaggle: `!pip install cudaq`), freeze exact versions (CUDA-Q, PennyLane,
      tntorch, PyTorch CUDA build) into `environment.yml`. Gates the C++ backend
      that links simulators/noise models. (owner: reproducibility-auditor)
- [ ] **Port the 4 kernels to C++/CUDA-Q** (pure C++ first; CUDA-Q hooks where the
      simulator helps; all CPU-testable, 0 GPU-h) — **parity vs the Python reference is
      the acceptance test** (tolerances in `tests/test_kernels.py::test_parity_harness`):
  - [ ] `tt_cross` — query-based DMRG-cross + maxvol (reference: `python_backend._cross_als`);
        adaptive truncation-error estimates; exact core spectral norms. Replaces the
        current `L *= 1.0; // TODO(Q19)` in `src/tt_cross.cpp`.
  - [ ] `pauli_family` — greedy commutation grouping + joint-eigenbasis diagonalization
        (reference: `python_backend.pauli_grouping/pauli_diagonalize`); pruning-certificate
        emission (`||sum c_a||_1` per family).
  - [ ] `iqae` — Bayesian IQAE with interleaved depth-0 anchors (Grinko schedule,
        reference: `python_backend.iqae`); configurable shot budget + noise model.
  - [ ] `shadow_monitor` — median-of-means over random projections + PDU syndrome gate
        (reference: `python_backend.shadow_statistics/shadow_syndrome`).
- [ ] **pybind11 bindings + CMake integration** — expose the four kernels to Python so
      `qicert.compress` / `qicert.certify` / `qicert.safety` / `qicert.monitor` actually
      call C++ via `--backend cpp`. Today the static lib builds but nothing calls it.
- [x] **Per-kernel property tests (Python reference)** — TT-cross reconstruction on known
      TT tensors; compiler exactness identity; IQAE interval on a known p; shadow
      false-alarm/anomaly gates; parity harness. **17/17 green 2026-08-11** — the
      unit floor under N14 and the conformance spec for the C++/Julia ports.
- [x] **Run recorder (Decision 2026-08-16)** — `qicert.record.RunRecorder` writes
      run.json/config.json/system.json/env.json/metrics.jsonl + ledger.csv for every
      real run; `bench.all --out DIR --exp-id ... --seed ... --run-tag ...` enables it;
      pilot wired into `bench.compress` kernel-smoke. Field inventory:
      `docs/run-data-register.md`. Tests: 25/25 green. **Never re-run a benchmark
      for missing data.**

## Phase 1 — Make the bench honest (Week 1 on GPU)

- [x] **N2′ bit-ordering sweep** (2 GPU-h, Q20) — 3 orderings x 1 layer x 1 seed;
      picks the ordering N2's 24-GPU-h sweep uses. **Winner: bit-reversed**
      (recon 0.919 vs 0.936 interleaved / 0.939 natural, +2% margin) — recorded
      in results/ledger.csv (178c849b6525).
- [x] **N1 baseline fine-tune + INT8 reference** (8 GPU-h) — SCORED run complete
      2026-08-16 on Kaggle (2x T4, 3 seeds x 2000 steps, batch 4): median eval
      acc 0.469 (LoRA FT) vs 0.121 (INT8 ref), delta -0.348; loss 8.9 -> 2.4-2.8,
      train acc 0.50-0.53. R1 comparator established; backbone behaves on our
      slice, kill criterion R1 not tripped. See `results/N1-report.md`.
- [ ] **N3 Layer-1 certificate table** (2 GPU-h) — exact Lipschitz products per layer per
      Pareto point; first real numbers for the report's certificate column.
- [ ] **Backbone download + license record** (0 GPU-h, parallel) — LLaVA-1.5-7B /
      MiniVLA-1B checkpoints, license terms recorded at download (Q11 evidence, §6).
      Unblocks N1/N13.
      - [x] MiniVLA-1B downloaded (weights/, gitignored) + exact license strings in
        `docs/licenses.md`; **Q11 decided 2026-08-16: MiniVLA for scored tables,
        documented** (DINOv2 CC-BY-NC caveat stated; qicert code MIT/Apache; derived
        checkpoints not redistributed). See council-log.md.
- [x] **Step 0 — single-layer CPU smoke** (2026-08-16) — `bench.step0` runs TT-SVD +
      TT-cross + L1 Lipschitz cert + INT8/SVD baselines + compiler identity on the real
      MiniVLA q_proj 896x896. Toolchain validation only. Results: cert soundness PASS
      (product 24.8 >= tight 19.0), TT-SVD within 1.56x of matched SVD, INT8 4.1%,
      compiler identity 0.0; rank-8 TT is 1.8% params at ~94% err (real layer is
      high-rank — bond plans matter for N2).
- [x] **Step 1 — MiniVLA-1B robotics smoke on the 5060** (2026-08-16) — native
      Prismatic `load_vla` (only supported MiniVLA format), fp16: **1.25B params at
      2.35 GiB VRAM**; one LIBERO-style prompt→action decode **19.0 s @ 2.64 GiB
      peak**; one peft LoRA r=8 step **3.15 s @ 2.54 GiB**. Kill criterion (fp16 at
      batch 1 < 8 GB): **fit**. Requires `scripts/patches/prismatic-transformers5.patch`
      (transformers 5.15 compat: sdpa, lazy dlimp, dropped generation kwargs) —
      applied to weights/code, validated against clean tree. Docker/WSL2 memory raised
      to 12 GB (.wslconfig) for the 5.5 GB checkpoint load.

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

---

## Decisions (2026-08-16) — chair + user

4. **Dev runtime derived FROM the CUDA-Q image.** `Dockerfile` now builds from
   `nvcr.io/nvidia/nightly/cuda-quantum:cu13-latest` (the user's existing container) +
   torch cu130 wheels + transformers/peft/accelerate + pynvml. ONE container carries
   CUDA-Q AND the ML stack; `ENTRYPOINT []` clears the base banner; invoke
   `/usr/bin/python3` explicitly (WORKDIR has a `python/` dir). Verified on the RTX
   5060: torch 2.13.0+cu130 `cuda=True`, CUDA-Q `nvidia` target initializes, bench
   smoke green inside the container.

5. **Run capture is mandatory, not optional (user).** Every benchmark/fine-tune
   records complete artifacts via `qicert.record` (`--out`); no re-runs for missing
   data. Bench modules that run real experiments MUST open+close a recorder
   (`bench._base.start_run`/`finish_run`) — no recorder, no table. INT8 baseline path:
   **torch.quantization** (no bitsandbytes in v1, confirmed). Warm-up ladder order:
   Step-0 CPU smoke → Step-1 MiniVLA smoke on the 5060 → full N1 (3 seeds).

## Decisions (2026-08-11) — chair

1. **Track order: robotics-first.** MiniVLA-1B (MIT code, HF checkpoints, ~1B) is the plan scored
   robotics backbone and fits comfortably on the RTX 5060 (8 GB, sm_120) via Docker — real N1/N2/N3
   tables at ~1/10 compute, and robotics STL specs (psi1-3) are simpler than AD (phi1-4), so the L3
   chain de-risks first. AD (LLaVA-7B) follows on Kaggle T4s via the N13 mirror. Library is
   track-agnostic; watch-list: tag N2-prime by layer TYPE (transfers to AD), keep the Q-arm oracle
   spec-driven (predicate=cert mode in oracle.py), keep model-specific code in backbones/ad.py + rb.py.

2. **Backend architecture: Python-first v1 + flag.** v1 ships pure-Python kernels as the reference;
   v2 adds qicert --backend {python,cpp} (C++/CUDA-Q) and --sos-backend {python,julia} (Layer-2a SOS,
   Q22 bridge). C++/Julia must pass parity tests against the Python reference (Lipschitz 1e-8,
   TT-cross reconstruction, compiler identity, IQAE on known p, monitor bound). Flag lands in
   bench/all.py main(); wire the pyproject scripts entry point at the same time.

3. **Dev runtime: Docker + CUDA 13 on the RTX 5060 machine.** sm_120 (Blackwell): torch >=2.7 (cu128)
   or >=2.9 (cu13); NGC pytorch 25.01+ optimized. Avoid bitsandbytes in v1 (Blackwell pitfalls) —
   MiniVLA-1B fits fp16/bf16 LoRA, no quantization. CUDA-Q (Q19) works on Blackwell via cuQuantum
   >=1.8 (cu13 containers, driver >=570). Freeze container pins into environment.yml.
