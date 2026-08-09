# AGENTS.md — qicert repository

This repository is the **main solution repo** for the Global Quantum + AI Challenge 2026
submission ("qicert": certified quantum-inspired VLAM compression). The user-mandated
maintenance rule:

> **Always maintain this repo and push every change to it.** This is the authoritative
> home for all code, package, and benchmark deliverables of the submission. The research
> workspace (plan registers, council log, `submission/` blueprints) lives in the sibling
> `aqc-volswagen-quantum-insider` repo; this repo implements it.

## Conventions

- **Bench contract is sacred.** Every report table maps to a bench module in `bench/`;
  `python -m qicert.bench.all [--rows=...]` reproduces it from a clean env. Add rows by
  editing the module, never by creating one-off scripts.
- **Phase-0 skeleton state.** Python interfaces, the bench suite, and packaging are live;
  C++ kernels are interface-pinned stubs (`include/qicert/*.hpp`, `src/*.cpp`) until the
  CUDA-Q environment is pinned (open question Q19). Do not claim implemented what is a stub.
- **No GPU work in CI.** CI runs the smoke bench (`--rows=smoke`) on a tiny budget only.
  Real experiments run on Kaggle 2×T4 / local RTX 5060 via bench modules; results land in
  the plan workspace's `research/experiments.md` ledger (≥3 seeds, mean ± std).
- **Environment discipline.** Any dependency/version change updates `environment.yml`
  **in the same commit** (clean-env rule, plan `REPRODUCE.md` §6).
- **Experiment IDs.** Bench modules reference the plan's experiment IDs (N2, N2′, N3, …,
  N14) verbatim. Never reuse an ID; N9 does not exist (N8 → N10).
- **License.** Apache-2.0 (`LICENSE`). The challenge's open-source requirement (§4.2).

## Layout

```
include/qicert/   kernel interfaces (tt_cross, pauli_family, iqae, shadow_monitor)
src/              kernel stubs (CUDA-Q linking when env is pinned)
python/qicert/    PyTorch layers, compress, certify, safety, monitor, oracle, backbones
bench/            the bench suite — one module per report table (--rows contract)
cert/             certificate JSON templates
tests/            unit + integration (smoke = bench contract)
docs/             builds from the report
.github/workflows/ci.yml   clean-env smoke on every push
```
