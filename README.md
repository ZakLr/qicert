# qicert

**Certified quantum-inspired compression for VLAMs, with exact bounds, provable
interaction pruning, and formal safety tails — everything reproducible from one
`pip install`.**

```
python -m qicert.bench.all     # reproduces every table in the report
python -m qicert.bench.all --rows=smoke   # CI smoke set (tiny, fast)
```

## One-liner

Certified quantum-inspired compression for Vision-Language-Action Models: compress into
QTT tensor networks, compile cross-modal interactions into commuting-Pauli families,
certify what remains with a three-layer certificate stack (exact Lipschitz bounds, SOS
local proofs, STL tail bounds), and guard deployment with a syndrome-shadow monitor.

The INT8 baseline can match our accuracy at matched ratio. It **cannot** match our
certificates — quantization error is data-dependent and unfactorizable; ours are exact
properties of the compressed cores.

## Status

**Phase-0 skeleton.** Interfaces, packaging, and the bench-suite contract are live;
experiments have not run yet. GPU/CUDA-Q kernels land with the pinned environment
(open question Q19). No table in the report exists until its bench module runs ≥3 seeds.

## Layout (per `submission/11-package-spec.md` of the plan workspace)

```
include/qicert/   C++ kernel interfaces: tt_cross, pauli_family, iqae, shadow_monitor
src/              C++ stubs (CUDA-Q linking when the env is pinned)
python/qicert/    PyTorch layers, compress, certify, safety, monitor, oracle, backbones
bench/            the bench suite — one module per report table (--rows contract)
cert/             certificate JSON templates (Lipschitz table, pruning certificates)
tests/            unit + integration; CI runs the smoke bench
docs/             builds from the report
.github/workflows/ci.yml   clean-env build + smoke bench on every push
```

## The bench contract

- Every report table maps to a bench module: `compress` (N2, N2′, N3),
  `certify_layers12` (N4, N5), `safety` (N6), `compiler` (N7), `monitor` (N10),
  `training` (N11, N12), `cross_track` (N13), `latency` (≤100 ms profile).
- Every module takes a `--rows` argument; `qicert.bench.all` runs all modules and prints
  a Markdown table that drops into the report.
- CI runs `--rows=smoke` on a tiny parameter budget so every commit is verified.
- The clean-env gate (N14): fresh venv → `pip install qicert` → `bench.all --rows=smoke`
  succeeds with no manual steps, on a Kaggle T4 image and on the local RTX 5060.

## Environment

Pins are frozen at the end of Phase 0 (Q19/Q22): CUDA-Q, PennyLane, tntorch (reference
path only), PyTorch CUDA 12.x, Julia + SumOfSquares.jl behind a small CLI. See
`environment.yml` and the plan's `REPRODUCE.md`.
