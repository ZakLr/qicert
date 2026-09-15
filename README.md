# qicert

[![clean-env smoke](https://github.com/ZakLr/qicert/actions/workflows/ci.yml/badge.svg)](https://github.com/ZakLr/qicert/actions/workflows/ci.yml)

**Certified quantum-inspired compression for vision--language--action models
(VLAMs): exact per-layer bounds, a refusing runtime guard, and audited
safety statistics — everything reproducible from one `pip install`.**

```
python -m qicert.bench.all                # reproduces every report table
python -m qicert.bench.all --rows=smoke   # CI smoke set (tiny, fast)
```

## What this is

VLAMs (models that turn camera images plus a language instruction into robot
actions) are too large for embedded controllers, and standard compression
(quantization, pruning) shrinks them without saying anything about what the
smaller model may still do. qicert compresses with tensor-train (TT)
structure instead: each linear layer becomes a chain of small tensor cores,
and the spectral norms of those cores multiply into an **exact per-layer
Lipschitz bound** — computable from the compressed weights alone, with no
reference to data. Quantized weights admit no comparable bound, because
quantization error is data-dependent. A deployment guard verifies the
certificate manifest at load time and refuses any action outside the
certified ball at every inference step.

Headline result (full frozen protocol, 114,497 action tokens): the
repair-trained compressed model holds **0.4535 accuracy at 2.46× honest
compression** against the uncompressed reference's 0.4468 (non-overlapping
confidence intervals), with certificates re-derived from the deployed
weights on all 168 layers. The training-free route is reported as an honest
negative (0.164 with a diagnosed collapse mechanism), not hidden. Details,
caveats, and the full ablation are in `docs/submission/technical-report-v2.pdf`.

## Layout

```
include/qicert/   C++ kernel interfaces: tt_cross, pauli_family, iqae, shadow_monitor
src/              C++ stubs (CUDA-Q linking once the GPU env is pinned)
cpp/              self-tested C++ contraction skeleton (scope in cpp/README)
python/qicert/    PyTorch layers, compress, certify, safety, monitor, oracle, backbones
bench/            the bench suite — one module per report table (--rows selects rows)
cert/             certificate JSON templates (Lipschitz table, pruning certificates)
tests/            unit + integration (84 tests); CI runs the smoke bench
lean/             Lean 4 proofs of bound composition + guard properties
docs/             the reports (PDF) plus method notes
.github/workflows/ci.yml   clean-env build + smoke bench on every push
```

## The bench contract

- Every report table maps to a bench module: `compress` (baseline fine-tune,
  bit-ordering study, certificate table), `n2_sweep` (uniform compression),
  `n2r_sweep` / `n2r2_sweep` (residual repair arms), `n8c_int8` (calibrated
  INT8 reference), `n9_repair` (repair training + full-protocol confirm),
  `lyapunov_curve` (degradation analysis), `safety` (estimator comparison),
  `compiler` (Pauli compiler acceptance), `monitor` (runtime monitor),
  `training` (tensor-native training), `cross_track` (second backbone),
  `latency` (≤100 ms edge profile).
- Every module takes a `--rows` argument (`smoke` = tiny fast subset);
  `qicert.bench.all` runs modules and prints Markdown tables that drop into
  the report.
- CI runs `--rows=smoke` on a tiny parameter budget so every commit is verified.
- Clean-env check: fresh venv → `pip install qicert` →
  `bench.all --rows=smoke` succeeds with no manual steps.

## Machine-checked core claims

`lean/` holds Lean 4 proofs (`lake build` clean, zero `sorry`): bound
composition (the report's Eq. 1 chaining step) and guard
soundness/completeness, mirroring the Z3 checks in
`tests/test_certify_guard.py`. Scope boundary stated in `lean/README.md`.

## Environment

See `environment.yml` for pinned versions (PyTorch CUDA build, transformers,
numpy/scipy, CUDA-Q for the estimator simulation, Julia reserved for the
sum-of-squares bridge). The bench runner resolves its own interpreter and
`PYTHONPATH`; just launch it with the venv python.
