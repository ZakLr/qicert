# qicert

Certified quantum-inspired compression for vision--language--action models
(VLAMs): exact per-layer bounds, a refusing runtime guard, and audited
safety statistics.

**Headline result** (full frozen protocol, 114,497 action tokens): the
repair-trained compressed model holds **0.4535 accuracy at 2.46× honest
compression** against the uncompressed reference's 0.4468
(non-overlapping confidence intervals), with certificates re-derived from
the deployed weights on all 168 layers.

## Why tensor structure

Each linear layer becomes a chain of small tensor-train (TT) cores. The
spectral norms of those cores multiply into an **exact per-layer Lipschitz
bound** — computable from the compressed weights alone, with no reference
to data. Quantized weights admit no comparable bound, because quantization
error is data-dependent. That asymmetry is the whole submission:
compression anyone can do; certificates only structure can give.

## Contents

- [Quickstart](quickstart.md) — install, smoke test, first benchmark.
- [Pipeline](pipeline.md) — the five stages and what each proves.
- [Certificates](certificates.md) — the bound math and its limits.
- [Runtime guard](guard.md) — load-time manifest, per-step gate, entropy flag.
- [Reproducibility](reproducibility.md) — artifacts, seeds, ledger.
- [API reference](api/compress.md) — every public module and function.
