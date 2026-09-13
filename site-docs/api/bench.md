# API — Bench suite

The bench modules are the reproducibility surface. Each module exposes
`run(rows, out, ctx)` and declares a `SMOKE` row set; the driver
`bench.all` discovers them and records with the ledger.

::: bench.all
::: bench.compress
::: bench.n2_sweep
::: bench.n2r_sweep
::: bench.n2r2_sweep
::: bench.n8c_int8
::: bench.n9_repair
::: bench.lyapunov_curve
::: bench.compiler
::: bench.monitor
::: bench.safety
