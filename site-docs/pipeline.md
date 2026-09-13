# Pipeline

Five stages, each enforced by a bench module and recorded in the machine
ledger. The names below are the documentation's plain names — the historic
bench tags (e.g. N9, N2R2) map one-to-one and are listed once per row for
readers arriving from code or run logs.

| Stage | Bench module / rows | What it proves |
|---|---|---|
| 1. Baseline fine-tune | `compress` / `baseline-int8` (historic: N1 / N1v2) | The reference accuracy every compression claim is judged against. Reports token-level action accuracy with a Wilson confidence interval. |
| 2. Bit-ordering study | `n2prime` / `bit-ordering` (historic: N2′) | Which TT layout (tensor-train index grouping) reconstructs best at fixed rank? Winner feeds every later sweep. |
| 3. Uniform compression | `n2_sweep` / `compression-pareto` (historic: N2) | The honest negative: uniform-rank TT truncation destroys accuracy at ≥2× on this backbone's flat spectra. |
| 4. Residual repair (two arms) | `n2r_sweep` / `residual-go-no-go` (historic: N2R) and `n2r2_sweep` + `n2r_search` / `n2r-search`, `n2r-confirm` (historic: N2R-v2) | Closed-form low-rank residual folded back into the compressed operator, optionally weighted by per-channel activation statistics from train episodes. |
| 5. Repair training | `n9_repair` / `repair`, `repair-confirm` (historic: N9 / N9C) | Post-compression LoRA adapter trained on **train episodes only** (frozen eval excluded and verified), merged and re-evaluated on the full 6,496-batch frozen protocol. First GO in the paper. |
| Baseline reference | `n8c_int8` / `calibrated-int8` (historic: N8C) | Calibrated INT8 at 1.997×, essentially lossless. The classical comparator the challenge asks for. |
| Degradation analysis | `lyapunov_curve` / `lyapunov-degradation` | Agreement-vs-compression curve; diagnosis that deep compression collapses to the modal token, motivating the entropy gate. |
| Compiler | `compiler` | Commuting-Pauli interaction compiler; identity exact to 2.3e-14, pruning predictions track measurement at r=0.975. |
| Monitor | `monitor` | Syndrome-shadow monitor: false-alarm rate, detection rates, added latency. |
| Safety comparison | `safety` | Three-way rare-event estimator comparison (IQAE vs extreme-value tail vs Monte Carlo). |

## How the claims compose

The repair-trained compressed model (stage 5) carries certificates re-derived
from its **final deployed weights** at honest ratio 2.46× (cores + residual +
adapter bytes counted). The per-seed pipeline for independent replicas is:
`calib-s1/s2 → repair-s1/s2 → confirm-s1/s2` (one reproducible command per
seed, gated on its calibration file so seed-0 stats are never reused on a
different seed's weights).

## Reproducibility contract

Every report table drops out of `qicert.bench.all` from a clean environment;
the experiment queue (`scripts/run_remaining_experiments.py`) enforces
one-GPU-at-a-time, auto-resume, and a mechanical gate (ratio ≥ 2× and
accuracy within 0.05 of reference) read from run artifacts, never hand-edited.
