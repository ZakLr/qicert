# qicert EXECUTION PLAN — to submission (2026-09-12)

**Audience:** any agent or human continuing this work. This file is
self-contained: it assumes you have read nothing else. Follow phases in
order; do NOT launch two GPU jobs at once; record every result in
`results/EXPERIMENT-LOG.md`.

**Repo root:** `challenge/qicert/` · **Interpreter:** `.venv312/Scripts/python.exe`
(bash on Windows: `.venv312/Scripts/python.exe`, NOT `python`)
**All bench commands run from repo root with `PYTHONPATH=python` (and
`PYTHONPATH=python:.` for direct bench module imports).**

---

## 0. Rules for the executing agent (non-negotiable)

1. **AGENTS.md governs everything**: matched-budget baselines, noise-model
   labels, seed medians (never single-run claims), honest kill criteria,
   provenance for every number.
2. **Never invent CLI flags.** Check `bench/all.py --help` and the module
   source before running. The known-good command shape is:
   `PYTHONPATH=python .venv312/Scripts/python.exe -m qicert.bench.all --module <name> --rows=<row> --out results --exp-id <ID> --seed 0 --run-tag <tag> --capture heavy`
   (run from repo root; use `PYTHONPATH=python:. -m bench.all` equivalently).
3. **One GPU job at a time.** Each long job: launch detached with output
   piped through `tee` into `results/logs/<name>.log` (live logs are
   mandatory — past runs that printed only at the end looked "stuck").
4. **Full-split eval is the only R1-grade protocol** (`QICERT_N2_EVAL=full`).
   Single-batch numbers are order-of-magnitude evidence only and must be
   labeled as such.
5. **Pre-registered gates are executed as written.** A NO-GO is a result.
   Never re-tune after seeing the eval number and re-run silently.
6. Tests after every code change: `PYTHONPATH=python .venv312/Scripts/python.exe -m pytest tests/ -q` (baseline: **80 passed**).
7. Commit + push after every completed unit of work. No force-push.
8. Never claim anything in the report that is not a number in
   `results/` with a run dir / JSON artifact behind it.

---

## 1. Status snapshot (verified as of 2026-09-12, commit `4facb25`)

### Measured results (all full-split, validated harness, FT anchor +0.0000)
| Method | Ratio | Eval acc | Delta vs FT 0.4468 | Certificates |
|---|---:|---:|---:|---|
| FT baseline (N1v2 seed 0) | 1.00× | 0.4468 | — | none (unverified model) |
| Calibrated INT8 (N8C) | 2.00× | 0.4466 | −0.0001 | structurally impossible |
| TT+residual (N2R best: plan 0.50, r′=32) | 2.86× | 0.1529 | −0.2939 | 168/168 sound |
| TT uniform (N2, all points) | 2–16× | 0.0000 | −0.4468 | sound |
| Naive weight-only INT8 (N1v2) | ~2× | 0.0000 | — | (calibration artifact) |

**R1 bar** (pre-registered): some point at ratio ≥ 2× with delta ≥ −0.05.
**Status: NOT MET by N2/N2R.** The honest current claim: *the only compressed
model that ships with sound runtime certificates* — not accuracy leadership.

### Infrastructure verified working
- Eval harness `_N2EvalHarness` (bench/n2_sweep.py): validated swap path,
  contamination guard, full-split mode via `QICERT_N2_EVAL=full`.
- FT checkpoint handoff: `results/N1v2-ckpt/seed0.pt` (seeds 1, 2 missing).
- Calibration stats: `results/N2R/calib_seed0.npz` (376 layers, absmax +
  mean-abs after this commit; OLD file has absmax only — recomputed only if
  a weighted experiment needs mean; absmax fallback works).
- Data: local NPZ bridge `weights/libero_spatial_no_noops_npz/` +
  `results/eval_split.json` (108 eval episodes → 6,496 batches). Do NOT use
  the fork's RLDS loader (documented dead end).
- Guard: load gate + step gate machine-checked (Z3, test_certify_guard.py).
- N6 three-arm race RAN: IQAE-sim width 4.0e-5 @ 630 queries vs MC 1e6
  queries for 4.4e-5 (~1587× budget), all covering truth; GEV covers at 1e6.
  Artifacts: `results/safety/threearm_seed0.json`, `conformal_seed0.json`.
- Guard demo: `scripts/demo_guard.py` → 1 SERVE / 4 REFUSE transcript at
  `results/guard-demo/transcript.json`.

### Implemented this cycle (commit 4facb25, 80 tests green)
- Activation-weighted residual fit (`compress_residual.svd_residual` with
  `activation_weight=`; GPTQ-style closed form; certificate unchanged).
- `bench/n2r2_sweep.py` — N2R-v2: weighted arm + mixed allocation.
- `bench/lyapunov_curve.py` — N5-v2: degradation curve (GPU leg) +
  weight-space predictor with LOO R² (CPU leg) + spectral deviation log.
- `safety.py`/`bench/safety.py` — live three-arm race + scenario-opt +
  conformal (all computed, not pending).
- `calibrate.py` — stores mean|activation| per channel (in addition to absmax).

---

## 2. GPU QUEUE (launch strictly one at a time, in this order)

**One-command option (preferred, 2026-09-12):** the whole queue runs via

```bash
cd "/c/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert" && \
.venv312/Scripts/python.exe scripts/run_remaining_experiments.py
```

`scripts/run_remaining_experiments.py` implements everything below: per-stage
pre-flight compile/import checks, full pytest first, GPU-busy detection,
live timestamped logs in `results/logs/queue/<stage>.log`, pre-registered
gating (N2R2 GO/NO-GO decides the mixed stage), resume/skip on existing
artifacts, checkpoint-duplicate detection, and a final
`results/QUEUE-SUMMARY.json`. Flags: `--list`, `--dry-run`, `--exclude n1v2`,
`--only <stages>`, `--no-tests`. Interrupted? Just re-run it — finished
stages skip automatically.

Manual commands (equivalent, if you prefer step-by-step): each command run
from repo root in bash. Expect live progress lines (`<- live` markers).
Record the verdict + numbers in EXPERIMENT-LOG.md under a new dated entry
BEFORE launching the next job.

### JOB 1 — Calibration top-up (mean-abs stats) — ~10 min
Only needed for the `stat=mean` arm. If skipped, N2R-v2 falls back to
absmax (already on disk) — acceptable.
```bash
cd "/c/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert" && \
PRISMATIC_DATA_ROOT="C:/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert/weights/data" \
PYTHONPATH="C:/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert/python" \
.venv312/Scripts/python.exe -m qicert.calibrate 2>&1 | tee results/logs/calib-mean-topup.log
```
Check: log ends `wrote results/N2R/calib_seed0.npz (376 linear layers)`.
`load_stats` must now return `mean_abs` keys (quick check:
`PYTHONPATH=python .venv312/Scripts/python.exe -c "from qicert.calibrate import load_stats; from pathlib import Path; s=load_stats(Path('results/N2R/calib_seed0.npz')); k=next(iter(s)); print(k, 'mean_abs' in s[k], 'absmax' in s[k])"`).

### JOB 2 — N2R-v2 go/no-go (weighted arm, full-split) — ~1.5–3 h
Pre-registered grid: weighted fit at the N2R-best operating point ± one
step, plus the mixed-allocation point. Gate: **GO = any point ratio ≥ 2×
with delta ≥ −0.05** (same R1 bar; no re-tuning).
```bash
cd "/c/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert" && \
PRISMATIC_DATA_ROOT="C:/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert/weights/data" \
PYTHONPATH="C:/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert/python" \
QICERT_FT_CKPT="C:/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert/results/N1v2-ckpt" \
QICERT_N2_EVAL=full QICERT_N2R2_MODE=weighted QICERT_N2R2_STAT=absmax \
QICERT_N2R2_PLANS=0.50 QICERT_N2R2_RRANKS=32,64 \
.venv312/Scripts/python.exe -m bench.all --module n2r2_sweep \
  --rows=n2r2-go-no-go --out results --exp-id N2R2 --seed 0 \
  --run-tag N2R2-weighted-go-no-go --capture heavy \
  2>&1 | tee results/logs/N2R2-weighted-go-no-go.log
```
Check per point: `DONE ratio=… acc=… sound=…/… full-split (6496 batches …)`.
- If acc ≥ 0.42 at ratio ≥ 2× → **GO**: follow §3.1.
- If not → **NO-GO**: record, then run the ONE mixed-allocation point:
```bash
QICERT_N2_EVAL=full QICERT_N2R2_MODE=weighted QICERT_N2R2_STAT=absmax \
QICERT_N2R2_MIXED=1 QICERT_N2R2_PLANS=0.50 QICERT_N2R2_RRANKS=32 \
.venv312/Scripts/python.exe -m bench.all --module n2r2_sweep \
  --rows=n2r2-go-no-go --out results --exp-id N2R2 --seed 0 \
  --run-tag N2R2-mixed-go-no-go --capture heavy \
  2>&1 | tee results/logs/N2R2-mixed-go-no-go.log
```
(Mixed keeps q/k/v/o full-precision; expect ratio ≈ 1.9–2.1×. Same R1 gate.)
- If neither hits the bar → compression claim is CLOSED: final claim =
  "certified compressed models + measured degradation predictor" (§3.2).
  Do NOT iterate further on compression.

### JOB 3 — N5-v2 degradation curve — ~2–3 h
Measures token-level agreement vs compression ratio (5 points across two
plans/modes) + per-layer spectral deviations. Points APPEND to
`results/lyapunov/degradation_seed0.json`; reference preds cached once.
```bash
cd "/c/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert" && \
PRISMATIC_DATA_ROOT="C:/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert/weights/data" \
PYTHONPATH="C:/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert/python" \
QICERT_FT_CKPT="C:/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert/results/N1v2-ckpt" \
QICERT_N5_PLANS=0.50,0.33,0.25 QICERT_N5_RRANKS=32 \
QICERT_N5_MODE=plain QICERT_N5_BATCHES=500 \
.venv312/Scripts/python.exe -m bench.all --module lyapunov_curve \
  --rows=lyapunov-degradation --out results --exp-id N5 --seed 0 \
  --run-tag N5-degradation-plain --capture heavy \
  2>&1 | tee results/logs/N5-degradation-plain.log
```
Then the weighted-mode pass (adds 2–3 points; uses absmax weights):
```bash
QICERT_N5_PLANS=0.50,0.33 QICERT_N5_RRANKS=32 QICERT_N5_MODE=weighted \
QICERT_N5_BATCHES=500 \
.venv312/Scripts/python.exe -m bench.all --module lyapunov_curve \
  --rows=lyapunov-degradation --out results --exp-id N5 --seed 0 \
  --run-tag N5-degradation-weighted --capture heavy \
  2>&1 | tee results/logs/N5-degradation-weighted.log
```
Then the CPU predictor (seconds, no GPU):
```bash
PYTHONPATH=python:. .venv312/Scripts/python.exe -m bench.all \
  --module lyapunov_curve --rows=lyapunov-predictor --out results \
  --exp-id N5 --seed 0 --run-tag N5-predictor 2>&1 | tee results/logs/N5-predictor.log
```
Check: table row `| lyapunov | n | a +b*r | R2 | LOO R2 | r* | r_meas | completed |`
and `results/lyapunov/predictor.json`. Need ≥ 5 points total for a defensible
fit; report LOO R² honestly (indicative only under ~8 points).

### JOB 4 — N1v2 seeds 1 and 2 (3-seed baseline medians) — ~9–12 h EACH
Run overnight, one at a time. This upgrades the baseline row from a single
seed to a median (AGENTS.md requirement for the headline table).
```bash
cd "/c/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert" && \
PRISMATIC_DATA_ROOT="C:/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert/weights/data" \
PYTHONPATH="C:/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert/python" \
.venv312/Scripts/python.exe -m bench.all --module compress --rows baseline-int8 \
  --out results --exp-id N1v2 --seeds 1 --steps-per-seed 1200 --batch 2 \
  --save-ckpt results/N1v2-ckpt --run-tag host-seed1 2>&1 | tee logs/N1v2-seed1.log
```
(seed 2: same with `--seeds 2 --run-tag host-seed2 --tee logs/N1v2-seed2.log`)
Check: `results/N1v2-ckpt/seed1.pt`, `seed2.pt` exist; eval acc within
~±0.03 of seed 0's 0.4468; update the headline table to the 3-seed median.

---

## 3. Claim assembly (after JOBS 1–3)

### 3.1 If N2R-v2 or mixed hits the R1 bar (GO)
- Headline becomes: "at ≥2× compression, our certified TT+weighted-residual
  model holds ≥0.42 acc — the regime INT8 cannot follow without int4 —
  with sound runtime certificates INT8 structurally cannot produce."
- Run one calibrated-INT4-style reference ONLY IF time permits (do not
  block submission on it; INT8@2× remains the matched-ratio comparator).
- Update report tables + abstract (§4) with the measured numbers.

### 3.2 If compression stays NO-GO (likely)
- Final claim (pre-registered demotion): **"the certificate stack is the
  contribution"** — certified compressed models (any ratio), measured
  degradation predictor, honest three-arm safety estimation, runtime
  guard enforcement, full reproducibility. INT8 remains accuracy leader at
  2× — stated plainly in the report — and cannot produce any of the above.
- Do NOT soften or spin this in the report; the scoring rubric rewards
  executed kill criteria.

---

## 4. Report phase (CPU-only, after claim assembly)

All in `docs/technical-report.tex` (sections verified to exist):
1. **Results Framework (sec at L609):** replace pre-registered targets with
   the measured table from §1 + Job results. Keep the FT-anchor sentence
   ("swap path reproduces the reference to +0.0000 over 6,496 batches").
2. **Certificate Stack (L186):** L2b subsection — add the measured
   degradation curve + predictor (r*, LOO R², spectral-deviation figure
   data from `results/lyapunov/`). L3 — replace "[pending]" with the
   measured three-arm table + conformal coverage + scenario-opt eps lines
   (artifacts in `results/safety/`). Add the honest scoping note for the
   GEV arm (no RESTART trajectory access on seed→rho oracles).
3. **Safety Evaluation Engine (L368):** add the guard-demo transcript
   (1 SERVE / 4 REFUSE) as the enforcement subsection evidence.
4. **Limitations (L722):** add backbone/dataset specificity (flat spectra),
   single-suite evaluation, simulated-IQAE scoping, GEV scoping, seeds 1–2
   status. Future work: second backbone, closed-loop Lyapunov rollouts,
   int4 comparator, hardware run.
5. **Abstract/Intro:** one-sentence honest claim per §3.
6. Rebuild PDF; every number must trace to an artifact (spot-check 3).

## 5. Submission mechanics (final)
- `PYTHONPATH=python .venv312/Scripts/python.exe -m pytest tests/ -q` → all pass.
- `PYTHONPATH=python:. .venv312/Scripts/python.exe -m bench.all --rows=smoke`
  from a fresh shell → no crashes (clean-env gate N14).
- `git status` clean; push; tag `submission-rc1`.
- README: claims section = §3 outcome; artifact map (which file proves what).
- EXPERIMENT-LOG.md: every job above has an entry with run dir/artifact.

## 6. Explicitly OUT of scope for this submission (do not start)
C++/CUDA-Q kernel parity run, Julia SOS bridge, second backbone training,
RESTART trajectory collection, hardware deployment. All documented as
future work. (Exception: if Q22/Julia pin already exists on the machine and
SOS boxes run in <1 h, they may be added to Layer 2a — otherwise no.)
