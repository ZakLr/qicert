# HANDOFF.md — read this first if you are a new model/session taking over

**Last verified:** 2026-09-13, commit `5400b7f` on `main` (pushed).
**Machine:** Windows laptop, RTX 5060 8 GB, Git Bash. Python: `.venv312/Scripts/python.exe`.
**Rule zero:** every number below was read directly from `results/*/run.json` on this
date. Do not trust numbers from chat history — re-read the artifacts.

---

## 1. What this project is (one paragraph)

`qicert` is a submission to The Quantum Insider's Global Quantum + AI Challenge
(Volkswagen problem statement): make a compressed VLA (vision-language-action) model
deployable with **provable safety guarantees**. Our QI (quantum-inspired) component is
**tensor-train (TT) structure** on linear layers, which yields *exact per-layer
Lipschitz bounds* computable from the compressed cores — a property INT8 quantization
structurally cannot provide. The pipeline: fine-tune baseline (N1/N1v2) → TT-compress
+ closed-form residual repair (N2R) → certificates (soundness per layer) → degradation
predictor (N5) → calibrated INT8 honest comparator (N8C) → compiler/latency (N7) →
report. Local model: MiniVLA (prism-qwen25 0.5B) fine-tuned on LIBERO-spatial.
All training/eval runs on the local GPU only (user explicitly dropped Kaggle).

## 2. Current status — what is PROVEN and what is OPEN

### ✅ Proven and measured (full-split = 6496 batches / 114,497 tokens, seed 0)

| Experiment | Result | Meaning |
|---|---|---|
| **N1v2 FT baseline** | eval acc **0.4468** | the reference everything is compared to |
| **N8C calibrated INT8** | ratio **1.997×**, acc **0.4466** (Δ −0.0001), sound n/a | the honest classical comparator — essentially lossless at ~2× |
| **N2R2 weighted TT+residual** | ratio **2.54–2.86×**, acc **0.1639–0.1640**, certificates sound **168/168 layers** | compression >2× works structurally; accuracy collapses |
| **N8C naive (no calibration) INT8** | acc **0.0000** | earlier "INT8 fails" was a calibration artifact, not an INT8 property — do NOT cite the 0.000 as an INT8 result |
| N7 compiler | identity exact to ~2.3e-14, measured r=0.975 predicted-vs-pruned cert | ablation column is measured, not asserted |
| Latency (CPU edge profile) | all paths < 100 ms | challenge §latency satisfied |

**Pre-registered gate for the accuracy+compression claim:** ratio ≥ 2.0 AND
acc ≥ 0.3968 (= FT − 0.05). **Current verdict: NO-GO on accuracy** — the weighted
repair did not rescue accuracy at 2.5–2.9×. The certificate claim is unaffected and
sound on every compressed layer.

### ⚠️ Open items (what's actually left)

1. **N2R2 configuration search + confirm** — the live experiment. Search over
   plan/residual-rank/mixed/repair-stat configs with a cheap fixed-budget eval
   (40 batches), then confirm the best candidate on the full protocol. This decides
   whether ANY configuration reaches acc ≥ 0.3968 at ratio ≥ 2. If NO-GO again, the
   submission leans fully on the safety/certificate story + honest degradation
   characterization (challenge §5.5 explicitly does not penalize degradation that is
   "clearly characterized" — we characterize it with curves + predictor).
2. **N5 degradation curve + predictor** (stage `n5-plain`, `n5-weighted`,
   `n5-predictor`) — predicts max safe compression from cheap probes. These stages
   crashed on launch last session; the crash (`harness._split_stream` NoneType +
   `dense_params` KeyError) is now fixed, but **no N5 results exist yet**.
3. **N1v2 baseline seeds 1 & 2** — challenge requires ≥3 independent runs, mean ± std.
   Only seed 0 exists. Stage `n1v2` (~1–2 h on the 5060) trains seeds 1,2 (batch 2,
   1200 steps). A prior bug (−-seeds never wired → seed1.pt was a byte-duplicate of
   seed0.pt) is FIXED in code; the duplicate checkpoints are detected by hash and re-run.

## 3. How to run things (commands that WORK — validated 2026-09-13)

```bash
cd "/c/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert"

# See stages:
.venv312/Scripts/python.exe scripts/run_remaining_experiments.py --list

# Dry run (no GPU work):
.venv312/Scripts/python.exe scripts/run_remaining_experiments.py --no-tests --dry-run

# JUST the N2R2 search + confirm (user's current focus, ~4 h):
.venv312/Scripts/python.exe scripts/run_remaining_experiments.py --only n2r2-search,n2r2-confirm

# Everything remaining, sequential, resumable (overnight):
.venv312/Scripts/python.exe scripts/run_remaining_experiments.py
```

Runner properties: runs stages one at a time; full pytest preflight (80 tests, ~3 min);
GPU-busy check; per-stage compile+import double-check; live timestamped logs in
`results/logs/queue/<stage>.log`; **auto-resume — completed stages skip on re-run**;
gates: `n2r2-confirm` only runs if search picked a candidate; `n2r2-weighted`/`mixed`
fallback stages only run if no search results exist; writes
`results/QUEUE-SUMMARY.json` at the end. Times MEASURED on the RTX 5060 (supersede
the stale T4 estimates): search+confirm ~1.5 h total, both N5 stages ~1.5 h, baseline
seeds 1+2 ~1–2 h (train ~8.5 min + full eval ~20.5 min per seed). Run one thing at a
time — two concurrent GPU jobs froze this machine before.

## 4. Bugs fixed in this session (do not re-introduce)

1. **Runner SyntaxError** — duplicated `n2r2-confirm` dict fragment (broken-session
   edit). Fixed; `--list`/`--dry-run` clean.
2. **`bench/n2r_search.py` kernel backend** — search stage had `k = None`, confirm had
   `k` undefined; both called `k.tt_svd(...)`. Both now use
   `from qicert.kernels import get_backend; k = get_backend()` like every sibling
   module.
3. **Pixel device mismatch (THREE files)** — eval loops guarded pixel conversion with
   `hasattr(harness, "_to_half_cuda")`, but `_to_half_cuda` is a MODULE-level function
   in `n2_sweep.py`, not a harness method → guard always False → CPU/None pixels to
   CUDA model. Fixed in `n2r_search.py` (both eval loops) AND `lyapunov_curve.py`
   (N5 — this was the latent twin of the N2R bug, would have crashed n5-plain's
   ref-pred collection). Also removed a duplicated `ref_cache` assignment in N5.
4. **Verdict/resume helpers read phantom `run.json["config"]`** — RunRecorder schema is:
   config lives in sibling `config.json` (`{"config": {...}}`) and its fields are
   MIRRORED inside `run.json["results"]` (e.g. `results.fit_mode`, `results.ratio`,
   `results.eval_acc`). `run.json` has NO top-level `config`. This is why
   QUEUE-SUMMARY showed `best: null` despite three completed runs.
5. **Stale stage time estimates** — all `expect_min` values assumed Kaggle T4s; on the
   5060 the real numbers (from your queue logs) are 3–8× smaller: calib ~3 min,
   one full N2R2 point ~27 min (11.5 compress + 15 eval), search ~1 h (4–5 compress
   passes + seconds-eval each), N5-plain ~45 min (3 plans), N5-weighted ~30 min
   (2 plans), n1v2 seeds 1+2 ~1–2 h. Updated in the runner docstring + STAGES.
6. Earlier session (already committed at `6b47326`): `--seeds` was parsed but never
   wired in `bench/all.py` (silent duplicate seeds); `lyapunov_curve._collect_preds`
   called `harness._split_stream` when None; `n2r2_sweep._record_n2r2_point` had a
   `ttsplit` NameError.

## 5. Artifact map (where the evidence lives)

- `results/N1v2-ckpt/seed0.pt` (+ `baseline_seed0.json`, `eval_batch_seed0.pt`) — FT
  baseline handoff. **All N2/N5 stages require this file to exist.**
- `results/N2R2/<run_id>*/` — completed compression runs. `run.json` carries
  `results.eval_acc`, `results.ratio`, `results.eval_scope` ("full-split (6496
  batches, 114497 tokens)" = the real protocol), `results.cert_sound_layers`.
- `results/N8C/7d97bc0c8ae9_calibrated-int8-mse-seed0/` — the calibrated-INT8 honest
  comparator row.
- `results/N2R/calib_seed0.npz` — activation calibration stats (includes
  `mean_abs`; regenerated by stage `calib-topup`).
- `results/N2R-search/` — search candidates (created by `n2r2-search`; each has
  `run.json` with `results.search_acc`, `results.provisional_go`, and
  `results.config` = the candidate config).
- `results/lyapunov/degradation_seed0.json`, `predictor.json` — N5 outputs (do not
  exist yet).
- `results/EXPERIMENT-LOG.md` — chronological what/why/result per experiment. APPEND
  a dated entry for every new run; never put a number in a report without a run dir.
- `results/ledger.csv` — every run's row (config_hash keyed).
- `N2R-SEARCH-CONFIG.md` — the search design (grid, honest-speedup rules, selection
  rule, report wording). `EXECUTION-PLAN.md` — master plan. `SESSION-CHECKPOINT.json`
  — a 2026-09-12 snapshot (pre-fix; superseded by this file for context, still useful
  for the exact pre-fix commit `6b47326`).

## 6. Environment gotchas (learned the hard way)

- Windows + Git Bash. Always `cd` into the repo; use forward slashes in bash.
- `PYTHONPATH` needs BOTH `python` and repo root (`PYTHONPATH="python;."` — semicolon
  on Windows). The runner sets this itself; manual `-m bench.all` invocations must too.
- The runner resolves `PY = sys.executable`, so it uses whatever python launches it —
  use `.venv312/Scripts/python.exe` to launch the runner.
- VLA weights are local: `PRISMATIC_DATA_ROOT` = `qicert/weights/data`; checkpoint at
  `weights/ckpt/checkpoints/step-122500-epoch-55-loss=0.0743.pt`. Loading takes
  ~1.5 min per invocation (oneDNN/timm noise in logs is normal).
- FA2 unavailable → SDPA fallback (recorded as env deviation per AGENTS.md §7).
- 8 GB VRAM: batch 2 max, one job at a time. Peak ~4.8 GiB for FT training.
- Do NOT run two experiments concurrently — froze the machine once already.

## 7. The narrative (what the report claims, and does NOT claim)

**Claim 1 (proven, exclusive):** TT structure gives exact per-layer Lipschitz bounds
from the compressed cores, chained into a certified output-perturbation ball, ENFORCED
at runtime by a guard that refuses out-of-ball actions (Z3-falsified checker, guard
demo transcript). Sound on 168/168 layers in all compression runs. INT8 cannot state
this kind of bound.

**Claim 2 (measured, honest):** calibrated INT8 is ~lossless at 2× and WINS accuracy
at that ratio; our TT+residual at 2.5–2.9× is certificate-sound but accuracy-collapsed
in the first weighted configuration. We report that plainly (challenge §5.5 rewards
honest characterization). The search/confirm decides whether any config closes the
gap at >2×.

**Claim 3 (design tool):** N5 degradation curve + predictor turns "post-hoc luck"
into "predict the max safe compression from cheap probes" — pending the N5 stages.

**Never claim:** beating INT8 on accuracy at 2×; quantum-hardware speedup; hardware
readiness from noiseless/sim results. Every reported number must resolve to a run dir
in `results/` (AGENTS.md rules).

## 8. Remaining work after N2R2 search+confirm + N5 (submission checklist)

1. Baseline seeds 1–2 (`n1v2` stage, ~20 h) → report mean ± std over 3 seeds.
2. Update `docs/technical-report.tex` results tables with the final measured rows
   (search/confirm verdict, N5 curve + predictor, 3-seed baseline mean ± std).
   Report MUST keep the two-column framing: accuracy vs certificates.
3. If search is NO-GO: strengthen the "clearly characterized degradation" framing
   (§5.5) and the degradation-predictor as the practical contribution; the honest
   sentence is already drafted in `N2R-SEARCH-CONFIG.md`.
4. Cross-track + energy + resource tables: mostly computed; verify against ledger.
5. Reproducibility pass: fresh-venv smoke (`qicert` smoke rows), README quickstart,
   pinned env. Challenge requires public repo + clean-env repro.
6. Final conformity pass against `submission/Phase-1-Submission-Guidelines.md` and
   the challenge statement (the organizer docs, NOT the `submission/` folder drafts,
   which are our own old planning docs — user said to ignore those as references).
7. User instruction: never attribute commits to Codebuff (no Co-Authored-By footer).
