# HANDOFF.md — read this first if you are a new model/session taking over

**Last verified:** 2026-09-13 ~17:30 UTC, commit after `56169dc` on `main`.
**Docs current as of this date:** `docs/technical-report-v2.pdf` (8 pp, 0 errors,
0 overfull; search verdict + mode-collapse diagnosis folded into Limitations &
Expected Impact) and `docs/concept-proposal.pdf` (3 pp, validation plan updated:
Phase-1 capability experiment completed NO-GO, design-tool claim re-scoped to a
Phase-2 collapse-boundary detector). `PHASE2-CASE.md` (new) holds the full
failure analysis: how the pre-registered bar was set (FT 0.4468 − 0.05), the
three untested levers, the costed arm ladder, and the certificate-impact of
repair training (re-derived from final weights, one SVD/layer, ~2% ratio cost).
**Final remaining GPU experiment:** baseline seeds 1–2 (`--only n1v2`, ~1–2 h).
**New experiments ready (added this session, validated by compile+tests only —
not yet run on GPU):** `n9-repair` (post-compression LoRA repair training on the
already-compressed seed-0 model, ~1.5 h) and `n10-scale` (Qwen2.5-1.5B
LLM-backbone compression scale probe, ~3–4 h, first launch downloads ~3 GB
weights from HF). See §3 for commands and `PHASE2-CASE.md` §4 for the design.
**Session rule:** update this file at the end of every work session (and mention
it in `README.md`) so the next model/session starts here, not from chat history.
**Live now:** user ran the full queue solo 13:22–15:10 UTC — COMPLETE.
Summary `results/QUEUE-SUMMARY.json` (15:10:51 UTC): N2R2 NO-GO (best
full-split 0.1640 @2.536x), N5 5 points + predictor present, seeds 1–2 still
missing. Prior 13 N2R2 sweep dirs remain archived in
`results/_archive_20260913_meansearch/` with `MANIFEST.json`.
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

1. **N2R2 confirm stage** — the live experiment. The configuration search
   COMPLETED 2026-09-13 (5 candidates, 50 min, 840/840 certificate-sound; rows in
   `results/N2R2/*search-*`; table in `results/EXPERIMENT-LOG.md`). No candidate
   reached the provisional GO bar (0.30); best search accuracy 0.122 at 4.07x
   (INVALID mixed row) and 0.115 at 2.54x (valid weighted). The queue's selection
   rule picked frac0.33/rr32/weighted/mixed for the full-protocol confirm.
   NOTE: the search's mixed rows are INVALID (kept=0 bug below); the confirm runs
   with repaired mixed semantics, so its true ratio will be lower (~2.3x) and
   accuracy possibly higher. Decide only on the confirm row.
   Also flagged: the earlier full-split run `45a8d6f5...weighted-mixed` is invalid
   as mixed evidence (identical ratio/acc to its non-mixed twin) but remains a
   valid plain weighted point.
   **UPDATE 2026-09-13 ~13:00 UTC:** that whole mean-stat sweep is now archived —
   all 13 N2R2 dirs (10 mean-stat search/search-prefix rows + 3 full-split absmax
   rows incl. the INVALID-mixed one, renamed with an `INVALID-mixed-kept0`
   prefix) moved to `results/_archive_20260913_meansearch/N2R2/` with
   `MANIFEST.json` (sha + per-row summary); the 5 empty `results/N2R-search/`
   stubs moved alongside and the folder removed. Proven numbers are untouched in
   `results/ledger.csv`, `results/EXPERIMENT-LOG.md`, and the report. A fresh
   queue was launched (`--only
   n2r2-search,n2r2-confirm,n2r2-weighted,n2r2-mixed,n5-plain,n5-weighted,n5-predictor
   --exclude calib`, PID 792): the new search runs with `STAT=absmax` (current
   runner default — the user's pasted skip-log showing `mean` candidates is from
   the pre-fix code) and repaired mixed semantics, so expect higher search acc
   and a different confirm target. The pasted `--only n2r2-search,n2r2-confirm`
   run did zero work — both stages reported "already complete" and only
   regenerated `QUEUE-SUMMARY.json`.
   **CRASH 2026-09-13 ~13:10 UTC (no result invalidated):** the background queue
   above and a second foreground queue the user started at 13:06 ran
   CONCURRENTLY (two VLA loads + two full test suites) and both died in
   `n2r2-search` at the same candidate with `_ArrayMemoryError: Unable to
   allocate 4.84 MiB ... float64` in `tt_matvec` (`operator_norm_tight` via
   `layer_report`) followed by `rc=3221225477`. A 5 MB alloc failing means host
   RAM was exhausted by contention, not by the N2R2 fix — the kernels are
   byte-identical to the successful 11:41 search. `results/N2R2/` holds no
   partial dirs (failure precedes recording); archive + ledger untouched.
   Recovery: run ONE queue at a time (machine rule, §6), confirm
   `nvidia-smi` is idle first, then rerun the same command. Note: the
   "configuration plan" printout shows `mean` because it reads the runner's
   base env before stages; the actual `n2r2-search` stage env sets
   `STAT=absmax` — verify in the stage log header, not the plan print.
   **FRESH RUN 2026-09-13 15:10 UTC (post-archive, solo, all-absmax):**
   search 5/5 completed, every row `cfg_stat=absmax` (the stat fix verified in
   `config.json`); mixed rows now genuine (`kept=96`, `sound=72/72`, ratios
   2.35–2.99x instead of the invalid 2.86–4.07x). Search acc: weighted
   rr32/rr64 @0.50 → 0.1222/0.1222 (mean-stat was 0.0312/0.1151 — absmax
   confirmed ~4x better at rr32), mixed 0.1222/0.0781/0.0994; nothing near the
   0.30 provisional bar. Confirm re-ran weighted rr64/0.50 full-split →
   **0.1640 @2.536x, exact replication of the archived 0.1640** (deterministic
   pipeline). Mixed at matched ratio scores LOWER than non-mixed (0.078–0.122
   vs 0.122 search-prefix) — keeping attention layers full-precision does not
   rescue accuracy; the collapse is backbone-wide, not attention-localized.
   Verdict unchanged: **NO-GO** (needs ≥0.3968). N5: 5 points collected but the
   agreement curve is NON-MONOTONIC (plain 0.50→0.555, 0.33→0.190,
   0.25→0.580; weighted 0.50→0.109, 0.33→0.580) so the linear predictor is
   vacuous (`R²=0.16`, `LOO R²=-1.21`, safe-ratio `NaN`) — the "design tool"
   claim cannot ship on this evidence. Audit flags: (a) each search candidate
   is recorded TWICE under mirrored tags (`search-0_*` + `n2r2-weighted[-mixed]*`,
   identical numbers — bookkeeping quirk, inflates ledger rows, changes no
   number); (b) RESOLVED 2026-09-13 post-queue: the exact equality (0.57994 = 584/1007 on
   both `weighted-0.33` and `plain-0.25`) is NOT a ref-cache coincidence — it is
   **mode collapse to the modal action token**. The reference stream's modal
   token `151515` occupies exactly 584/1007 positions; a collapsed model that
   emits only that token agrees on exactly those. Same signature in the search:
   three distinct configs all scored exactly 86/704 = 0.1221590909… (the count
   of ground-truth modal tokens in the fixed search prefix). All deep-compression
   accuracies sit at the modal base rate: the models collapsed, they did not
   gracefully degrade. Full analysis in `results/EXPERIMENT-LOG.md`
   (2026-09-13 post-queue entry).
2. **N5 degradation curve + predictor** — DONE 2026-09-13: 5 points
   (`results/lyapunov/degradation_seed0.json`) + predictor
   (`results/lyapunov/predictor.json`). Verdict: the linear predictor is
   VACUOUS (R²=0.16, LOO R²=−1.21, safe-ratio NaN) and that is now understood
   to be the honest outcome — the agreement curve is bimodal because deep
   compression causes mode collapse (see audit flag (b) above), which a linear
   model cannot fit by construction. Do NOT ship the predictor as a design
   tool; ship the curve + collapse mechanism as the degradation analysis.
   Caveat: the N5 protocol's break condition caps the probe at ~1007 tokens
   (token-floor `n_batches*2` triggers after ~57 real batches of the 500
   requested) — small-sample agreement, fine for collapse detection, too
   small for precise partial-degradation estimates.
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

# JUST the N2R2 search + confirm (COMPLETED 2026-09-13 - kept for reference):
.venv312/Scripts/python.exe scripts/run_remaining_experiments.py --only n2r2-search,n2r2-confirm

# N9 repair training (NEW, validated, not yet run - ~1.5 h GPU):
#   compress (N2R2-confirmed frac0.50/rr64/absmax) -> LoRA r=8 x 1200 steps
#   on TRAIN-ONLY episodes -> prefix evals -> saved repaired checkpoint.
#   Smoke first (~10 min): --only n9-smoke
.venv312/Scripts/python.exe scripts/run_remaining_experiments.py --only n9-repair

# N10 scale probe (NEW, validated, not yet run - ~1.5-2 h CPU, no GPU):
#   Qwen2.5-1.5B backbone, same TT+residual machinery, recon/spectrum/
#   certificates at Phase-1 ratios; first launch downloads ~3 GB from HF.
#   Smoke first (one ratio, 8 layers): --only n10-smoke
.venv312/Scripts/python.exe scripts/run_remaining_experiments.py --only n10-scale

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
8. **Mixed-allocation no-op (2026-09-13)** — `_llm_linear_keys` yields dashed
   layer types (`self_attn-q_proj`) while `keep_types` held bare `q_proj`, so
   `layer_type in keep_types` was always False and every "mixed" run kept 0
   layers, silently duplicating the non-mixed arm. Fixed with `_is_kept()`
   (suffix match) in `n2r2_sweep` + both `n2r_search` stages. All mixed rows
   produced before this fix are invalid as mixed evidence.
9. **Candidate selector scanned the wrong directory** — search rows are recorded
   under `results/N2R2/` (stage shares `--exp-id N2R2`), not `results/N2R-search/`;
   selection now identifies search rows by their `search_acc` field, which also
   prevents confirm rows being mistaken for candidates.
10. Earlier session (already committed at `6b47326`): `--seeds` was parsed but never
   wired in `bench/all.py` (silent duplicate seeds); `lyapunov_curve._collect_preds`
   called `harness._split_stream` when None; `n2r2_sweep._record_n2r2_point` had a
   `ttsplit` NameError.
7. **First search launch failed silently (2026-09-13 10:27)** — n2r_search's lazy
   `from .compress_residual import ...` resolved against the *bench* package (that
   module lives at `python/qicert/compress_residual.py`), so ALL 5 candidates died
   with ModuleNotFoundError while the stage exited rc=0 and the queue reported
   COMPLETE. Fixed: imports hoisted to module top level (preflight now catches),
   `done()` for n2r2-search requires a selectable candidate, and main() hard-fails
   the queue when the search produced none. ALSO: the ref-pred cache break counted
   TOKENS (`total >= n_batches*2`) instead of batches — the "40-batch" cache held
   89 tokens (~5 batches) and did not cover the candidate prefix; now counts
   batches, matching `_score_candidate`. (`_split_stream` is a lambda building a
   fresh stream per call, so repeated iteration across candidates is safe.)

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
  `results.config` = the candidate config). Currently absent (archived; the
  fresh search writes here again under `--exp-id N2R2` rows).
- `results/_archive_20260913_meansearch/` — pre-rerun archive (13 N2R2 dirs +
  5 empty N2R-search stubs + `MANIFEST.json`). Read-only; never feed back into
  the queue globs.
- `results/logs/queue/manual-rerun-20260913.log` — live log of the fresh queue
  (PID 792); per-stage logs in `results/logs/queue/<stage>.log`.
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
8. **N9 design (2026-09-13):** compresses the seed-0 FT weights (NOT the base
   checkpoint - N2R2 convention) with the confirmed config, evals the collapsed
   model on a fixed 600-batch prefix, trains LoRA r=8 on train-only episodes
   (`LocalEpisodeStream(episode_ids=...)` filter added to vla_local.py,
   backward-compatible), MERGES the adapter before the post-eval, and saves
   `results/N9-repaired/repaired_seed0.pt` for a full-protocol confirm.
   Pre/post numbers are PREFIX evidence, never reportable as final.
9. **N10 design (2026-09-13):** NO pretrained 1.5B VLA checkpoint exists
   publicly (verified by web search 2026-09-13) - N10 is therefore an honest
   WEIGHT-SPACE probe of the bare Qwen2.5-1.5B backbone (uniform weighting,
   no activations exist), measuring recon error / spectral decay / cert
   soundness at the Phase-1 ratios. Its result supports or refutes the scale
   lever (PHASE2-CASE.md L3) but supports NO task-accuracy claim. Bare-HF
   state dicts need `_bare_llm_linear_keys` (n10_scale.py), not
   `_llm_linear_keys` (which requires the `llm.` prism prefix).
