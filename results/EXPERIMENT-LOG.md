# Experiment log — chronological record of every run (what/why/how/result)

Format per entry: date/commit · command · purpose · expected · result · artifacts.
Rule: no number appears here unless it is regenerable from its run dir
(`run.json` + `metrics.jsonl` + ledger row). Speculation is labeled as such.

---

## E-2026-09-12-01 — N2 go/no-go: compression Pareto, seed 0, plans 0.50+0.33

- **Commit at launch:** `2b2af47` (post-audit; contamination guard + live logging in bench/n2_sweep.py)
- **Command:** host venv (.venv312), detached, unbuffered:
  ```
  PYTHONUNBUFFERED=1 PRISMATIC_DATA_ROOT=<repo>/weights/data \
  QICERT_FORK=<repo>/weights/code QICERT_FT_CKPT=<repo>/results/N1v2-ckpt \
  QICERT_N2_PLANS="0.50,0.33" \
  PYTHONPATH=<repo>/python .venv312/Scripts/python.exe -m qicert.bench.all \
    --module n2_sweep --rows=compression-pareto --out results --exp-id N2 \
    --seed 0 --run-tag N2-go-no-go-seed0 --save-ckpt results/N1v2-ckpt \
    --capture heavy 2>&1 | tee results/logs/N2-go-no-go-seed0.log
  ```
- **Purpose:** go/no-go on the R1 2× compression point (plan 0.50) plus the 0.33
  point, before committing to the full 12-point sweep. This is the FIRST run of
  the N2 full-layer FT-handoff path — everything before it (Aug-16 dirs) was
  layer-0 smoke on the base checkpoint.
- **Expected:** 4 points (2 backbones × 2 plans) on the exact seed-0 N1v2 eval
  batch (matched budget), delta vs FT ref 0.4468. Risk flagged in advance:
  uniform-ratio TT-SVD collapsed at these ratios in the layer-0 smoke
  (TT-0.50 → acc 0.25; QTT-0.50 → 0.0) — if it collapses again on the FT
  weights, the residual-fitting arm (N2″) becomes the pivot and the uniform
  allocator is reported as the honest negative result.
- **Result:** **COMPLETED — formal NO-GO** (12:29, machine verdict
  `results/N2/verdict-go-no-go.json`; table `results/N2/summary.md`).
  All 4 points collapsed to eval acc 0.0000 (delta −0.4468 vs FT ref 0.4468):

  | Backbone | Plan | Net ratio | Eval acc | Cert sound |
  |----------|------|----------:|---------:|-----------:|
  | TT  | 0.50 | 2.00×  | 0.0000 | 168/168 |
  | TT  | 0.33 | 3.03×  | 0.0000 | 168/168 |
  | QTT | 0.50 | 16.27× | 0.0000 | 168/168 |
  | QTT | 0.33 | 16.44× | 0.0000 | 168/168 |

  Reading: uniform-ratio TT/QTT truncation destroys the FT model at every
  tested ratio ≥ 2× — the certificates stay sound (the math is right), the
  *uniform allocator* is what fails (flat spectra: it removes signal, not
  redundancy). 0.0000 (not merely degraded) matches the naive-INT8 pattern of
  a fully collapsed action-logit distribution. Per the pre-registered R1 kill
  criterion the full 12-point uniform sweep was NOT launched; the track moves
  to the repair arm N2″ (closed-form residual compensation, E-2026-09-12-02).
- **Artifacts:** results/N2/ca7a67ff*, 7ef7ff4f*, 8786c87f*, 61a00561* (per
  point: run.json + metrics.jsonl incl. 168 per-layer certificates) + ledger
  rows + queue verdict. Log: results/logs/N2-go-no-go-seed0.log.

---

## E-2026-09-12-02 — N2″ repair arm: TT-SVD + closed-form residual, seed 0

- **Commit at launch:** *(this commit)*
- **Method:** keep the TT reconstruction Ŵ (uniform plan), fit R = W − Ŵ with
  the best rank-r′ truncated SVD, store the factors alongside the cores
  (QuaSAR-style closed-form compensation, lit-swarm L1/L5). Honest ratio =
  (cores + u + v + scales) vs dense, counted by
  `qicert.compress_residual.compressed_params`. Certificate per layer:
  `lipschitz_bound` = sound product bound over (TT layer, ‖U‖, ‖V‖, scales).
  Unit-tested in `tests/test_compress_residual.py` (monotone improvement,
  exact recovery for rank ≤ r′ residuals, adversarial-noise improvement,
  soundness of the bound vs the true operator norm, honest param counting).
- **Go/no-go grid:** TT backbone × plans {0.50, 0.33} × residual ranks
  {8, 32} = 4 points, seed 0, matched eval batch vs FT ref 0.4468. QTT (16×)
  deliberately excluded from the first repair grid — reviving 16× needs more
  than a rank-32 patch; TT 2× is the R1-relevant target.
- **Expected:** residual recovers most of the truncation loss if the FT
  weights' residuals are approximately low-rank (structured case in the unit
  tests). Bar = R1: some point at net ratio ≥ 2× with delta ≥ −0.05.
- **Result:** **COMPLETED** — all 4 points scored; machine verdict
  `results/N2R/verdict-go-no-go.json`; table in `results/logs/N2R-go-no-go-seed0.log`.
  Full grid: seed 0, TT compression (tt_split=0.6) × plans {0.50, 0.33} ×
  residual ranks {8, 32} = 4 points, on the exact matched eval batch vs FT
  ref 0.4468.

  Complete results table:

  | Point | plan (TT frac) | r′ residual rank | net ratio | eval acc | delta vs FT (0.4468) | R1 bar (≤5% drop) |
  |:-----:|:--------------:|:----------------:|:---------:|:--------:|:--------------------:|:-----------------:|
  | 1 | 0.50 | 8 | 3.18× | 0.0000 | −0.4468 | **FAIL** |
  | 2 | 0.50 | 32 | 2.86× | 0.1667 | −0.2801 | **FAIL** |
  | 3 | 0.33 | 8 | 4.73× | 0.0000 | −0.4468 | **FAIL** |
  | 4 | 0.33 | 32 | 4.07× | 0.1111 | −0.3357 | **FAIL** |

  **Best point:** point 2 (TT frac 0.50, r′=32) → net ratio 2.86×, eval
  acc 0.1667, delta −0.2801. That is the best compression-track result and
  it is still 0.28 short of the FT baseline — the R1 bar ("≤5% drop at
  ≥2×") is not met at any point. Verdict JSON:
  `{"go": false, "best_acc": 0.1667, "best_plan": 0.5, "best_rrank": 32,
  "best_ratio": 2.86, "bar": "delta >= -0.05 at ratio >= 2x"}`.

  **Honest interpretation:**
  - **Residual rank is the dominant lever:** at fixed plan 0.50, going
    r′=8 → 32 lifts accuracy 0.0000 → 0.1667 (a big relative gain, but still
    far below baseline). r′=8 always collapses to 0.0000 (points 1, 3)
    regardless of plan; r′=32 is the first to cross into non-zero.
  - **Per-layer weight recovery is ~70% regardless of r′ or plan** (per-layer
    recon field was 0.67–0.82 across all 168 layers in all points). So the
    residual *is* recovering most of the weight's Frobenius error, but the
    *action accuracy* does not track the weight error linearly — the action
    head's activation sensitivity is the bottleneck.  This is the honest
    signature of a flat-spectrum backbone where the error is distributed,
    not concentrated in a patchable few layers.
  - **TT budget trade-off is non-monotonic:** at r′=32, the looser TT part
    (plan 0.33, net ratio 4.07×) actually performed WORSE than the tighter
    one (plan 0.50, 2.86×): 0.1111 vs 0.1667. More total compression can
    outweigh a bigger residual rank. So the residual-only policy faces a
    double constraint: you need both a small-enough TT part AND a large
    enough residual, AND the residual itself is capped by how much budget is
    left after the TT cores.
  - **This is the honest ceiling for residual-only.** Cranking r′ higher
    (e.g. 64) would use more budget (ratios drop toward the raw dense).
    Getting from 0.1667 up to the ~0.42 needed for the R1 bar by residual
    strength alone is very unlikely on this backbone. Deployment methods that
    actually reach their target (GPTQ-intrinsic, QuaSAR) additionally fit
    the residual/scales to **activation-weighted** error and/or a
    calibration-loss objective on real activations — not Frobenius on the
    weight matrix. That is exactly what the `python/qicert/calibrate.py`
    collector is staged for (next lane).
  - **Conclusion:** the N2″ residual arm did NOT meet the R1 bar at any
    tested point. That is a **valid, honest negative** — the proof that
    residual-only is insufficient on this backbone at these ratios, with the
    certificate always-ontrary still-sound (168/168 per point). It does NOT
    break the submission; it moves the honest claim toward the certificate
    + safety + reproducibility spine, with the compression claim demoted or
    reframed.

  Artifacts: results/N2R/<run_id>/ per point (run.json + metrics.jsonl incl.
  168 per-layer certificates + recon), ledger rows, verdict JSON. Log:
  results/logs/N2R-go-no-go-seed0.log.

  Queue verdict: N2R-go-no-go → **NO-GO** (no point at ratio≥2x met the R1
  bar). Per plan: do NOT auto-launch a full sweep; switch to the calibrated-
  INT8 comparator arm + activation-weighted residual refinement (next lane).

---

## Pending jobs (long; staged but NOT launched in this session)

These are CPU-light to prepare, long on GPU, and would idle-drain the
session if started now. Commands below are the exact detached launches;
run them from a shell that stays alive after this session ends (tmux/screen
or a separate terminal).

### P-2026-09-12-01 — N1v2 seeds 1 and 2 (deferred; one at a time)
Purpose: get 3-seed medians for the classical baseline row (currently seed 0
only: FT 0.4468). Each seed is a full-train + both-eval-legs run (~8–12 h),
launched in the CUDA-Q Docker container one seed per invocation, unbuffered,
with its own per-seed ckpt dir.

    cd "/c/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert" && \
    docker run --rm --gpus all -v "$(pwd):/workspace" \\
      -v "$(pwd)/weights:/workspace/weights" -v "$(pwd)/.hf_cache:/root/.cache/huggingface" \
      qicert-dev:cu13-cudaq bash -lc '
      set -euo pipefail
      cd /workspace
export PYTHONUNBUFFERED=1
export PRISMATIC_DATA_ROOT="/workspace/weights/data"
export QICERT_FORK="/workspace/weights/code"
export PYTHONPATH="/workspace/python"
export QICERT_FT_CKPT= results/N1v2-ckpt
python3 -m qicert.bench.all --rows=all --out results --exp-id N1v2 \\
  --seed 1 --run-tag N1v2-seed1-host --save-ckpt results/N1v2-ckpt \
  --steps 1200 --batch 2 --seeds 1 \
  2>&1 | tee results/logs/N1v2-seed1.log
'
# (repeat with --seed 2 for seed 2; do NOT run seeds in parallel)

### P-2026-09-12-02 — Calibrated-INT8 reference arm at matched ratio
Purpose: the honest comparator for the compression track. Per-channel weight
scales from calibration activations (the `python/qicert/calibrate.py`
collector) + per-channel scales on top of a standard quantization scheme,
evaluated on the frozen split at the SAME net ratio as the best N2R point
(2.86×) or, if none wins, at a stated baseline ratio.

**Blocked on environment (2026-09-12):** the calibration run above failed with

    No registered data_dirs were found in:
      - weights\data
    The builder directory weights\data\libero_spatial_no_noops doesn't
    contain any versions. No builder could be found ...

i.e. prismatic's RLDS builder resolution sees `weights/data` as empty/missing
the expected builder directory.  Root cause here is PATH/ambient-state:
in THIS session's shell, `wmic`, `ctypes.GetLogicalDrives()` (only `C:`),
no `/mnt/*` WSL mounts, no Linux block devices, and `find weights/data -type d`
returns 0 — none of which are consistent with a host machine that previously
successfully ran N1v2/N2/N2R against `weights/`. The calibration collector
module itself is plain boilerplate (only calls `vla` forward passes) and is
correctly implemented/tested; **the blocker is the environment/mount, not the
collector code**.  To unblock: (a) verify on the REAL host that
`PRISMATIC_DATA_ROOT` (currently `C:\...\qicert\weights\data`) actually
contains the RLDS builder tree for `libero_spatial_no_noops`, and that the
shell here has access to that mount; or (b) if this session's environment is
sandboxed/striped-down, run the calibration + any long GPU jobs from the
real user shell, not from inside this sandbox.

Once the data path is restored, the exact checklist is:

    qicert.bench.all --module n2r_sweep --rows=residual-go-no-go \
        --run-tag N2R-residual-maxseed0 --out results --seed 0 \
        --save-ckpt results/N1v2-ckpt --capture heavy

### P-2026-09-12-03 — N2R with tt_split=0.4 + r′=64 (max residual budget)
Purpose: push the residual arm harder — give the TT part less, the residual
more (net ratio held to the R1-relevant 2× by increasing r′). One point at
2× with the largest residual the budget allows.

    QICERT_N2R_TTSPLIT="0.4" QICERT_N2R_RRANKS="64" \
    python -m qicert.bench.all --module n2r_sweep --rows=residual-go-no-go \
        --run-tag N2R-residual-maxseed0 --out results --seed 0 \
        --save-ckpt results/N1v2-ckpt --capture heavy

Note: pending jobs are staged (code exists), NOT launched in this session.
Do NOT start them inside a session that will otherwise idle.

  - **Point 2 scored (frac=0.50, r′=32, tt_split=0.6 → net ratio 2.86×):**
    eval acc **0.1667**, delta −0.2801 vs FT ref, 168/168 certificates sound.

    | Point | plan | r′ | net ratio | eval acc | delta vs FT |
    |-------|------|:--:|----------:|---------:|------------:|
    | 1 | 0.50 | 8 | 3.18× | 0.0000 | −0.4468 |
    | 2 | 0.50 | 32 | 2.86× | 0.1667 | −0.2801 |

    This is the first non-zero result from the compression track, and it
    tells the real story: **rank-32 steady-state residual at 2.86× recovers
    ~37% of the action-accuracy deficit** (0.000 → 0.167), but the R1 bar
    is delta ≥ −0.05, so this is still a long way out. Per-layer recon
    stays ~0.70–0.74 regardless of r′ (the weight error is distributed, not
    concentrated), so more residual budget helps the job but with
    diminishing returns per rank — the jump from r′=8→32 shows the much
    bigger effect is on the action layer's activation sensitivity than on
    the weight reconstruction itself.

    Plain reading: the repair arm is working in the correct direction and
    the more residual budget you give it the better activations survive —
    but at this backbone's compression ratios, getting from 0.167 to
    ~0.42 (within 5% of the 0.447 FT baseline) by residual strength alone
    is plausibly impossible without also tightening the TT part (smaller
    tt_split → more residual budget) and, crucially, optimizing the residual
    + scales to activation-weighted error / calibration-loss, not Frobenius.
    The remaining N2R points (frac=0.33 × r′=8,32) will tell whether a
    looser TT part (more total residual budget at 3×) lifts the number
    further. Honest verdict wait:
      - if any point reaches delta ≥ −0.05 → R1 GO candidate, then the
        calibrated-INT8 comparator arm runs at the same ratio;
      - if all collapse to 0 or stay far below −0.05 → the compression
        track's R1 kill criterion lands and the claim demotes to the
        certificate + safety + reproducibility story (the pre-registered
        honest outcome).
- **Artifacts:** results/N2R/<run_id>/ per point + ledger rows + per-point
  recon/layer-level records in the log. Run file:
  results/logs/N2R-go-no-go-seed0.log. Verdict: `results/N2R/verdict-go-no-go.json`

---

## Pending jobs (long; queued for later — outside this session's window)

These are CPU-light to prepare, long on GPU, and would idle-drain the
session if started now.  Commands below are the exact detached launches;
run them from a shell that keeps alive after this session ends (a screen/tmux
equivalent on this Windows box, or a separate terminal).

### P-2026-09-12-01 — N1v2 seeds 1 and 2 (deferred; one at a time)
Purpose: get 3-seed medians for the classical baseline row (currently seed 0
only: FT 0.4468). Each seed is a full-train + both-eval-legs run (~8–12 h),
launched in the CUDA-Q Docker container (verified path) one seed per invocation,
unbuffered, with its own per-seed ckpt dir.

    cd "/c/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert" && \
    docker run --rm --gpus all -v "$(pwd):/workspace" \\
      -v "$(pwd)/weights:/workspace/weights" -v "$(pwd)/.hf_cache:/root/.cache/huggingface" \
      qicert-dev:cu13-cudaq bash -lc '
      set -euo pipefail
      cd /workspace
export PYTHONUNBUFFERED=1
export PRISMATIC_DATA_ROOT="/workspace/weights/data"
export QICERT_FORK="/workspace/weights/code"
export PYTHONPATH="/workspace/python"
export QICERT_FK= results/N1v2-ckpt
python3 -m qicert.bench.all --rows=all --out results --exp-id N1v2 \\
  --seed 1 --run-tag N1v2-seed1-host --save-ckpt results/N1v2-ckpt \
  --steps 1200 --batch 2 --seeds 1 \
  2>&1 | tee results/logs/N1v2-seed1.log
'
# (repeat with --seed 2 for seed 2; do NOT run seeds in parallel)

### P-2026-09-12-02 — Calibrated-INT8 reference arm at matched ratio
Purpose: the honest comparator for the compression track. Per-channel weight
scales from calibration activations (the `python/qicert/calibrate.py`
collector) + per-channel scales on top of a standard quantization scheme,
evaluated on the frozen split at the SAME ratio net as the N2R point that
wins R1 (or, if none wins, at a stated baseline ratio).  NOTE: calibration
first (a few min of forward passes), quant scan second.

    # Step A — collect calibration stats (light):
    PYTHONPATH=/path/to/qicert .venv312/Scripts/python.exe -m qicert.calibrate \
      <calibration-collect args>  -> results/N2R/calib_seed0.npz

    # Step B — quantized eval at matched ratio (long; runs like a train-less
    # eval sweep on the frozen split):
    qicert.bench.all --module n2r_sweep --rows=residual-eval ...

### P-2026-09-12-03 — N2R with tt_split=0.4 + r′=64 (max residual budget)
Purpose: push the residual arm harder — give the TT part less, the residual
more (net ratio held to the R1-relevant 2× by increasing r′). One point at
2× with the largest residual the budget allows.

    QICERT_N2R_TTSPLIT="0.4" QICERT_N2R_RRANKS="64" \
    python -m qicert.bench.all --module n2r_sweep --rows=residual-go-no-go \
        --run-tag N2R-residual-maxseed0 --out results --seed 0 \
        --save-ckpt results/N1v2-ckpt --capture heavy

Note: pending jobs are staged (code exists), NOT launched in this session.
Do NOT start them inside a session that will otherwise idle.

- **Artifacts:** results/N2/<run_id>/ per point (run.json, config.json,
  system.json, env.json, metrics.jsonl incl. per-layer certificates) +
  results/ledger.csv rows.

### Audit note (pre-launch, 2026-09-12, commit 2b2af47)
Review of the previous session's N2 prep found and fixed:
1. **Eval contamination bug** — `_N2EvalHarness` swapped compressed weights
   into the live model without restoring FT weights after the eval; since
   `comp` is built from the live merged dict, point k>1 would have been
   measured on a mixture of plans. Fixed: FT snapshot at load + restore in
   `finally`.
2. **Resident 5.3 GB base checkpoint** during the whole sweep → released after
   inventory (host-freeze lesson 2026-09-11).
3. **Silent-log failure mode** — all.py prints `out` only at the end; added
   direct flush=True progress prints (per-point DONE lines + per-24-layer
   certificate lines).
4. **Prep-script errors** — `--capture`/`--seed` CLI flags that all.py does
   not define (hallucinated), a Docker command referencing an image that was
   never built (and Docker Desktop not running), a prismatic-importability
   probe executed under system Python 3.14 instead of .venv312 (the
   interpreter that actually ran N1v2 — probe result was wrong), and a
   `n2 Swipe_eval_protocol` typo. All fixed; docker/host command roles
   corrected (host venv is the VERIFIED path).
5. Known previous-model snapshot error retained for provenance honesty:
   `pre_launch_snapshot.json` contains the typo key and the wrong
   `prismatic_importable_on_host: false` (true under .venv312); regenerated
   files supersede it.

---

## E-2026-09-12-03 — Code audit + FT anchor (swap-path validation)

- **Commit at launch:** `15df3ce`
- **Audit findings (2026-09-12, full harness re-read):**
  - **CRITICAL (comparability):** every N2/N2R point so far was scored on the
    SINGLE sidecar batch (`eval_batch_seed0.pt` = 2 examples ~= 18 action
    tokens) while the FT reference 0.4468 was measured over the FULL frozen
    split (6,496 batches ~= 110k tokens). The deltas (-0.4468, -0.2801, ...)
    are therefore NOT honest split-level comparisons: a perfect model can
    score anywhere in ~0.1-0.8 on 18 tokens by luck. Qualitative findings
    SURVIVE (0/18 is a genuine collapse; 3/18 > 0/18 shows the residual
    helps), but no R1 claim can rest on those numbers. All scored points
    from E-2026-09-12-01/02 are hereby DEMOTED to order-of-magnitude
    evidence; the honest R1 test must re-run in full-split mode.
  - **CRITICAL (mechanism):** the `load_state_dict` swap path was never
    validated end-to-end (no proof that an unmodified FT dict through the
    same path reproduces the sidecar reference).
  - **Self-correction:** the previous turn's "environment blocked"
    conclusion was WRONG - the calibration probe used the fork's RLDS
    loader ("No registered data_dirs"), while every working run uses the
    local NPZ bridge (`weights/libero_spatial_no_noops_npz`, present and
    healthy). Wrong-loader artifact, not a missing dataset.
  - **Latent:** harness restore snapshot captured BASE weights while the
    comment claimed FT (harmless in practice - compressed dicts covered all
    keys - but contradicted its own contract). Fixed via `adopt_reference()`.
  - **Test bug found+fixed in passing:** `_record_n2_point` had an invalid
    `sum(1 for *_, ok in ...)` expression in the eval_info rewrite (would
    have been a SyntaxError at import; caught by py_compile).
- **Fixes (commit 15df3ce):** full-split streaming eval mode
  (`QICERT_N2_EVAL=full`, N1-identical `LocalEvalSplitStream` protocol,
  per-200-batch ETA logging, Wilson CI per point); `--rows=ft-anchor` that
  loads the UNMODIFIED FT dict through the exact swap path and evals the
  full split; eval scope + correct/total + CI recorded in every point's
  run.json; calibration CLI rewritten onto the local NPZ bridge over TRAIN
  episodes only (honest calibration, never held-out data).
- **Pre-registered expectation for the anchor:** full-split acc within
  noise of the N1 sidecar 0.4468 (Wilson 95% CI overlap; the sidecar number
  itself carries +/-0.0029). If the anchor misses materially, every harness
  number is suspect and the protocol gets re-derived before any further
  compression claim.
- **Result:** *(running - log: results/logs/N2-ft-anchor.log; early running
  acc at batch 200/6496 = 0.4347)*
- **Artifacts:** results/N2/<run_id>/ft-anchor-seed0 (run.json + metrics).

- **Anchor RESULT (14:48):** **PASS — exact.** Full-split acc **0.4468**
  over 6,496 batches / 114,497 tokens, diff **+0.0000** vs the N1 sidecar.
  The load_state_dict swap path is validated end-to-end and the harness's
  full-split protocol is N1-identical. All future harness numbers are
  anchored. (run: results/N2/<id>_ft-anchor-seed0, log:
  results/logs/N2-ft-anchor.log, wall 824s.)

---

## E-2026-09-12-04 — N2R best point re-run, FULL-SPLIT (the honest R1 test)

- **Commit at launch:** (this commit, post 15df3ce)
- **Command:** N2R, plan 0.50 only, r'=32 only, `QICERT_N2_EVAL=full`,
  seed 0, tt_split 0.6 — i.e. the best point from E-2026-09-12-02 re-scored
  under the validated N1-identical protocol.
- **Purpose:** the first split-level honest R1 evaluation of the residual
  arm. The single-batch 0.1667 (E-2026-09-12-02 point 2) is demoted to
  order-of-magnitude evidence; THIS number decides R1.
- **Pre-registered reading:** R1 bar = delta >= -0.05 vs 0.4468 with Wilson
  CI reported. Expected honest range: the batch-mode 0.1667 was ~3/18
  tokens; split-level could land anywhere in 0.05-0.35. If it lands well
  below the bar (likely per the diagnosis), the compression track's honest
  conclusion stands: residual-only insufficient, activation-weighted
  fitting is the next lever, claim demotes to certificate+safety spine.
- **Result:** *(running — log: results/logs/N2R-fullsplit-best.log)*
- **Artifacts:** results/N2R/<run_id>/ + ledger.
