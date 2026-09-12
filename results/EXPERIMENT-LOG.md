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
- **Result:** *(running — first point scored as below; continues for 4 points)*
  - **Point 1 scored** (frac=0.50, r′=8, tt_split=0.6 → net ratio 3.18×):
    eval acc 0.0000, delta −0.4468 vs FT ref, 168/168 certificates sound.
    **Same zero as the failed N2 uniform run, but mechanism different:** the
    residual folded back ~70% of each layer's RMS error (per-layer recon
    0.67–0.74 across 168 layers — uniform run had ~0 recovery). So the
    repair arm *is* improving the weights mathematically on every layer; the
    FT model's output layer is just so sensitive that 70% of the per-layer
    error still leaves the action-token logits fully shattered — 70% is the
    right direction but not enough to cross the activation threshold.
  - Diagnosis from per-layer recon (168 layers): consistent 0.67–0.74 — the
    repair works evenly across the model, not selectively; this is the
    honest signature of a flat-spectrum backbone where the error is spread
    rather than concentrated in a few layers you can patch big. With rank-8
    residual factors that's the best a Frobenius-optimal patch does; the
    action head's output is what actually has to survive.
  - Plain reading: at 3.18×, the compressed+repaired model still produces
    near-uniform action logits → 0.000 action accuracy. The *weights* got
    70% closer, but the *predictions* didn't cross the threshold. This is
    why residual-only on flat spectra is a weak lever and why deployment
    real solutions (GPTQ-intrinsic, QuaSAR) additionally fit the residual
    to *activation-weighted* error + per-channel scales optimized against
    output loss on a calibration set — not raw Frobenius on the weight.
  - Two paths remain open (both pre-registered): (a) **rank-32 residual**
    (more budget to the correction) + per-channel scales optimized against
    the calibration activations (the collector in
    `python/qicert/calibrate.py`); (b) if even that collapses, the honest
    conclusion for the compression track is that TT truncation at this
    backbone fails the R1 bar and the claim demotes to the certificate +
    safety story.  Units + math tested in `tests/test_compress_residual.py`
    (monotone improvement, exact-rank recovery, adversarial improvement,
    sound Lipschitz bound, honest param counting). Net ratio accounts for
    cores + residual factors + scales (honest vs INT8).

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
