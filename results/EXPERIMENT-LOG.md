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
- **Result:** *(queued — runs automatically after E-2026-09-12-01; log:
  results/logs/N2R-go-no-go-seed0.log)*
- **Artifacts:** results/N2R/<run_id>/ per point + ledger rows.
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
