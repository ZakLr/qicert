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
- **Result:** *(running — log: results/logs/N2-go-no-go-seed0.log)*
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
