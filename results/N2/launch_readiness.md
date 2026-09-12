# N2 launch readiness — what we have, what we need, how to run later

**Generated 2026-09-12 (CPU-only preparation).** No GPU, no VLA, no dataset
used to produce this — it is a self-contained readiness summary plus the
deterministic launch specs, so that when we press Go we can reproduce any
point from `results/N2/blueprint_seed*.json` without re-deriving parameters.

---

## 1. The N2 contract (single source of truth: bench/n2_sweep.py)

- **Backbones:** TT (d=2) + QTT (d=4), both **bit-reversed** (N2′ winner:
  `178c849b6525`, recon err 0.9185 vs 0.9365/0.9387 for the other two
  orderings).
- **Bond plans:** `(0.02, 0.04, 0.08, 0.16, 0.33, 0.50)` — uniform-param-fraction
  targets spanning 2×..50× compression, including the 0.50 (2×) point where
  R1's kill criterion ("≤5% accuracy drop at ≥2×") is evaluated.
- **Kernel:** `tt_svd` deterministic reference; `tt_cross` is the Q19 CUDA-Q
  port story (NOT this run).
- **Layers:** all LLM linear projections (24 transformer layers × q/k/v/o/
  gate/up/down); action head excluded in v1.
- **Eval:** the **SAME batch as N1 (matched budget)**, on the **FINE-TUNED**
  weights — i.e. N2 scores each compressed model on the exact eval batch the
  FT baseline was measured on, so the delta vs the FT baseline is a
  matched-budget comparison, not a fresh batch.
- **Capture:** every point records `run.json` + `config.json` + `system.json` +
  `env.json` + `metrics.jsonl`, and appends `results/ledger.csv`.  That is the
  minimum; the "maximum data" additions are in §4.

**Points per seed:** 2 backbones × 6 plans = **12**.

---

## 2. Handoff state right now (which seeds are clean)

A seed is a **CLEAN matched-budget N2 handoff** only when all three exist in
`results/N1v2-ckpt/`:

- `seed{N}.pt`            — LoRA-merged FT LLM backbone (~942 MB for seed 0)
- `eval_batch_seed{N}.pt` — the EXACT eval batch the FT was measured on (~1.2 MB)
- `baseline_seed{N}.json` — FT reference acc for delta (0.4468 for seed 0)

| seed | seed.pt | eval_batch_seed.pt | baseline_seed.json | FT ref acc | clean handoff |
|------|--------:|-------------------:|-------------------:|-----------:|:------------:|
| 0    | ✅      | ✅                 | ✅                 | 0.4467715  | **YES**      |
| 1    | ❌      | ❌                 | ❌                 | —          | NO           |
| 2    | ❌      | ❌                 | ❌                 | —          | NO           |

**Consequence:** seed 0 is the only seed that can run N2 now. Seeds 1 and 2
cannot be clean N2 handoffs until their N1v2 run completes with
`--save-ckpt results/N1v2-ckpt` (so the trio exists).

See `results/N2/handoff_index.json` for the machine-readable version.

---

## 3. What's already on disk from earlier (not from this preparation)

There are **4 pre-existing N2 point dirs** from an earlier Aug-16 smoke:

| dir | backbone | plan | ratio | eval_acc | delta vs FT | sound layers | matched batch | wall sec |
|-----|----------|------|------:|----------:|------------:|-------------:|:------------:|---------:|
| `9f6f5e61d06b_pareto-TT-0.500-seed0` | TT | 0.50 | 2.00× | 0.250 | 0.000 | 7/7 | ✅ | 25.8 |
| `2a3343003fb8_pareto-QTT-0.500-seed0` | QTT | 0.50 | 16.27× | 0.000 | -0.250 | 7/7 | ✅ | 16.3 |
| `dc5715f6a3f5_pareto-TT-0.160-seed0` | TT | 0.16 | 6.25× | 0.125 | -0.125 | 7/7 | ✅ | 52.6 |
| `cb296fa62e15_pareto-QTT-0.160-seed0` | QTT | 0.16 | 23.51× | 0.125 | -0.125 | 7/7 | ✅ | 18.2 |

**Honest reading of these:** these are **Layer-0 smoke points** on the *base
checkpoint* (Aug-16, `run-tag n2-layer0-smoke`), NOT the current N2 plan's
full-layer, seed-0, matched-budget N1v2 handoff run. They are useful as a
"TT-SVD on this backbone collapses at uniform ratio" signal (consistent with
the N2local/E4 failure), but they are **not** the scored N2 result — the scored
run is the one launched later from the blueprint, against the N1v2 handoff,
full layer scope, bit-reversed ordering, with per-point provenance. Do not cite
these as "N2 results" in the report without labeling them as the earlier
layer-0 smoke.

Each of these dirs has a full `run.json` (provable, regenerable), so they're
valid recorded artifacts — just the wrong scope for the final N2 table.

See `results/N2/<dir>/run.json` for the exact record of each.

---

## 4. Maximum-data capture — what we will record per point (the extra stuff)

The bench recorder already gives us, per point, the minimum: `run.json` +
config/system/env + `metrics.jsonl` + ledger row, with per-layer `lipschitz`,
`tight_norm`, `sound` metrics and the final `eval_acc`, `delta_vs_n1_ft`,
`ratio`, `cert_sound_layers`, `matched_batch`, `wall_sec`.

On top of that, the later run should persist (manually or via a small harness
edit to `bench/n2_sweep.py`'s `_run_n2` / `_record_n2_point`):

1. **Pre-launch snapshot** — already done: `results/N2/pre_launch_snapshot.json`
   (timestamp, host, OS, python, torch, cuda, GPU name + memory, docker status,
   prismatic-on-host status, N1v2 handoff for seeds 0/1/2, checkpoint size,
   N2′ winner + recon errs, the full sweep contract).  This is the provenance
   record **before** the run, so the run is self-explanatory.

2. **Per-point full per-layer certificate table** — `results/N2/<run_id>/
   per_layer_certificates_<backbone>_<plan>.json`: for every compressed layer,
   `layer_type`, `L_raw`, `L_min` (gauge), `tight`, `kappa_raw`, `kappa_min`,
   `params`, `sound`.  The report can then print the certificate table verbatim
   instead of recomputing.

3. **The exact eval batch used, copied into the run dir** — `results/N2/<run_id>/
   eval_batch_seed0.pt` + `baseline_seed0.json`.  A future reader can re-score
   any compressed model on the identical inputs without touching the handoff dir.

4. **Per-point summary** — `results/N2/<run_id>/point_summary.json`:
   `run_id`, `run_tag`, `backbone`, `plan_fraction`, `ratio`, `eval_acc`,
   `delta_vs_n1_ft`, `cert_sound_layers`, `matched_batch`, `wall_sec`,
   `n_layers_compressed`, `note`.  One-glance summary separate from the raw
   ledger.

5. **Run-level summary** — `results/N2/<run_id>/run_summary.json` (where
   `<run_id>` is the seed-level run, not per-point): `run_id`, `run_tag`,
   `exp_id`, `seed`, `backbones`, `plans`, `n_points`, `n_points_passed`,
   `n_points_collapsed`, `best_ratio_within_5pct_drop`,
   `worst_ratio_within_5pct_drop`, `certificate_sound_count_total`,
   `wall_sec_total`, `n2prime_winner`, `n1_ft_ref_acc`, `config_hash`,
   `start_utc`, `end_utc`.

6. **Failure capture** — if a point crashes, the per-point `metrics.jsonl` must
   include the exception type + message + the layer(s) being processed when it
   failed, so a future rerun can skip exactly the failing point rather than
   re-deriving the failure.

Items 2–6 are the "maximum data" layer.  Items 1 and the bench recorder are
already in place.  The cleanest path is to fold 2–6 into a small helper that
`_run_n2` calls per point + at run end, so it's automatic and never a manual
afterthought.  That helper is small and contained; not done yet (CPU-only prep
today, harness edit is also CPU-only).

---

## 5. Deterministic launch specs (reproducible later)

For each seed, `results/N2/blueprint_seed{N}.json` is the fully-determined
launch spec: seed, handoff status, exp_id, run_tag, backbones, bond_plans,
layers_scope, ordering, kernel, points, eval_protocol, handoff dir, env
overrides, and **both** the Docker command and the host-venv fallback command,
with the per-seed tag already set.

**Seed 0 blueprint highlights (the only seed ready now):**
- `run_tag = N2-compression-pareto-seed0`
- `exp_id = N2`, `seed = 0`, `backbones = [TT, QTT]`, `bond_plans = [0.02,
  0.04, 0.08, 0.16, 0.33, 0.50]`, `points = 12`
- `fine_tuned_handoff_dir = results/N1v2-ckpt`
- env overrides: `PRISMATIC_DATA_ROOT`, `QICERT_FORK`, `QICERT_FT_CKPT`,
  `QICERT_N2_LAYERS=all` (NOTE: `QICERT_N2_PLANS` intentionally NOT set —
  set it to `"0.50,0.33"` for the go/no-go variant).

**Docker command (preferred — CUDA-Q actually available here):**
```
cd "/c/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert" && \
MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
  -v "$(pwd):/workspace" \
  -v "$(pwd)/weights:/workspace/weights" \
  -v "$(pwd)/.hf_cache:/root/.cache/huggingface" \
  qicert-dev:cu13-cudaq bash -lc '
set -euo pipefail
cd /workspace
export PYTHONUNBUFFERED=1
export PRISMATIC_DATA_ROOT="/workspace/weights/data"
export QICERT_FORK="/workspace/weights/code"
export QICERT_FT_CKPT="/workspace/results/N1v2-ckpt"
python -m qicert.bench.all --module n2_sweep --rows=compression-pareto \
  --out results --exp-id N2 --seed 0 \
  --run-tag N2-compression-pareto-seed0 \
  --save-ckpt results/N1v2-ckpt \
  --capture heavy \
  2>&1 | tee results/logs/N2-compression-pareto-seed0.log
'
```

**Go/no-go variant (2× point first, smaller wall time) — recommended first press:**
```
export QICERT_N2_PLANS="0.50,0.33"
<same docker command as above>
```
This restricts seed 0 to 2 backbones × 2 plans = 4 points (the two least-
aggressive), which is the real go/no-go on R1's 2× point.  If those survive,
expand to the full sweep by unsetting `QICERT_N2_PLANS`.

**Host venv fallback (only if `prismatic` importable on host — it was NOT when
I checked today, so this is a conditional fallback only):**
```
cd "/c/Users/zakil/Desktop/AQC/Quantum Insider/challenge/qicert" && \
export PYTHONUNBUFFERED=1 && \
export PRISMATIC_DATA_ROOT="$(pwd)/weights/data" && \
export QICERT_FORK="$(pwd)/weights/code" && \
export PYTHONPATH="$(pwd)/python:$(pwd)/weights/code" && \
.venv312/Scripts/python.exe -m qicert.bench.all --module n2_sweep --rows=compression-pareto \
  --out results --exp-id N2 --seed 0 \
  --run-tag N2-compression-pareto-seed0 \
  --save-ckpt results/N1v2-ckpt \
  --capture heavy \
  2>&1 | tee results/logs/N2-compression-pareto-seed0.log
```

**Seeds 1 and 2 — block on handoff:**
Do NOT launch N2 for seeds 1/2 yet.  First run N1v2 for each seed with
`--save-ckpt results/N1v2-ckpt` (one seed at a time, Docker, unbuffered), so
the `seed{N}.pt` + `eval_batch_seed{N}.pt` + `baseline_seed{N}.json` trio exists.
Only then is the N2 handoff clean.  The blueprints for seeds 1/2 are written
(`results/N2/blueprint_seed1.json`, `results/N2/blueprint_seed2.json`) but their
handoff status is `clean_handoff: false`.

---

## 6. Recommended order when we press Go

1. **seed 0, go/no-go variant** — `QICERT_N2_PLANS="0.50,0.33"`, Docker,
   unbuffered, `--capture heavy`.  Read the 4 point dirs + their run.json.  If
   2× + 0.33 survive (accuracy within noise of the FT baseline, certificates
   sound), proceed; if they collapse, pivot to the residual-fitting arm instead
   of burning the full sweep.
2. **seed 0, full sweep** — unset `QICERT_N2_PLANS`, same command, 12 points.
3. **seeds 1, 2** — only after their N1v2 runs complete with `--save-ckpt
   results/N1v2-ckpt`, one seed at a time.

---

## 7. Files in this folder right now

- `pre_launch_snapshot.json` — provenance record **before** any run (see §4.1)
- `handoff_index.json` — machine-readable handoff state for seeds 0/1/2 (§2)
- `blueprint_seed0.json` / `blueprint_seed1.json` / `blueprint_seed2.json` —
  deterministic launch specs (§5)
- `launch_readiness.md` — this file
- `2a3343003fb8_pareto-QTT-0.500-seed0/`
- `9f6f5e61d06b_pareto-TT-0.500-seed0/`
- `cb296fa62e15_pareto-QTT-0.160-seed0/`
- `dc5715f6a3f5_pareto-TT-0.160-seed0/` — earlier Aug-16 layer-0 smoke points
  (§3); not the scored N2 result, but valid recorded artifacts with full run.json

---

*This is a readiness + preparation document, not a results document.  When the
run lands, the scored N2 result lives in `results/N2/<run_id>/` with its full
run.json + the extra capture artifacts from §4, and the report's N2 row is
filled from those, not from this file.*
