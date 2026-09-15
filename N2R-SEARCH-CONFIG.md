# N2R2 search plan — fast configuration search for compression > 2x

**Status:** ready to run.  Built into the queue as two stages:
- `n2r2-search` — fast configuration search
- `n2r2-confirm` — full-protocol confirmation of the chosen candidate

Checkpoint saved locally before this work started.
(commit `6b47326`, branch `main`).

---

## 1. Goal

Find whether any N2R-v2 configuration can give a decent **accuracy +
certificates** result at **more than 2x compression**, without collapsing
accuracy the way the earlier naive compression did.

This is the decision stage for:
- compression beyond 2x,
- certificates still sound,
- accuracy not destroyed.

---

## 2. What a "configuration" actually is

Each candidate is defined by:

- **plan fraction** — overall compression target for the compressed layers
  (example: 0.50 vs 0.33).
- **residual rank cap** — how much low-rank correction we allow per layer
  after the tensor decomposition.
- **fit mode** — plain vs activation-weighted repair.
- **mixed allocation** — whether attention projections are kept full
  precision while everything else is compressed.
- **calibration statistic** — which activation statistic is used for the
  activation-weighted repair, if any.
- **tensor/budget split** — how much of the per-layer budget goes to the
  tensor cores vs the residual correction.

So "configuration" is not a vague word here. It is a concrete set of
compression/repair choices.

---

## 3. Why this can be faster without cheating

The main cost is not compression. It is the **full eval stream** over the
frozen split.

The search stage reduces cost in four honest ways:

1. **fixed smaller eval budget**  
   The search uses a fixed, smaller number of eval batches for every
   candidate. This is set before the search and is identical across
   candidates.

2. **cached FT reference predictions**  
   The reference predictions are computed once per seed and reused for
   every candidate. This removes a redundant full eval pass per candidate.

3. **conservative early stopping**  
   A candidate can be stopped early only if it is clearly failing. Early
   stopping can never promote a candidate.

4. **separate confirm stage**  
   The reported number is always the full-protocol confirm run for the one
   chosen configuration. The search results are only search-phase evidence.

Precision is preserved because:
- the metric definition is the same: token-level action accuracy on the
  frozen eval split, same harness, same scoring rule;
- we only reduce the number of batches in search, and we mark that
  explicitly in every artifact;
- the reference stream is cached, but the compressed candidates are still
  evaluated independently;
- the confirm stage removes the search cap.

---

## 4. Config space we are trying

The default search grid is small and motivated, not a brute-force random
sweep. Each entry has a reason attached so the report can explain *why* we
tried it.

Default grid:

1. weighted, no mixed, residual cap 32, plan 0.50  
   reason: search baseline — repair only, no mixed allocation.

2. weighted, no mixed, residual cap 64, plan 0.50  
   reason: same but more correction budget.

3. weighted, mixed, residual cap 32, plan 0.50  
   reason: keep attention projections full precision.

4. weighted, mixed, residual cap 64, plan 0.33 if available  
   reason: deeper compression with mixed allocation and larger residual cap.

5. weighted, mixed, residual cap 32, plan 0.33 if available  
   reason: deeper compression, mixed, moderate residual cap.

If the env plans do not include 0.33, the deeper compression candidates are
omitted automatically.

---

## 5. Search parameters

Default search env:

- `QICERT_N2_EVAL=full`
- `QICERT_N2R2_MODE=weighted`
- `QICERT_N2R2_STAT=mean`
- `QICERT_N2R2_PLANS=0.50,0.33`
- `QICERT_N2R2_RRANKS=32,64`
- `QICERT_N2R2_MIXED=1`
- `QICERT_N2R_TTSPLIT=0.6`
- `QICERT_N2R_SEARCH_BATCHES=100`
- `QICERT_N2R_SEARCH_GO=0.30`
- `QICERT_FT_CKPT=results/N1v2-ckpt`
- `PRISMATIC_DATA_ROOT=weights/data`

Interpretation:
- search budget = 100 eval batches per candidate,
- provisional GO bar = 0.30 search accuracy for ranking,
- mixed allocation on by default in the search because that is one of the
  main plausible levers,
- mean activation statistic preferred over absmax for the weighted repair,
  with absmax as fallback if mean is missing.

---

## 6. Candidate selection rule for confirmation

After the search, the confirm stage picks **one** configuration to run
through the full protocol.

Selection rule is deterministic:

1. Prefer a provisional-GO candidate with the highest search accuracy.
2. If none is provisional-GO, pick the highest search accuracy overall.
3. Tie-break is by directory order.

This rule is written down here so it cannot be changed after seeing the
numbers without notice.

---

## 7. What gets saved for every candidate

For every search candidate, the run writes:

- `results/N2R-search/<seed>_<cfgkey>_frac<p>_rr<r>/run.json`
- full config, including the human reason,
- search accuracy, agreement vs reference, sound layers, kept layers,
  provisional GO flag,
- eval scope string that says `search-prefix`,
- wall time.

So even if we only report one confirm result later, every searched
configuration is still auditable.

---

## 8. What gets reported

Only the **confirm** result is reportable as the final N2R2 number.

The report must say:

- the search used a smaller fixed eval budget,
- the FT reference stream was cached during search,
- early stopping was conservative,
- the final number comes from the full-protocol confirm run,
- the chosen configuration and why it was chosen.

If the confirm run does not pass the pre-registered bar, we do not claim it
does. If it does, we cite the confirm row, not the search row.

---

## 9. Relationship to existing results

The existing N2R2 results are still on disk:

- `results/N2R2/..._n2r2-weighted-mixed-0.500-rr32-seed0`
- `results/N2R2/..._n2r2-weighted-0.500-rr64-seed0`
- `results/N2R2/..._n2r2-weighted-0.500-rr32-seed0` (duplicate point)

Those are full-split results from the earlier run. They are real, but they
did not pass the accuracy bar.

The new search/confirm pipeline does not delete them. It adds a faster way
to test more configurations and a cleaner separation between search and
reported result.

---

## 10. How to run it

From the `qicert` directory:

```bash
cd "<repo>"
.venv312/Scripts/python.exe scripts/run_remaining_experiments.py
```

The queue will now run:

1. preflight
2. calib-topup
3. n2r2-search
4. n2r2-confirm
5. n2r2-weighted (legacy fallback only if no search was done)
6. n2r2-mixed (legacy fallback only if still needed)
7. n5-plain
8. n5-weighted
9. n5-predictor
10. n1v2 seeds 1 and 2
11. summary

If you only want the N2R2 search part, use:

```bash
.venv312/Scripts/python.exe scripts/run_remaining_experiments.py --only n2r2-search
```

If you want search plus confirm only:

```bash
.venv312/Scripts/python.exe scripts/run_remaining_experiments.py --only n2r2-search,n2r2-confirm
```

---

## 11. Notes for later report wording

Useful honest phrasing once results exist:

- “We first searched a small motivated configuration space using a fixed
  smaller eval budget with a cached reference stream and conservative early
  stopping.”
- “The reported result is the full-protocol confirmation run for the chosen
  configuration, not the search score.”
- “The search was used only to select the configuration.”
- “Each searched candidate is preserved in the repo for audit.”

If the confirm run still misses the bar, the honest line is:

- “We attempted to rescue accuracy at >2x compression with activation-weighted
  residual repair and mixed allocation. The configuration that looked most
  promising in search did not meet the pre-registered accuracy bar in the
  full run. The certificate property still holds.”

If the confirm run passes, the honest line is:

- “The chosen configuration reached the pre-registered accuracy bar at
  <ratio>x compression with certificates sound, under the full eval protocol.”

---

## 12. Safety reminder

Do not:

- change the eval metric between search and confirm,
- change the split between search and confirm,
- cherry-pick the best search batch subset as the final number,
- compare a partial search eval against a full baseline eval and call that
  the result.

The whole point of this design is to make the search faster without turning
it into a hidden claim.
