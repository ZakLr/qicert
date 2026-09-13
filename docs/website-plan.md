# qicert website plan — competition only

**Branch:** `clean/docs-release` (worktree `qicert-clean`). Main branch untouched.
All site work reuses already-measured artifacts; no new GPU jobs.

**Decisions locked by user (2026-09-13):**
* Publish full `results/` (with privacy scrub of absolute paths).
* Build both variants and let the user choose: (A) single-page proof-sheet, (B) 7 routed plates. Same content, two navigations.
* JS guard demo allowed — small, deterministic.

## 1. Grounding — one subject, one job

* Subject: a single shrunken VLA that still carries a proof. Not "quantum AI".
* Audience: challenge judges (30s scan) + quantum-ML engineers who will clone the repo.
* Job: prove (a) honest 2x wall, (b) diagnosed collapse, (c) certified recovery — every number links to a run directory.

## 2. Visual identity — anti-slop, opinionated

* Palette — lab instrument: Ink `#0B1215`, Paper `#E9E6DD`, Ruler `#8A9BA8`, Signal `#E35D33` (GO), Cert `#2EC4B6`, Warn `#F7B32B`. One accent at a time.
* Type — Instrument Serif (display, only headline number) + Suisse Int (body) + JetBrains Mono (hashes, run IDs, manifest SHA). Type is memorable, not containers.
* Layout — proof-sheet: wide margins, hairline rules that *are* the grid, left-rail TOC that is also the artifact index. Numbering `01–05` is correct here because the content *is* a sequence: Baseline → Collapse → Repair → Guard → Safety.
* Signature — **manifest ruler:** sticky top rule that is a content-addressed hash tape (`…a3f9 — 114,497 tok — 2.46× — GO`). Scroll advances it; click opens `run.json`. One orchestrated moment, not scattered effects.

Avoided defaults: no centered gradient hero, no cream+terracotta, no acid-green on black, no broadsheet columns. The ruler *is* the certificate.

## 3. Information architecture — 7 plates

1. **Hero / Thesis** — `‖W‖₂ ≤ ∏‖G_k‖₃ — measured, not claimed`. Right: 3-bar proof-sheet `INT8 1.997× 0.4466` vs `Uniform 2.0× 0.000` vs `Repair 2.46× 0.4535/0.4609`, Wilson CIs, Sound 168/168. Two CTAs: *Read report* + *Reproduce*.
2. **The honest wall** — uniform TT 2.0×→0.000, TT+residual 0.164 at 2.54/2.86×, spectra PR 40–845, bimodal agreement curve (584/1007, 86/704), predictor R² 0.16 honest failure.
3. **Repair → Confirm** — per-seed timeline: seed 0 `0.1586→0.4535`, seed 1 `0.3970→0.4609`, seed 2 running; repair cost 1200 steps charged visibly; honest ratio 2.459×.
4. **Certificates** — per-layer `L ≥ tight` (168/168), limit note: naive chain `log10L 71` vacuous, shippable claim is per-layer exactness.
5. **Guard (interactive)** — 5-case transcript (SERVE, out-of-ball REFUSE, tampered REFUSE, NaN REFUSE, missing REFUSE) as state machine + entropy gate `ActionDiversityMonitor` with live 64-token window demo (collapsed warns, held-position ok, warn≠refuse).
6. **Safety + Monitor** — estimator race (IQAE 630 vs MC 1e6), conformal panel, shadow-syndrome `FPR 0.0000/4000` vs 0.01 budget, 0.35ms/step, latency ≤100ms.
7. **Reproduce** — `pip install -e ".[dev]"`, `pytest -q` (84), `bench --rows=smoke`, `run_remaining_experiments.py --list`, `make_report_figures.py` regenerating 7 vector PDFs.

Proposal (3pp) + report (8pp) embedded as PDFs; site is narrative, docs are record. Cold-reader rule: every ID (N9, N2R2, etc.) defined on first use via the pipeline legend.

## 4. Tech

* Marketing site at `/` as single static page (Astro or plain HTML+CSS); docs at `/docs` = existing `mkdocs-material` build (12 pages, build 0 errors, `site/` gitignored). Shared palette/type.
* No backend. Guard interactivity mirrors `guard.py` predicate exactly, linked to Lean proof (`lean/`). No model load.
* Deploy: same repo, `clean/docs-release` branch, GitHub Pages (`/` + `/docs`).

## 5. Data & privacy blockers (audit 2026-09-13)

* `clean` branch missing `results/N9*` dirs on disk (only ledger rows) — `fig_compression_plane` expects `results/N9C/*/run.json`. Sync `main → clean` or rebuild figures from ledger before public build.
* `results/ledger.csv` and `EXPERIMENT-LOG.md` leak `C:\Users\zakil\…`. Add scrub step stripping to repo-relative paths before site build and before publishing `results/`.
* `resource-declaration.md` stale (25 rows, 6.37h) — regenerate from current ledger (now 62+ scored rows) before submission.

## 6. Execution — 4 phases

**A. Evidence sync (0.5d):** sync results, regenerate figures + resource declaration, privacy scrub.
**B. Shell + ruler (1d):** scaffold rail, ruler, 7 plates, responsive + keyboard focus + reduced-motion.
**C. Content + interactivity (1.5d):** wire real ledger numbers, guard state machine, entropy window, race chart. Critique: remove one accessory.
**D. Final scrub (0.5d):** cold-reader pass (no `HANDOFF`/`AGENTS.md`/internal paths without definition), rebuild both PDFs, `pytest` 84, `lake build`, `mkdocs build` all green, then snapshot.

## 7. Variants to deliver

* **Variant A — Single-page proof-sheet:** sticky rail + ruler, continuous scroll, section hashes (`#collapse`, `#repair`). Best for judges.
* **Variant B — Routed plates:** same 7 plates as `/`, `/collapse`, `/repair`, `/guard`, etc., with shared ruler. Best for engineers deep-linking. Both built from same content; user picks winner.
