# Swarm Provenance — 2026-09-10

## Method
- Harness: `qicert/scripts/lit_swarm.py` (committed) — 6 lanes × 5 queries, arXiv Atom API
  primary (relevance-sorted, 15 results/query, 3 s politeness delay, disk-cached), optional
  Semantic Scholar citation enrichment (disabled this run: unauthenticated S2 returned 429
  persistently, even with 30 s backoffs — recorded as tooling finding T-S8).
- Scoring: keyword-relevance heuristic (TT/TN/VLA terms ×3, recency boost 2025+, citation
  boost, negative-topic penalty, query-term coverage) + multi-query hit-count boost.
  Scores rank *reading order*; they are not verdicts. All decision-critical items were
  human-read (abstract level) before inclusion in findings.
- Dedupe: by ArXiv ID (fallback title-hash); cross-query hits tracked and boosted.
- Personas: this run was executed as one analyst doing L1→L6 sequentially rather than
  parallel subagents (subagent spawning unavailable in-session). Query fan-out per lane +
  independent per-lane findings files preserve the swarm structure; the recommended
  re-runs below can be parallelized across personas verbatim.

## Coverage
| Lane | Queries | Hits/query | Unique | Shortlisted | Notes |
|---|---|---|---|---|---|
| L1 compression | 5 | 15,15,15,9,0 | 52 | 15 | last query syntax too narrow |
| L2 certificates | 5 | 15,11,0,15,7 | 48 | 15 | ditto |
| L3 VLA | 5 | 15,15,15,13,15 | 61 | 15 | full coverage |
| L4 safety | 5 | 10,15,15,15,15 | 69 | 15 | full coverage |
| L5 calibration | 5 | 15,9,15,15,4 | 57 | 15 | good |
| L6 quantum | 5 | 1,6,1,15,15 | 38 | 15 | AE queries thin — lane quiet, expected |
| **Total** | 30 | | **325** | **90** | |

## Verification status
- All papers cited in findings: **[abstract-verified]** (arXiv Atom abstract read this run).
- KARIPAP quantitative claims: **[SECONDHAND]** — abstract-level only, numbers unverified.
- Minima/CompactifAI/GPTQ-intrinsic details used in Plan-A actions: consistent with
  abstracts read this run + earlier DEEP-CRITIQUE verification; full-text reads flagged
  where decisions hinge on them (GPTQ-intrinsic §3 theorem, HiTaB §4).
- Zero-hit queries: recorded, not silently dropped (4 queries had overly-narrow syntax;
  re-run list in 00-SWARM-SUMMARY).

## Artifacts
- `papers/L{1-6}.json` — shortlists (title/abstract/year/ids/url/score/hits)
- `raw/arxiv__*.json` — 26 query caches (reruns reuse them; delete to force refetch)
- `findings/L{1-6}-*.md` + `findings/00-SWARM-SUMMARY.md` — analysis layer
- `state.json` — run ledger

## Re-run instructions (standing swarm)
```bash
cd qicert && python scripts/lit_swarm.py --lanes all --results-per-query 15 --limit 12
# lane edits: edit LANES dict in the script; new lanes append L7+
```
Re-query queue before final report: see 00-SWARM-SUMMARY §"Swarm coverage gaps".
