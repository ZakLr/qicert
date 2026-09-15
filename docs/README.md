# qicert docs

- `submission/` — **the Phase-1 submission package**: technical report (PDF + LaTeX),
  concept proposal (PDF + LaTeX), resource declaration, and the dependency license
  table. Start with `submission/README.md`.
- `backbones.md` — model/backbone provenance and download notes.
- `lipschitz-proof.md` — the exact per-layer bound algebra used by the certificates.

Every number in `submission/` traces to a run recorded in `results/ledger.csv`
(reproduce any table with `python -m qicert.bench.all`).
