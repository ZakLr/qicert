# qicert docs

- **`technical-report-v2.pdf`**: the technical report (source: `.tex`, figures in
  `figures2/`).
- **`concept-proposal.pdf`**: the 3-page concept proposal (source: `.tex`).
- `resource-declaration.md`: compute accounting, generated from
  `results/ledger.csv` by `scripts/export_resource_declaration.py`.
- `licenses.md`: component-by-component license table (models, datasets,
  dependencies).
- `backbones.md`: model/backbone provenance and download notes.
- `lipschitz-proof.md`: the exact per-layer bound algebra used by the certificates.
- `run-data-register.md`: which run directories back which report claims.

Every number in the reports traces to a run recorded in `results/ledger.csv`
(reproduce any table with `python -m qicert.bench.all`).
