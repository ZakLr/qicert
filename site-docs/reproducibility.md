# Reproducibility

## Artifacts

Every bench invocation with `--out results --exp-id <name>` writes:

- `run.json` — results + config mirror (the survived-results schema).
- `config.json` — full config under `{"config": {...}}`.
- `system.json` + `env.json` — host, GPU, pinned versions.
- `metrics.jsonl` — per-step trajectory (if captured).
- A row appended to `results/ledger.csv` keyed by `config_hash`.

Per-run figures are regenerated from those directories by
`scripts/make_report_figures.py` — no figure is hand-drawn.

## Seeds and splits

- Evaluation split: 108 episodes / 6,496 batches / 114,497 action tokens,
  frozen into `results/eval_split.json` with SHA-256 hash before any
  compression run.
- Seeds 0, 1, 2: independent fine-tunes (batch 2, 1200 steps) with their
  own `seed{S}.pt` + `baseline_seed{S}.json` (Wilson interval recorded).
  Pre-registration fixes the acceptance bar before the runs that are judged;
  the verdict is a mechanical read of `run.json`, not a judgment call.
- Train/eval contamination guard: repair stages stream from `LocalEpisodeStream`
  filtered by `episode_ids=train_episode_ids` (complement of the frozen eval set),
  verified in code and in every N9 metrics file.

## Resource declaration

`scripts/export_resource_declaration.py` (produces `docs/submission/resource-declaration.md`)
collects every completed run's wall time, GPU name, and status from
`results/ledger.csv` — the paper's resource table is not hand-edited.

## Clean environment

```bash
pip install -e ".[dev]"
python -m pytest tests -q               # 84 tests
python -m qicert.bench.all --rows=smoke # tiny, fast, no private data
```

Both commands must succeed from a fresh venv — this is what CI checks.

## Machine

All Phase-1 scored numbers in the frozen reports ran on one laptop GPU
(NVIDIA RTX 5060 8 GB). Earlier ledger rows carry the legacy Tesla T4 tag
from the Kaggle phase and remain auditable; they do not affect the current
bar.
