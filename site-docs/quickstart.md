# Quickstart

## Install

```bash
pip install -e ".[dev]"
```

Pins are frozen in `environment.yml` (PyTorch CUDA build, transformers,
numpy/scipy, CUDA-Q for the estimator simulation, Julia reserved for the
sum-of-squares bridge).

## Smoke test (no GPU, no weights, ~1 min)

```bash
python -m pytest tests -q                          # 84 tests
python -m qicert.bench.all --rows=smoke            # bench smoke set
```

CI runs exactly this on every push; the badge in the README reflects it.

## First benchmark (needs weights, optional GPU)

```bash
# every report table from scratch
python -m qicert.bench.all

# one table at a time
python -m qicert.bench.all --rows=compression-pareto
python -m qicert.bench.all --rows=calibrated-int8 --out results --exp-id int8-ref --seed 0 --run-tag n8c-fullsplit

# the resumable experiment queue (what the paper's numbers came from)
python scripts/run_remaining_experiments.py --list
python scripts/run_remaining_experiments.py --only baseline-seeds,compress-search,compress-confirm --no-tests --dry-run
python scripts/run_remaining_experiments.py --only calib-s1,calib-s2,repair-s1,confirm-s1,repair-s2,confirm-s2
```

Single-GPU rule: run one bench module at a time. Two concurrent GPU jobs
exhaust host RAM and freeze the machine (documented in the reports and in
run logs).

## Pretrained weights

- Base checkpoint: `weights/ckpt/checkpoints/step-122500-epoch-55-loss=0.0743.pt` (HF).
- Dataset bridge: `weights/libero_spatial_no_noops_npz/` (local NPZ episode files,
  verified against the upstream loader; see `python/qicert/data/`).
- The harness freezes an evaluation split into `results/eval_split.json` before
  any compression run, so no result below was tuned on its own eval set.
