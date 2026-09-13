#!/usr/bin/env bash
# N1 smoke runner (background): MiniVLA LoRA fine-tune + INT8 reference.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf results/N1
MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
  -v "$(pwd):/workspace" \
  -v "$(pwd)/weights:/workspace/weights" \
  -v "$(pwd)/.hf_cache:/root/.cache/huggingface" \
  qicert-dev:cu13-cudaq \
  python -m qicert.bench.all --module compress --rows baseline-int8 \
    --out results --exp-id N1 --steps 24 --batch 1 --seeds 0 \
    --run-tag libero-spatial-lora-r8-smoke
