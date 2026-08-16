#!/usr/bin/env bash
# N2 pipeline smoke: N1 (fine-tune 8 steps, save-ckpt, skip INT8) then N2
# (layer0 sweep, 2 plans x {TT, QTT}).
#
# Detached (-d) + ckpt saved to the MOUNTED results/ dir so a laptop sleep
# or Docker Desktop restart only costs the current step, not the pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."

IMG=qicert-dev:cu13-cudaq
FT_DIR=/workspace/results/ft-smoke          # inside container (host: results/ft-smoke)
LOG=/workspace/results/n2-pipe-smoke.log

docker rm -f n2-smoke-n1 n2-smoke-n2 2>/dev/null || true

echo ">>> STEP 1: N1 fine-tune (8 steps) + save-ckpt + skip-int8"
MSYS_NO_PATHCONV=1 docker run -d --name n2-smoke-n1 --gpus all \
  -v "$(pwd):/workspace" \
  -v "$(pwd)/weights:/workspace/weights" \
  -v "$(pwd)/.hf_cache:/root/.cache/huggingface" \
  "$IMG" bash -lc "cd /workspace && \
    python -m qicert.bench.all --module compress --rows baseline-int8 \
      --out results --exp-id N1 --steps 8 --batch 1 --seeds 0 \
      --save-ckpt $FT_DIR --skip-int8 --run-tag n1-save-smoke > >(tee -a $LOG) 2>&1; \
    echo 'STEP1_EXIT='$? >> $LOG"

echo ">>> waiting for N1 (model load ~4 min + 8 steps)..."
for i in $(seq 1 40); do
  sleep 30
  if grep -q "STEP1_EXIT=" "$LOG" 2>/dev/null; then break; fi
done
grep -q "STEP1_EXIT=0" "$LOG" && echo "N1 OK" || { echo "N1 FAILED (see $LOG)"; docker logs n2-smoke-n1 2>&1 | tail -20; exit 1; }

echo ">>> STEP 2: N2 sweep (layer0, plans 0.16+0.50, TT+QTT)"
MSYS_NO_PATHCONV=1 docker run -d --name n2-smoke-n2 --gpus all \
  -v "$(pwd):/workspace" \
  -v "$(pwd)/weights:/workspace/weights" \
  -v "$(pwd)/.hf_cache:/root/.cache/huggingface" \
  -e QICERT_FT_CKPT=$FT_DIR \
  -e QICERT_N2_LAYERS=layer0 \
  -e QICERT_N2_PLANS=0.16,0.50 \
  "$IMG" bash -lc "cd /workspace && \
    python -m qicert.bench.all --module n2_sweep --rows compression-pareto \
      --out results --exp-id N2 --seeds 0 --run-tag n2-layer0-smoke > >(tee -a $LOG) 2>&1; \
    echo 'STEP2_EXIT='$? >> $LOG"

echo ">>> waiting for N2 (model load + 4 eval points)..."
for i in $(seq 1 40); do
  sleep 30
  if grep -q "STEP2_EXIT=" "$LOG" 2>/dev/null; then break; fi
done
grep -q "STEP2_EXIT=0" "$LOG" && echo "N2 OK" || { echo "N2 FAILED (see $LOG)"; docker logs n2-smoke-n2 2>&1 | tail -30; exit 1; }

echo ">>> pipeline smoke COMPLETE — see $LOG and results/N2/"
