#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export E2E_STEP_CANDIDATES="${E2E_STEP_CANDIDATES:-40}"
OUTPUT="${OUTPUT:-outputs/e2e_step40_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$OUTPUT")"
python -u scripts/train_e2e_single.py \
  --output "$OUTPUT" --device "${DEVICE:-cuda}" \
  --initialization sft --regression-weight "${REGRESSION_WEIGHT:-0.2}" \
  --schedule-init earliest_finish \
  --workers "${WORKERS:-16}" --branches "${BRANCHES:-16}" \
  --roots-per-cycle 128 --load-weight "${LOAD_WEIGHT:-0.5}" --decision-batch "${DECISION_BATCH:-16}" \
  --trace-hops "${TRACE_HOPS:-6}" \
  --load-share 0 \
  --horizon 10 --episode-steps 200 --long-every 0 \
  --adaptive-after "${ADAPTIVE_AFTER:-2}" --long-horizon "${LONG_HORIZON:-20}" \
  --perturb-after 0 \
  --epochs "${EPOCHS:-1}" --cycles "${CYCLES:-100}" \
  --pretrained-lr 0.00001 --lr 0.0001 "$@"
