#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
OUTPUT="${OUTPUT:-outputs/e2e_dispatch_v14_$(date +%Y%m%d_%H%M%S)}"
python -u scripts/train_e2e_single.py \
  --output "$OUTPUT" --device "${DEVICE:-cuda}" \
  --initialization sft --regression-weight "${REGRESSION_WEIGHT:-0.2}" \
  --schedule-init earliest_finish \
  --workers "${WORKERS:-16}" --branches "${BRANCHES:-16}" \
  --roots-per-cycle 128 --load-weight "${LOAD_WEIGHT:-0.2}" --decision-batch "${DECISION_BATCH:-16}" \
  --trace-hops "${TRACE_HOPS:-6}" \
  --load-share "${LOAD_SHARE:-0.30}" \
  --horizon 10 --episode-steps 200 --long-every 0 \
  --adaptive-after "${ADAPTIVE_AFTER:-2}" --long-horizon "${LONG_HORIZON:-20}" \
  --perturb-after "${PERTURB_AFTER:-3}" \
  --epochs "${EPOCHS:-1}" --cycles "${CYCLES:-100}" \
  --pretrained-lr 0.00001 --lr 0.0001 "$@"
