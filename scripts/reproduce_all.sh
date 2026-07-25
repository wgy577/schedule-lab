#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
export CAUSAL_SCHEDULE_LAB_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON" ]]; then
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/python" -m pip install --upgrade pip
  "$ROOT/.venv/bin/python" -m pip install "$ROOT[dev]"
fi

"$PYTHON" -m compileall -q "$ROOT/src"
"$PYTHON" -m pytest -q "$ROOT/tests"
"$PYTHON" -m causal_schedule_lab.cli compile-semantics \
  --project-root "$ROOT" \
  --output "$ROOT/outputs/semantic_compilation.json"

for family in jsp fsp fjsp hfsp; do
  "$PYTHON" -m causal_schedule_lab.cli demo \
    --family "$family" \
    --mode optimize \
    --iterations 1 \
    --candidate-budget 2 \
    --output "$ROOT/outputs/${family}_optimization.json" \
    --audit-log "$ROOT/outputs/${family}_experiments.jsonl"
done
