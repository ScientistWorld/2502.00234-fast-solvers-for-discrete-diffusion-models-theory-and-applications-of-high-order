#!/bin/bash
# Run the first-order tau-leaping baseline from the paper.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python baseline/run_tau.py \
  --output-dir outputs/toy \
  --samples "${SAMPLES:-50000}" \
  --steps "${BASELINE_STEPS:-8,16,32,64}" \
  --horizon "${HORIZON:-12}" \
  --seed "${BASELINE_SEED:-23456}"
