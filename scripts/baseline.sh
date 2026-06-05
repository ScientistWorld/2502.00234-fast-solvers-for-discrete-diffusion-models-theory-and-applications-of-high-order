#!/bin/bash
# Run the first-order tau-leaping baseline from the paper.

set -euo pipefail

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

python baseline/run_tau.py \
  --output-dir outputs/toy \
  --samples "${SAMPLES:-50000}" \
  --steps "${BASELINE_STEPS:-8,16,32,64}" \
  --horizon "${HORIZON:-12}" \
  --seed "${BASELINE_SEED:-23456}"
