#!/bin/bash
# Run the first-order tau-leaping baseline from the paper.

set -euo pipefail

cd /home/user

python baseline/run_tau.py \
  --output-dir outputs/toy \
  --samples "${SAMPLES:-1000000}" \
  --steps "${BASELINE_STEPS:-16,32,64,128}" \
  --horizon "${HORIZON:-12}" \
  --seed "${BASELINE_SEED:-23456}"
