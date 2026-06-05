#!/bin/bash
# Run the paper's theta-RK-2 and theta-Trapezoidal solvers on the toy CTMC.

set -euo pipefail

cd /home/user

python method/run_toy.py \
  --output-dir outputs/toy \
  --samples "${SAMPLES:-1000000}" \
  --steps "${METHOD_STEPS:-16,32,64,128}" \
  --horizon "${HORIZON:-12}" \
  --theta "${THETA:-0.5}" \
  --seed "${METHOD_SEED:-12345}"
