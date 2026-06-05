#!/bin/bash
# Run the paper's theta-RK-2 and theta-Trapezoidal solvers on the toy CTMC.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python method/run_toy.py \
  --output-dir outputs/toy \
  --samples "${SAMPLES:-50000}" \
  --steps "${METHOD_STEPS:-8,16,32,64}" \
  --horizon "${HORIZON:-12}" \
  --theta "${THETA:-0.5}" \
  --seed "${METHOD_SEED:-12345}"
