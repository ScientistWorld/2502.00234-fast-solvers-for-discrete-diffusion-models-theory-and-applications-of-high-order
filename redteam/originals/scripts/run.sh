#!/bin/bash
# Job script run by the scheduler inside the container.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

rm -rf outputs/toy
SAMPLES="${SAMPLES:-50000}" \
BASELINE_STEPS="${BASELINE_STEPS:-8,16,32,64}" \
METHOD_STEPS="${METHOD_STEPS:-8,16,32,64}" \
HORIZON="${HORIZON:-12}" \
THETA="${THETA:-0.5}" \
bash scripts/reproduce.sh

bash scripts/evaluate_train.sh outputs/toy scoring/scores_train.json
bash scripts/evaluate_test.sh outputs/toy scoring/scores_test.json scoring/scores_train.json
