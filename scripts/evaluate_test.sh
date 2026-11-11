#!/bin/bash
# Hidden test: run every solver registered under method/ and baseline/ on the
# held-out instances and write scoring/scores_test.json.  Takes no arguments.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p scoring

python eval/test/evaluate.py --scores scoring/scores_test.json "$@"
