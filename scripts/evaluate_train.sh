#!/bin/bash
# Train slice: run every solver registered under method/ and baseline/ on the
# public instances and write scoring/scores_train.json.
# Optional flags: --only NAME [NAME ...]   restrict to some method keys.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p scoring

python eval/train/evaluate.py --scores scoring/scores_train.json "$@"
