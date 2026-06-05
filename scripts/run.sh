#!/bin/bash
# Job script run by the scheduler inside the container.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
bash scripts/evaluate_train.sh outputs/toy scoring/scores_train.json
bash scripts/evaluate_test.sh outputs/toy scoring/scores_test.json scoring/scores_train.json
