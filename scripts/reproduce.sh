#!/bin/bash
# End-to-end reproduction: score every registered solver on the train slice
# (scores_train.json, copied to scores.json) and, when the held-out evaluator
# is present, on the test slice (scores_test.json).  No training is involved.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
if [ -f eval/test/evaluate.py ]; then
  bash scripts/evaluate_test.sh
fi
python validate.py
