#!/bin/bash
# Evaluate saved solver outputs on the held-out test slice.

set -euo pipefail

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
mkdir -p scoring

OUTPUT_DIR="${1:-outputs/toy}"
SCORES="${2:-scoring/scores_test.json}"
TRAIN_SCORES="${3:-scoring/scores_train.json}"

python eval/test/evaluate_toy_split.py \
  --output-dir "${OUTPUT_DIR}" \
  --scores "${SCORES}" \
  --train-scores "${TRAIN_SCORES}"
