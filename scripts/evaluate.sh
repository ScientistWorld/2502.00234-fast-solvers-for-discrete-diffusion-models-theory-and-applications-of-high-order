#!/bin/bash
# Evaluate saved solver outputs and write scoring/scores.json.

set -euo pipefail

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
mkdir -p scoring

python eval/evaluate_toy.py \
  --output-dir outputs/toy \
  --scores scoring/scores.json
