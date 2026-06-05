#!/bin/bash
# Evaluate saved solver outputs and write scoring/scores.json.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p scoring

python eval/evaluate_toy.py \
  --output-dir outputs/toy \
  --scores scoring/scores.json
