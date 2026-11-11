#!/bin/bash
# First-order tau-leaping baseline (baseline/tau_leaping.py), scored on the
# public train instances.  Needs no training.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p outputs
bash scripts/evaluate_train.sh --only tau_leaping --scores outputs/baseline_train_scores.json
