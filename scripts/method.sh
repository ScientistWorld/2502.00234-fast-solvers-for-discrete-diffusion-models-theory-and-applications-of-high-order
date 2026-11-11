#!/bin/bash
# The paper's samplers need no training.  They are registered in
# method/fast_solvers/samplers.py (SOLVERS) and run by the evaluators, which
# own the problem instances and count intensity evaluations.  This script
# scores only the paper's samplers on the public train instances.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p outputs
bash scripts/evaluate_train.sh --only theta_rk2 theta_trapezoidal --scores outputs/method_train_scores.json
