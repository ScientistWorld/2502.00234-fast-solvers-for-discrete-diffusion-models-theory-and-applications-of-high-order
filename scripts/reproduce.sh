#!/bin/bash
# End-to-end reproduction for the finite-state core experiment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/evaluate.sh
python validate.py --compare
