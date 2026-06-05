#!/bin/bash
# End-to-end reproduction for the finite-state core experiment.

set -euo pipefail

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/evaluate.sh
python validate.py --compare
