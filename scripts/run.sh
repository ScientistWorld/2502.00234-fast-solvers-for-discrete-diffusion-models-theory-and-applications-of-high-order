#!/bin/bash
# Job script run by the scheduler inside the container.

set -euo pipefail

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
bash scripts/evaluate_train.sh outputs/toy scoring/scores_train.json
bash scripts/evaluate_test.sh outputs/toy scoring/scores_test.json scoring/scores_train.json
