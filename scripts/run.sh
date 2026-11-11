#!/bin/bash
# Job script run by the scheduler inside the container (no internet, 1 CPU).
#
# Reproduces the reference end to end (no training is involved: the paper's
# samplers and the tau-leaping baseline are run by the evaluators), then runs
# the hardening audit: oracle exactness, discovery/failure demo, shortcut,
# seed-noise and variant attempts through the hidden test, a train-side check,
# the rank analysis and the unscored paper Fig. 2 toy.  Audit steps resume
# from saved results and do not stop the job when they fail.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p hardening/audit/logs outputs

python hardening/oracle_check.py

# Reference runs: every registered solver on the hidden test and the train slice.
if [ ! -f outputs/.test_done ]; then
  bash scripts/evaluate_test.sh 2> hardening/audit/logs/evaluate_test.log
  touch outputs/.test_done
fi
if [ ! -f outputs/.train_done ]; then
  bash scripts/evaluate_train.sh 2> hardening/audit/logs/evaluate_train.log
  cp scoring/scores_train.json scoring/scores.json
  touch outputs/.train_done
fi

bash hardening/discovery_demo.sh || echo "discovery demo failed"
python hardening/run_audit.py test shortcuts variants seeds 2> hardening/audit/logs/audit_test.log || echo "test audit failed"
python hardening/analyze.py || echo "analysis failed"
python hardening/run_audit.py train train_variants 2> hardening/audit/logs/audit_train.log || echo "train audit failed"
python hardening/paper_toy.py || echo "paper toy failed"
