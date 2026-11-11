#!/bin/bash
# Job script (calibration of the multi-mode product family at d = 48).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p hardening/calibration hardening/audit
python hardening/oracle_check.py 2>&1 | tee hardening/calibration/products_oracle_check.txt
python hardening/calibrate_family.py 10000 2>&1 | tee hardening/calibration/products_log.txt
