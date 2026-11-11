#!/bin/bash
# Job script run by the scheduler inside the container.
# Current job: calibration and smoke test of the hardened multi-site gym.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p hardening/calibration
python hardening/calibrate.py 12000 2>&1 | tee hardening/calibration/log.txt
