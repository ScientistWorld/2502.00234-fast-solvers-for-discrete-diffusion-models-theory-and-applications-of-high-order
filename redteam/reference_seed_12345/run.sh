#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
python redteam/harness.py reference_seed_12345
