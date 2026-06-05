#!/bin/bash
# Job script run by the scheduler inside the container.

set -euo pipefail

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
bash scripts/reproduce.sh
