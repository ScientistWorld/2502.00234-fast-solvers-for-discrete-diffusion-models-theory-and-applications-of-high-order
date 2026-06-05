#!/bin/bash
# Job script run by the scheduler inside the container.

set -euo pipefail

cd /home/user
bash scripts/reproduce.sh
