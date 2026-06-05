#!/bin/bash
# Download all datasets/models needed by this reproduction.
#
# Estimated download size: 0 bytes.
# Estimated disk usage after extraction: 0 bytes.
#
# The core toy-model reproduction is analytic and needs no external assets.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p data/downloads
echo "No external downloads are required for the analytic toy-model experiment."
