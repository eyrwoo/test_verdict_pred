#!/usr/bin/env bash
# Setup script for LiveCodeBench actual execution environment.
#
# Clones the official LiveCodeBench repo (used for evaluation via sys.path injection).
# No pip install needed — evaluate.py adds LiveCodeBench/ to sys.path directly.
#
# Usage:
#   bash src/LiveCodeBench/actual_exec/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LCB_DIR="$SCRIPT_DIR/LiveCodeBench"

if [ -d "$LCB_DIR" ]; then
    echo "[setup] LiveCodeBench/ already exists, skipping clone."
else
    echo "[setup] Cloning LiveCodeBench..."
    git clone https://github.com/LiveCodeBench/LiveCodeBench.git "$LCB_DIR"
fi

echo "[setup] Installing LiveCodeBench dependencies..."
python3 -m pip install -r "$LCB_DIR/requirements.txt"

echo "[setup] Done."
