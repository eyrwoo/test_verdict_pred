#!/usr/bin/env bash
# Setup script for BigCodeBench_Hard actual execution environment.
#
# Clones the official BigCodeBench repo, installs it in editable mode,
# then replaces two files with our modified versions that add per-test-case
# timing (time_breakdown) required for test case prioritization.
#
# Usage:
#   bash src/BigCodeBench_Hard/actual_exec/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BCB_DIR="$SCRIPT_DIR/bigcodebench"
OVERRIDES_DIR="$SCRIPT_DIR/bcb_overrides"

if [ -d "$BCB_DIR" ]; then
    echo "[setup] bigcodebench/ already exists, skipping clone."
else
    echo "[setup] Cloning bigcodebench..."
    git clone https://github.com/bigcode-project/bigcodebench.git "$BCB_DIR"
fi

echo "[setup] Installing bigcodebench in editable mode..."
python3 -m pip install -e "$BCB_DIR"

echo "[setup] Applying overrides..."
cp "$OVERRIDES_DIR/eval/__init__.py" "$BCB_DIR/bigcodebench/eval/__init__.py"
cp "$OVERRIDES_DIR/evaluate.py"      "$BCB_DIR/bigcodebench/evaluate.py"

echo "[setup] Done."
