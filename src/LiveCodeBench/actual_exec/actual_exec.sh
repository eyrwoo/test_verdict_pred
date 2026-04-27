#!/usr/bin/env bash
# LiveCodeBench Code Generation → Execution Evaluation Pipeline
#
# Prerequisites:
#   - setup.sh execution completed → LiveCodeBench repo cloned
#   - .env file has MODEL_BASE_URL configured
#
# Usage:
#   bash src/LiveCodeBench/actual_exec/actual_exec.sh [options]
#
# Options:
#   --n_samples  N   (default: 10)
#   --parallel   N   (default: CPU count / 2)
#   --limit      N   for dry-run (default: no limit)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="qwen3-coder-30B-A3B-instruct"
RESULTS_DIR="$SCRIPT_DIR/results/$MODEL"
N_SAMPLES=10
PARALLEL=$(python3 -c 'import os; print(max(1, os.cpu_count() // 2))')
LIMIT=""

# ── arguments parsing ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_samples) N_SAMPLES="$2"; shift 2 ;;
        --parallel)  PARALLEL="$2";  shift 2 ;;
        --limit)     LIMIT="$2";     shift 2 ;;
        *) echo "[actual_exec] Unknown argument: $1"; exit 1 ;;
    esac
done

SAMPLES_JSONL="$RESULTS_DIR/nucleus_samples.jsonl"

# ── 1. LiveCodeBench installation check ──────────────────────────────────────────────
if [ ! -d "$SCRIPT_DIR/LiveCodeBench" ]; then
    echo "[actual_exec] LiveCodeBench not found. Running setup.sh..."
    bash "$SCRIPT_DIR/setup.sh"
fi

# ── 2. Code Generation ───────────────────────────────────────────────────────────────
echo "[actual_exec] Generating codes (n_samples=$N_SAMPLES)..."

GENERATE_ARGS=(
    --n_samples "$N_SAMPLES"
    --out_dir   "$SCRIPT_DIR/results"
)
[[ -n "$LIMIT" ]] && GENERATE_ARGS+=(--limit "$LIMIT")

python3 "$SCRIPT_DIR/code_generate.py" "${GENERATE_ARGS[@]}"

# ── 3. LCB Evaluation ────────────────────────────────────────────────────────────────
echo "[actual_exec] Evaluating $SAMPLES_JSONL (parallel=$PARALLEL)..."

python3 "$SCRIPT_DIR/evaluate.py" \
    --samples  "$SAMPLES_JSONL" \
    --parallel "$PARALLEL"

echo "[actual_exec] Done. Results → $RESULTS_DIR"
