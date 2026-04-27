#!/usr/bin/env bash
# BigCodeBench-Hard Code Generation → Execution Evaluation Pipeline
#
# Prerequisites:
#   - setup.sh execution completed → bigcodebench installed
#   - .env file has MODEL_BASE_URL configured
#
# Usage:
#   bash src/BigCodeBench_Hard/actual_exec/actual_exec.sh [options]
#
# Options:
#   --strategy   greedy|nucleus  (default: nucleus)
#   --n_samples  N               (default: 10)
#   --parallel   N               (default: CPU count / 2)
#   --limit      N               for dry-run (default: no limit)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="qwen3-coder-30B-A3B-instruct"
RESULTS_DIR="$SCRIPT_DIR/results/$MODEL"
STRATEGY="nucleus"
N_SAMPLES=10
PARALLEL=$(python3 -c 'import os; print(max(1, os.cpu_count() // 2))')
LIMIT=""

# ── arguments parsing ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --strategy)  STRATEGY="$2";  shift 2 ;;
        --n_samples) N_SAMPLES="$2"; shift 2 ;;
        --parallel)  PARALLEL="$2";  shift 2 ;;
        --limit)     LIMIT="$2";     shift 2 ;;
        *) echo "[actual_exec] Unknown argument: $1"; exit 1 ;;
    esac
done

SAMPLES_JSONL="$RESULTS_DIR/${STRATEGY}_samples.jsonl"

# ── 1. bigcodebench installation check ────────────────────────────────────────────────
if ! python3 -c "import bigcodebench" 2>/dev/null; then
    echo "[actual_exec] bigcodebench not installed. Running setup.sh..."
    bash "$SCRIPT_DIR/setup.sh"
fi

# ── 2. Code Generation ─────────────────────────────────────────────────────────────
echo "[actual_exec] Generating codes (strategy=$STRATEGY, n_samples=$N_SAMPLES)..."

GENERATE_ARGS=(
    --sampling  "$STRATEGY"
    --n_samples "$N_SAMPLES"
    --out_dir   "$RESULTS_DIR/.."
)
[[ -n "$LIMIT" ]] && GENERATE_ARGS+=(--limit "$LIMIT")

python3 "$SCRIPT_DIR/code_generate.py" "${GENERATE_ARGS[@]}"

# ── 3. BCB Evaluation ─────────────────────────────────────────────────────────
echo "[actual_exec] Evaluating $SAMPLES_JSONL (parallel=$PARALLEL)..."

python3 -m bigcodebench.evaluate \
    --split     complete \
    --subset    hard \
    --samples   "$SAMPLES_JSONL" \
    --execution local \
    --pass_k    1,5,10 \
    --parallel  "$PARALLEL" \
    --no_gt \
    --calibrated False

EVAL_RESULTS="${SAMPLES_JSONL/.jsonl/_eval_results.json}"
echo "[actual_exec] Done. Results → $EVAL_RESULTS"
