"""
Wilcoxon signed-rank test for TCP method comparisons.

For each VP method, compare its per-problem scores against a baseline
using the Wilcoxon signed-rank test (two-sided, then check direction).

Input:  src/results/{dataset}/metrics/**/apfd.json  (or ffr.json)
        Each JSON must contain a "per_problem" dict  { task_id: score }.
        Run calc_apfd.sh / calc_ffr.sh first to (re)generate these files.

Output: src/results/{dataset}/wilcoxon/wilcoxon_{metric}_{dataset}.csv

CSV columns
-----------
  prompt_type, tcp_method, model,
  baseline,                   # which baseline was compared against
  n_pairs,                    # number of problems present in both
  stat,                       # Wilcoxon W statistic
  p_value,                    # two-sided p-value
  significant,                # True if p < 0.05
  direction                   # "better", "worse", or "no_diff" (median sign)

Usage
-----
  python src/calc_wilcoxon.py --dataset bcb --metric apfd
  python src/calc_wilcoxon.py --dataset bcb --metric ffr
  python src/calc_wilcoxon.py --dataset lcb --metric apfd --level hard
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from scipy.stats import wilcoxon
except ImportError:
    raise SystemExit("scipy is required: pip install scipy")

SRC_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SRC_DIR / "results"

# ------------------------------------------------------------------
# Configuration (mirrors apfd2csv.py / ffr2csv.py)
# ------------------------------------------------------------------

BASELINES: List[Tuple[str, str]] = [
    ("original", "original"),
    ("random",   "random"),
    ("ncd",      "ncd_lzma"),
    ("LEN",      "input_len_desc"),
]

VP_VARIANTS  = ["vp_original", "vp_random", "vp_length", "vp_token_input_desc", "vp_token_output_desc", "vp_token_output_asc"]
PROMPT_TYPES = ["direct_verdict", "reasoned_verdict", "failure_analysis"]

MODEL_DISPLAY: Dict[str, str] = {
    "qwen3-coder-30B-A3B-instruct": "Qwen3-coder",
    "claude-haiku-4-5-20251001":    "Claude-haiku",
    "gpt-5-mini-2025-08-07":        "GPT-5-mini",
}
MODELS = list(MODEL_DISPLAY.keys())

ALPHA = 0.05


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_per_problem(path: Path, level: str = "overall") -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get(level, {}).get("per_problem")


def run_wilcoxon(
    scores_a: Dict[str, float],
    scores_b: Dict[str, float],
    higher_is_better: bool,
) -> Optional[dict]:
    """
    Compare scores_a (VP method) vs scores_b (baseline).
    Returns None if insufficient paired data.

    direction: "better" / "worse" / "no_diff"  based on median_diff
    effect_size: rank-biserial correlation = (n_better - n_worse) / n_pairs
    """
    import statistics

    # Use VP (scores_a) as reference set; baseline may cover more problems
    common = sorted(k for k in scores_a if k in scores_b)
    if len(common) < 2:
        return None

    a_vals = [scores_a[k] for k in common]
    b_vals = [scores_b[k] for k in common]
    diffs  = [a - b for a, b in zip(a_vals, b_vals)]

    n_better = sum(1 for d in diffs if d > 0)
    n_worse  = sum(1 for d in diffs if d < 0)
    n_tied   = sum(1 for d in diffs if d == 0)

    # Drop zero differences (required by Wilcoxon)
    nz_diffs = [d for d in diffs if d != 0]
    if len(nz_diffs) < 2:
        return {
            "n_pairs":     len(common),
            "n_better":    n_better,
            "n_worse":     n_worse,
            "n_tied":      n_tied,
            "median_diff": round(statistics.median(diffs), 6),
            "effect_size": None,
            "stat":        None,
            "p_value":     None,
            "significant": False,
            "direction":   "no_diff",
        }

    stat, p_value = wilcoxon(nz_diffs, alternative="two-sided")

    median_diff  = statistics.median(diffs)
    effect_size  = (n_better - n_worse) / len(common)  # rank-biserial correlation

    if higher_is_better:
        direction = "better" if median_diff > 0 else ("worse" if median_diff < 0 else "no_diff")
    else:
        direction = "better" if median_diff < 0 else ("worse" if median_diff > 0 else "no_diff")

    return {
        "n_pairs":     len(common),
        "n_better":    n_better,
        "n_worse":     n_worse,
        "n_tied":      n_tied,
        "median_diff": round(median_diff, 6),
        "effect_size": round(effect_size, 4),
        "stat":        round(stat, 4),
        "p_value":     round(p_value, 6),
        "significant": bool(p_value < ALPHA),
        "direction":   direction,
    }


def fmt_opt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wilcoxon signed-rank test for TCP method comparisons",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["bcb", "lcb"], required=True)
    parser.add_argument(
        "--metric", choices=["apfd", "failing_test_based", "ffr"], default="apfd",
        help="Which metric JSON to read (apfd / failing_test_based / ffr)",
    )
    parser.add_argument(
        "--level", default="overall",
        help="Score level to compare: overall, easy, medium, hard",
    )
    parser.add_argument(
        "--baselines", nargs="+", default=["original", "random", "ncd"],
        help="Which baselines to compare VP methods against",
    )
    args = parser.parse_args()

    metrics_root = RESULTS_DIR / args.dataset / "metrics"
    metric_file  = f"{args.metric}.json"
    higher_is_better = (args.metric != "ffr")  # FFR: lower is better

    # Load baseline per-problem scores
    baseline_scores: Dict[str, Dict[str, float]] = {}
    for display_name, dir_name in BASELINES:
        if display_name not in args.baselines:
            continue
        pp = load_per_problem(metrics_root / dir_name / metric_file, args.level)
        if pp is None:
            print(f"[warn] {display_name}: {metric_file} missing or no per_problem data. "
                  f"Re-run calc_apfd.sh / calc_ffr.sh first.")
        else:
            baseline_scores[display_name] = pp

    if not baseline_scores:
        raise SystemExit("No baseline data loaded. Aborting.")

    all_cols = [
        "prompt_type", "tcp_method", "model", "baseline",
        "n_pairs", "n_better", "n_worse", "n_tied",
        "median_diff", "effect_size",
        "p_value", "significant", "direction",
    ]
    rows: List[Dict[str, str]] = []

    for prompt_type in PROMPT_TYPES:
        for model_dir in MODELS:
            for variant in VP_VARIANTS:
                path = metrics_root / variant / prompt_type / model_dir / metric_file
                pp = load_per_problem(path, args.level)
                if pp is None:
                    continue
                for baseline_name, baseline_pp in baseline_scores.items():
                    result = run_wilcoxon(pp, baseline_pp, higher_is_better)
                    if result is None:
                        continue
                    rows.append({
                        "prompt_type": prompt_type,
                        "tcp_method":  variant,
                        "model":       MODEL_DISPLAY[model_dir],
                        "baseline":    baseline_name,
                        "n_pairs":     str(result["n_pairs"]),
                        "n_better":    str(result["n_better"]),
                        "n_worse":     str(result["n_worse"]),
                        "n_tied":      str(result["n_tied"]),
                        "median_diff": fmt_opt(result["median_diff"]),
                        "effect_size": fmt_opt(result["effect_size"]),
                        "p_value":     fmt_opt(result["p_value"]),
                        "significant": str(result["significant"]),
                        "direction":   result["direction"],
                    })

    out_dir = RESULTS_DIR / args.dataset / "wilcoxon"
    out_dir.mkdir(parents=True, exist_ok=True)
    level_suffix = f"_{args.level}" if args.level != "overall" else ""
    out_path = out_dir / f"wilcoxon_{args.metric}{level_suffix}_{args.dataset}.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_path}  ({len(rows)} comparisons)")


if __name__ == "__main__":
    main()
