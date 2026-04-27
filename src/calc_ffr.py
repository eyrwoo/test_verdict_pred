"""
First Failure Rank (FFR) calculator for BigCodeBench-Hard (bcb) and LiveCodeBench (lcb).

FFR = rank (1-indexed) of the first failing test case in the prioritized tc_order.
Lower is better: a perfect prioritization puts all failures first (FFR = 1).

Only "mixed" codes (at least one pass AND at least one fail) are counted.
Problems are also classified:
  all_pass : no code has any failure           → skipped
  all_fail : no code has both pass and fail    → skipped
  mixed    : at least one code is mixed        → scored

Benchmark FFR = average over per-problem FFRs
Per-problem FFR = average FFR of all mixed codes in that problem

Dataset differences
-------------------
bcb: overall only, no difficulty breakdown
lcb: overall + per-difficulty (easy/medium/hard), compile_failure skipped

Usage
-----
  python src/calc_ffr.py --dataset bcb --prioritization-dir src/results/bcb/prioritization/random
  python src/calc_ffr.py --dataset lcb --prioritization-dir src/results/lcb/prioritization/random
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SRC_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SRC_DIR / "results"

DEFAULT_PRIORITIZATION_DIR = {
    "bcb": str(RESULTS_DIR / "bcb" / "prioritization" / "random"),
    "lcb": str(RESULTS_DIR / "lcb" / "prioritization" / "random"),
}

LCB_EVAL_JSON = str(
    SRC_DIR / "LiveCodeBench" / "actual_exec" / "results" / "filtered_output_eval_all.json"
)


# ------------------------------------------------------------------
# Per-problem FFR calculator
# ------------------------------------------------------------------

def ffr_per_problem(data: Dict) -> Optional[float]:
    """Return average FFR over all mixed codes in a problem, or None if no mixed code."""
    tc_names = [tc["name"] if isinstance(tc, dict) else tc for tc in data["tc_order"]]
    tc_position: Dict[str, int] = {tc: i + 1 for i, tc in enumerate(tc_names)}

    ffr_values: List[int] = []
    for code in data["codes_results"]:
        statuses = [tc["status"] for tc in code["tc_results"]]
        # skip all-pass (no fault) and all-fail (ordering irrelevant)
        if all(s == "pass" for s in statuses) or all(s == "fail" for s in statuses):
            continue
        # Find position of first failure in tc_order
        first_pos: Optional[int] = None
        for tc_result in code["tc_results"]:
            if tc_result["status"] == "fail":
                pos = tc_position[tc_result["name"]]
                if first_pos is None or pos < first_pos:
                    first_pos = pos
        if first_pos is not None:
            ffr_values.append(first_pos)

    if not ffr_values:
        return None
    return sum(ffr_values) / len(ffr_values)


# ------------------------------------------------------------------
# Problem-level classifier (same as calc_apfd.py)
# ------------------------------------------------------------------

def classify_problem(data: Dict) -> str:
    has_any_fail = False
    for code in data["codes_results"]:
        statuses = {tc["status"] for tc in code["tc_results"]}
        if "fail" in statuses:
            has_any_fail = True
            if "pass" in statuses:
                return "mixed"
    return "all_fail" if has_any_fail else "all_pass"


def count_codes_by_type(data: Dict) -> Tuple[int, int, int]:
    n_mixed = n_all_pass = n_all_fail = 0
    for code in data["codes_results"]:
        statuses = {tc["status"] for tc in code["tc_results"]}
        if "pass" in statuses and "fail" in statuses:
            n_mixed += 1
        elif "fail" in statuses:
            n_all_fail += 1
        else:
            n_all_pass += 1
    return n_mixed, n_all_pass, n_all_fail


# ------------------------------------------------------------------
# Batch computation
# ------------------------------------------------------------------

def compute_ffr_for_subset(
    result_files,
    check_compile_failure: bool = False,
    difficulty_map: Optional[Dict[str, str]] = None,
    target_difficulty: Optional[str] = None,
) -> Tuple[List[Tuple[str, float]], List[str], List[str], List[str], Dict[str, int]]:
    """Returns (ffr_values, skipped_all_pass, skipped_compile, skipped_all_fail, code_counts)."""
    ffr_values: List[Tuple[str, float]] = []
    skipped_all_pass: List[str] = []
    skipped_compile: List[str] = []
    skipped_all_fail: List[str] = []
    code_counts: Dict[str, int] = {"mixed": 0, "all_pass": 0, "all_fail": 0}

    for path in result_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        task_id = str(data.get("task_id", path.stem))

        if difficulty_map is not None and target_difficulty is not None:
            if difficulty_map.get(task_id) != target_difficulty:
                continue

        if check_compile_failure and data.get("compile_failure"):
            skipped_compile.append(task_id)
            continue

        problem_class = classify_problem(data)
        nm, nap, naf = count_codes_by_type(data)
        code_counts["mixed"] += nm
        code_counts["all_pass"] += nap
        code_counts["all_fail"] += naf

        if problem_class == "all_pass":
            skipped_all_pass.append(task_id)
            continue
        if problem_class == "all_fail":
            skipped_all_fail.append(task_id)
            continue

        ffr = ffr_per_problem(data)
        if ffr is not None:
            ffr_values.append((task_id, ffr))

    return ffr_values, skipped_all_pass, skipped_compile, skipped_all_fail, code_counts


def load_lcb_difficulty_map(eval_json_path: str) -> Dict[str, str]:
    with open(eval_json_path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(item["question_id"]): item["difficulty"] for item in data}


def format_section(
    label: str,
    ffr_values: List[Tuple[str, float]],
    skipped_all_pass: List[str],
    skipped_compile: List[str],
    skipped_all_fail: List[str],
    code_counts: Dict[str, int],
    verbose: bool,
) -> List[str]:
    lines: List[str] = [f"=== {label} ==="]
    if verbose:
        lines.append("Per-problem FFR:")
        for task_id, ffr in ffr_values:
            lines.append(f"  {task_id}: {ffr:.4f}")
        if skipped_all_pass:
            lines.append("Skipped (all pass):")
            for task_id in skipped_all_pass:
                lines.append(f"  {task_id}")
        if skipped_all_fail:
            lines.append("Skipped (all fail):")
            for task_id in skipped_all_fail:
                lines.append(f"  {task_id}")
        if skipped_compile:
            lines.append("Skipped (compile failure):")
            for task_id in skipped_compile:
                lines.append(f"  {task_id}")
        lines.append("")

    if not ffr_values:
        lines.append("No mixed problems found. No FFR to report.")
    else:
        benchmark_ffr = sum(v for _, v in ffr_values) / len(ffr_values)
        total_codes = sum(code_counts.values())
        lines += [
            f"Benchmark FFR               : {benchmark_ffr:.4f}",
            f"Problems included (mixed)   : {len(ffr_values)}",
            f"Problems skipped (all pass) : {len(skipped_all_pass)}",
            f"Problems skipped (all fail) : {len(skipped_all_fail)}",
            f"--- code-level (total {total_codes}) ---",
            f"  Codes used   (mixed)   : {code_counts['mixed']}",
            f"  Codes skipped (all pass): {code_counts['all_pass']}",
            f"  Codes skipped (all fail): {code_counts['all_fail']}",
        ]
        if skipped_compile:
            lines.append(f"Problems skipped (compile failure): {len(skipped_compile)}")
    return lines


def section_dict(
    vals: List[Tuple[str, float]],
    all_pass: List[str],
    compile_fail: List[str],
    all_fail: List[str],
    code_counts: Dict[str, int],
) -> dict:
    score = sum(v for _, v in vals) / len(vals) if vals else None
    return {
        "score": score,
        "codes_mixed": code_counts["mixed"],
        "codes_all_pass": code_counts["all_pass"],
        "codes_all_fail": code_counts["all_fail"],
        "per_problem": {task_id: v for task_id, v in vals},
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute First Failure Rank (FFR) from TCP prioritization results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["bcb", "lcb"],
        required=True,
        help="Dataset: bcb (BigCodeBench-Hard) or lcb (LiveCodeBench)",
    )
    parser.add_argument(
        "--prioritization-dir",
        default=None,
        help="Directory containing per-problem prioritization JSON files "
             "(default: src/results/<dataset>/prioritization/random)",
    )
    parser.add_argument(
        "--eval-json",
        default=LCB_EVAL_JSON,
        help="[lcb only] Path to filtered_output_eval_all.json with difficulty labels",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-problem FFR values",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save report as .json "
             "(default: src/results/<dataset>/metrics/<rel>/ffr.json)",
    )
    args = parser.parse_args()

    pri_dir = Path(args.prioritization_dir or DEFAULT_PRIORITIZATION_DIR[args.dataset])
    result_files = sorted(pri_dir.glob("*.json"))
    if not result_files:
        raise FileNotFoundError(f"No JSON files found in {pri_dir}")

    is_lcb = args.dataset == "lcb"
    check_compile = is_lcb

    # overall
    overall_values, overall_all_pass, overall_compile, overall_all_fail, overall_codes = \
        compute_ffr_for_subset(result_files, check_compile_failure=check_compile)

    all_lines: List[str] = [f"Dataset: {args.dataset}", "Metric: First Failure Rank (FFR)", ""]
    all_lines += format_section(
        "overall", overall_values, overall_all_pass, overall_compile,
        overall_all_fail, overall_codes, args.verbose,
    )

    result: dict = {
        "dataset": args.dataset,
        "metric": "ffr",
        "overall": section_dict(overall_values, overall_all_pass, overall_compile,
                                overall_all_fail, overall_codes),
    }

    # per-difficulty (lcb only)
    if is_lcb:
        difficulty_map = load_lcb_difficulty_map(args.eval_json)
        per_diff: Dict[str, dict] = {}
        for diff in ("easy", "medium", "hard"):
            values, all_pass, skipped_compile, all_fail, codes = compute_ffr_for_subset(
                result_files,
                check_compile_failure=check_compile,
                difficulty_map=difficulty_map,
                target_difficulty=diff,
            )
            all_lines += ["", ""]
            all_lines += format_section(diff, values, all_pass, skipped_compile,
                                        all_fail, codes, args.verbose)
            per_diff[diff] = section_dict(values, all_pass, skipped_compile, all_fail, codes)
        result.update(per_diff)

    print("\n".join(all_lines))

    # resolve output path
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix != ".json":
            output_path = output_path.with_suffix(".json")
    else:
        metrics_base = RESULTS_DIR / args.dataset / "metrics"
        pri_base = RESULTS_DIR / args.dataset / "prioritization"
        try:
            rel = pri_dir.resolve().relative_to(pri_base.resolve())
        except ValueError:
            rel = Path(pri_dir.name)
        output_path = metrics_base / rel / "ffr.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
