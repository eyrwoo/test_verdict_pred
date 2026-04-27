"""
Unified APFD calculator for BigCodeBench-Hard (bcb) and LiveCodeBench (lcb).

APFD = 1 - (TF1 + TF2 + ... + TFm) / (n * m) + 1 / (2n)

Two fault-type modes
--------------------
apfd  (default)
  m  : number of faulty codes (codes with at least one failing TC)
  TFi: position of the *first* failing TC for faulty code i

failing_test_based
  Per code: m = number of failing TCs for that code
            TFi = position of each failing TC in tc_order
            score = 1 - sum(TFi)/(n*m) + 1/(2n)   [skipped if m==0]
  Problem score = average over codes that have at least one failing TC

Exclusion rules (applied in order):
  1. compile_failure  (lcb only)
  2. all_pass  : every code passes every TC  (m=0, no failures at all)
  3. all_fail  : every code fails every TC   (ordering has no discriminating power)
  Only "mixed" problems (at least one code has both passing and failing TCs) are scored.

Dataset differences
-------------------
bcb: overall only, no difficulty breakdown, no compile_failure field
lcb: overall + per-difficulty (easy/medium/hard), compile_failure skipped

Usage
-----
  python src/calc_apfd.py --dataset bcb --prioritization-dir src/results/bcb/prioritization/random
  python src/calc_apfd.py --dataset lcb --prioritization-dir src/results/lcb/prioritization/random
"""
from __future__ import annotations

import argparse
import json
import sys
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
# Per-problem APFD calculators
# ------------------------------------------------------------------

def apfd_apfd(data: Dict) -> Optional[float]:
    n: int = data["num_test_cases"]
    tc_names = [tc["name"] if isinstance(tc, dict) else tc for tc in data["tc_order"]]
    tc_position: Dict[str, int] = {tc: i + 1 for i, tc in enumerate(tc_names)}

    first_fail_positions: List[int] = []
    for code in data["codes_results"]:
        statuses = [tc["status"] for tc in code["tc_results"]]
        # skip all-pass (no fault) and all-fail (ordering irrelevant)
        if all(s == "pass" for s in statuses) or all(s == "fail" for s in statuses):
            continue
        first_pos: Optional[int] = None
        for tc_result in code["tc_results"]:
            if tc_result["status"] == "fail":
                pos = tc_position[tc_result["name"]]
                if first_pos is None or pos < first_pos:
                    first_pos = pos
        if first_pos is not None:
            first_fail_positions.append(first_pos)

    m = len(first_fail_positions)
    if m == 0:
        return None
    return 1.0 - sum(first_fail_positions) / (n * m) + 1.0 / (2 * n)


def apfd_failing_test_based(data: Dict) -> Optional[float]:
    n: int = data["num_test_cases"]
    tc_names = [tc["name"] if isinstance(tc, dict) else tc for tc in data["tc_order"]]
    tc_position: Dict[str, int] = {tc: i + 1 for i, tc in enumerate(tc_names)}

    code_scores: List[float] = []
    for code in data["codes_results"]:
        statuses = [tc["status"] for tc in code["tc_results"]]
        # skip all-pass (no fault) and all-fail (ordering irrelevant)
        if all(s == "pass" for s in statuses) or all(s == "fail" for s in statuses):
            continue
        fail_positions: List[int] = [
            tc_position[tc_result["name"]]
            for tc_result in code["tc_results"]
            if tc_result["status"] == "fail"
        ]
        m = len(fail_positions)
        if m == 0:
            continue
        score = 1.0 - sum(fail_positions) / (n * m) + 1.0 / (2 * n)
        code_scores.append(score)

    if not code_scores:
        return None
    return sum(code_scores) / len(code_scores)


APFD_FUNCS = {
    "apfd": apfd_apfd,
    "failing_test_based": apfd_failing_test_based,
}


# ------------------------------------------------------------------
# Problem-level classifier
# ------------------------------------------------------------------

def classify_problem(data: Dict) -> str:
    """
    Classify a problem for APFD filtering.

    Returns
    -------
    "all_pass" : no code has any failure  (nothing to prioritize)
    "all_fail" : at least one code fails, but no code has a mix of
                 pass and fail TCs  (TC ordering has no discriminating power)
    "mixed"    : at least one code has both passing and failing TCs
    """
    has_any_fail = False
    for code in data["codes_results"]:
        statuses = {tc["status"] for tc in code["tc_results"]}
        if "fail" in statuses:
            has_any_fail = True
            if "pass" in statuses:   # found a mixed code → stop early
                return "mixed"
    return "all_fail" if has_any_fail else "all_pass"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_lcb_difficulty_map(eval_json_path: str) -> Dict[str, str]:
    with open(eval_json_path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(item["question_id"]): item["difficulty"] for item in data}


def count_codes_by_type(data: Dict) -> Tuple[int, int, int]:
    """Return (n_mixed, n_all_pass, n_all_fail) for all codes in a problem."""
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


def compute_apfd_for_subset(
    result_files,
    apfd_fn,
    check_compile_failure: bool = False,
    difficulty_map: Optional[Dict[str, str]] = None,
    target_difficulty: Optional[str] = None,
) -> Tuple[List[Tuple[str, float]], List[str], List[str], List[str], Dict[str, int]]:
    """Returns (apfd_values, skipped_all_pass, skipped_compile, skipped_all_fail, code_counts).

    code_counts: code-level breakdown across ALL processed problems
      {
        "mixed":    # codes with mixed pass/fail  (used in APFD)
        "all_pass": # codes with all-pass          (excluded from APFD)
        "all_fail": # codes with all-fail          (excluded from APFD)
      }
    """
    apfd_values: List[Tuple[str, float]] = []
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
        if problem_class == "all_pass":
            skipped_all_pass.append(task_id)
            # still count codes
            nm, nap, naf = count_codes_by_type(data)
            code_counts["mixed"] += nm
            code_counts["all_pass"] += nap
            code_counts["all_fail"] += naf
            continue
        if problem_class == "all_fail":
            skipped_all_fail.append(task_id)
            nm, nap, naf = count_codes_by_type(data)
            code_counts["mixed"] += nm
            code_counts["all_pass"] += nap
            code_counts["all_fail"] += naf
            continue

        nm, nap, naf = count_codes_by_type(data)
        code_counts["mixed"] += nm
        code_counts["all_pass"] += nap
        code_counts["all_fail"] += naf

        apfd = apfd_fn(data)
        if apfd is not None:
            apfd_values.append((task_id, apfd))

    return apfd_values, skipped_all_pass, skipped_compile, skipped_all_fail, code_counts


def format_section(
    label: str,
    apfd_values: List[Tuple[str, float]],
    skipped_all_pass: List[str],
    skipped_compile: List[str],
    skipped_all_fail: List[str],
    code_counts: Dict[str, int],
    verbose: bool,
) -> List[str]:
    lines: List[str] = [f"=== {label} ==="]
    if verbose:
        lines.append("Per-problem APFD:")
        for task_id, apfd in apfd_values:
            lines.append(f"  {task_id}: {apfd:.4f}")
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

    if not apfd_values:
        lines.append("No mixed problems found. No benchmark APFD to report.")
    else:
        benchmark_apfd = sum(v for _, v in apfd_values) / len(apfd_values)
        total_codes = sum(code_counts.values())
        lines += [
            f"Benchmark APFD              : {benchmark_apfd:.4f}",
            f"Problems included (mixed)   : {len(apfd_values)}",
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
        description="Compute APFD from TCP prioritization results",
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
        "--fault-type",
        choices=list(APFD_FUNCS.keys()),
        default="apfd",
        help="APFD calculation mode",
    )
    parser.add_argument(
        "--eval-json",
        default=LCB_EVAL_JSON,
        help="[lcb only] Path to filtered_output_eval_all.json with difficulty labels",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-problem APFD values",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save report as .json (default: src/results/<dataset>/metrics/<rel>/<fault_type>.json)",
    )
    args = parser.parse_args()

    pri_dir = Path(args.prioritization_dir or DEFAULT_PRIORITIZATION_DIR[args.dataset])
    result_files = sorted(pri_dir.glob("*.json"))
    if not result_files:
        raise FileNotFoundError(f"No JSON files found in {pri_dir}")

    apfd_fn = APFD_FUNCS[args.fault_type]
    is_lcb = args.dataset == "lcb"
    check_compile = is_lcb

    # overall
    overall_values, overall_all_pass, overall_compile, overall_all_fail, overall_codes = \
        compute_apfd_for_subset(result_files, apfd_fn, check_compile_failure=check_compile)

    all_lines: List[str] = [f"Dataset: {args.dataset}", f"Fault type: {args.fault_type}", ""]
    all_lines += format_section(
        "overall", overall_values, overall_all_pass, overall_compile,
        overall_all_fail, overall_codes, args.verbose,
    )

    result: dict = {
        "dataset": args.dataset,
        "fault_type": args.fault_type,
        "overall": section_dict(overall_values, overall_all_pass, overall_compile,
                                overall_all_fail, overall_codes),
    }

    # per-difficulty (lcb only)
    if is_lcb:
        difficulty_map = load_lcb_difficulty_map(args.eval_json)
        per_diff: Dict[str, dict] = {}
        for diff in ("easy", "medium", "hard"):
            values, all_pass, skipped_compile, all_fail, codes = compute_apfd_for_subset(
                result_files, apfd_fn,
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
        output_path = metrics_base / rel / f"{args.fault_type}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
