#!/usr/bin/env python3
"""
Evaluate LCB code generation results.

Input:  nucleus_samples.jsonl  (one line per problem, with code_list)
Output: eval.json              (overall + per-difficulty pass@1/5/10)
        eval_all.json          (per-problem graded_list + metadata with timing)

Usage:
  python3 evaluate.py --samples results/qwen3-coder-30B-A3B-instruct/nucleus_samples.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ACTUAL_EXEC_DIR = Path(__file__).resolve().parent
LCB_DIR = _ACTUAL_EXEC_DIR / "LiveCodeBench"
sys.path.insert(0, str(LCB_DIR))

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from lcb_runner.evaluation.pass_k_utils import extract_instance_results, compute_metrics_from_results


def load_lcb_benchmark(our_qids: set, cache_dir: str = None) -> list:
    """Load LCB benchmark problems matching our_qids from HF local cache."""
    if cache_dir:
        snapshot_dir = Path(cache_dir)
    else:
        hub_dir = Path.home() / ".cache/huggingface/hub/datasets--livecodebench--code_generation_lite/snapshots"
        snapshots = sorted(hub_dir.iterdir()) if hub_dir.exists() else []
        if not snapshots:
            raise RuntimeError("LCB dataset not found in HF cache. Run setup.sh first.")
        snapshot_dir = snapshots[-1]

    jsonl_files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
    raw_problems = []
    for fname in jsonl_files:
        fpath = snapshot_dir / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    p = json.loads(line)
                    if p["question_id"] in our_qids:
                        raw_problems.append(p)

    return [CodeGenerationProblem(**p) for p in raw_problems]


def normalize_metadata(raw_meta_str: str) -> dict:
    """Convert testing_util raw metadata → normalized schema with timing."""
    try:
        m = json.loads(raw_meta_str)
    except Exception:
        return {"status": "fail", "details": {"unknown": raw_meta_str}, "runtime": 0.0, "time_breakdown": {}}

    time_breakdown = m.get("time_breakdown", {})
    runtime = m.get("execution time", sum(time_breakdown.values()))
    details = m.get("details", {})

    if "error_code" in m:
        error_detail = m.get("error", m.get("error_message", "Unknown Error"))
        details = {"compile": error_detail}

    return {
        "status": "pass" if not details else "fail",
        "details": details,
        "runtime": runtime,
        "time_breakdown": time_breakdown,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=str,
        required=True,
        help="Path to nucleus_samples.jsonl produced by code_generate.py",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Number of parallel evaluation processes (default: CPU count / 2)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to HF dataset snapshot directory (auto-detected if omitted).",
    )
    args = parser.parse_args()

    if args.parallel is None:
        import os
        args.parallel = max(1, (os.cpu_count() or 2) // 2)

    samples_path = Path(args.samples)
    results_dir = samples_path.parent

    # ── Load generated samples ────────────────────────────────────────────────
    print(f"Loading {samples_path}...")
    data = []
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    our_qids = {item["question_id"] for item in data}
    our_codes = {item["question_id"]: item["code_list"] for item in data}
    n_codes = len(data[0]["code_list"]) if data else 0
    print(f"  {len(data)} problems, {n_codes} codes each")

    # ── Load LCB benchmark ────────────────────────────────────────────────────
    print("Loading LCB benchmark from local cache...")
    benchmark_all = load_lcb_benchmark(our_qids, args.cache_dir)
    benchmark = sorted(
        [p for p in benchmark_all if p.question_id in our_qids],
        key=lambda x: str(x.question_id),
    )
    print(f"  Matched: {len(benchmark)} / {len(data)} problems")

    missing = our_qids - {p.question_id for p in benchmark}
    if missing:
        print(f"  WARNING: {len(missing)} question_ids not found in benchmark: {sorted(missing)[:10]}")

    # ── Run evaluation ────────────────────────────────────────────────────────
    samples_list = [p.get_evaluation_sample() for p in benchmark]
    generations_list = [our_codes[p.question_id] for p in benchmark]

    print(f"\nRunning codegen_metrics on {len(samples_list)} problems x {n_codes} codes...")
    metrics, results, final_metadata = codegen_metrics(
        samples_list,
        generations_list,
        k_list=[1, 5, 10],
        num_process_evaluate=args.parallel,
        timeout=6,
    )

    # ── Per-difficulty metrics ────────────────────────────────────────────────
    diff_results = defaultdict(dict)
    for i, prob in enumerate(benchmark):
        diff = prob.difficulty.value
        diff_results[diff][i] = results[i]

    diff_metrics = {}
    for diff, res in diff_results.items():
        dm = compute_metrics_from_results(res, k_list=[1, 5, 10])
        diff_metrics[diff] = {k: v for k, v in dm.items() if k != "detail"}
        print(f"  [{diff}] {len(res)} problems")

    # ── Save eval.json ────────────────────────────────────────────────────────
    output_metrics = {
        "overall": {k: v for k, v in metrics.items() if k != "detail"},
        **{f"difficulty_{d}": m for d, m in diff_metrics.items()},
    }
    eval_path = results_dir / "eval.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(output_metrics, f, indent=4)
    print(f"\n=== Metrics ===")
    print(json.dumps(output_metrics, indent=2))
    print(f"Saved → {eval_path}")

    # ── Save eval_all.json ────────────────────────────────────────────────────
    graded = extract_instance_results(results)

    save_eval_results = [
        instance.insert_output_evaluation(
            generations_list[i],
            generations_list[i],
            graded[i],
            metadata=[normalize_metadata(m) for m in final_metadata[i]],
        )
        for i, instance in enumerate(benchmark)
    ]

    eval_all_path = results_dir / "eval_all.json"
    with open(eval_all_path, "w", encoding="utf-8") as f:
        json.dump(save_eval_results, f, indent=4)
    print(f"Saved → {eval_all_path}")


if __name__ == "__main__":
    main()
