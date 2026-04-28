"""
Unified per-TC prediction accuracy calculator.
Supports BigCodeBench-Hard (--dataset bcb) and LiveCodeBench (--dataset lcb).

Exclusions (not scored):
  - no_gt:     TC not in time_breakdown (never executed)
  - null_pred: TC where LLM returned NULL

Outputs:
  <results-dir>/accuracy/<method>/<model>/accuracy_raw.jsonl
  <results-dir>/accuracy/accuracy.csv
"""
import csv
import json
import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))


# ── Dataset-specific GT loaders ────────────────────────────────────────────────

def load_bcb_expected_tcs():
    """Returns {task_id: [tc_name]} from the BigCodeBench-Hard dataset."""
    from BigCodeBench_Hard.verdict_prediction.utils import load_bigcodebench_hard, split_test_cases
    dataset = load_bigcodebench_hard()
    expected = {}
    for item in dataset:
        task_id = item["task_id"]
        split = split_test_cases(item["test"])
        tcs = sorted({t[0] for t in split
                      if t[0] not in ("error_parsing", "no_class_found", "no_test_methods")})
        expected[task_id] = tcs
    return expected


def load_bcb_gt(truth_file):
    """Returns {task_id: {tc_name: 'PASS'|'FAIL'}} from nucleus_eval_all.json."""
    with open(truth_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    gt_map = {}
    for item in raw:
        task_id = str(item.get("question_id") or item.get("task_id"))
        meta0 = item["metadata"][0] if item.get("metadata") else None
        if meta0 is None:
            gt_map[task_id] = {}
            continue
        tb = meta0.get("time_breakdown", {})
        det = meta0.get("details", {})
        gt_map[task_id] = {tc: "FAIL" if tc in det else "PASS" for tc in tb}
    return gt_map


def load_lcb_gt(truth_file):
    """Returns (gt_map, difficulty_map) from filtered_output_eval_all.json.
    gt_map: {qid: {tc_name: 'PASS'|'FAIL'}}  (empty dict = compile error)
    """
    with open(truth_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    gt_map = {}
    difficulty_map = {}
    for item in raw:
        qid = str(item["question_id"])
        difficulty_map[qid] = item.get("difficulty", "unknown").lower()
        meta_list = item.get("metadata", [])
        meta0 = meta_list[0] if meta_list else {}
        if not meta0:
            gt_map[qid] = {}
            continue
        tb = meta0.get("time_breakdown", {})
        det = meta0.get("details", {})
        gt_map[qid] = {tc: "FAIL" if tc in det else "PASS" for tc in tb}
    return gt_map, difficulty_map


# ── Shared utilities ───────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_csv(path: Path, header: list, row: list):
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def pct(c, t):
    return round(c / t * 100, 2) if t else 0.0


def pred_overall_from(included_preds):
    if any(p == "FAIL" for p in included_preds):
        return "FAIL"
    return "PASS" if included_preds else "NULL"


# ── Unified scoring loop ───────────────────────────────────────────────────────

def score(task_ids, expected_tcs_map, gt_map, pred_map, difficulty_map):
    """
    Core scoring loop shared by both datasets.

    task_ids:         ordered list of IDs to score
    expected_tcs_map: {task_id: [tc_name]} for BCB; None for LCB (union of pred/GT keys)
    gt_map:           {task_id: {tc_name: 'PASS'|'FAIL'}}
    pred_map:         {task_id: item} (item has 'pass_fail_list')
    difficulty_map:   {task_id: str} or {} (empty = no difficulty tracking)
    """
    DIFFICULTIES = sorted({v for v in difficulty_map.values()
                           if v not in ("unknown", "")}) if difficulty_map else []
    stats = {d: {"total": 0, "correct": 0} for d in DIFFICULTIES + ["overall"]}
    no_gt_total = null_pred_total = 0
    accuracy_raw = []

    for task_id in task_ids:
        gt_tcs = gt_map.get(task_id, {})
        pred_item = pred_map.get(task_id)
        pred_pf = pred_item.get("pass_fail_list", {}) if pred_item else {}
        diff = difficulty_map.get(task_id)

        if expected_tcs_map is not None:
            expected_tcs = expected_tcs_map.get(task_id, [])
        else:
            # LCB: expected TCs = union of pred and GT keys, sorted numerically
            expected_tcs = sorted(
                set(pred_pf.keys()) | set(gt_tcs.keys()),
                key=lambda x: int(x.split("_")[-1]),
            )

        gt_overall = "PASS" if (gt_tcs and all(v == "PASS" for v in gt_tcs.values())) else "FAIL"

        tc_results = {}
        included_preds = []

        for tc in expected_tcs:
            if tc not in gt_tcs:
                no_gt_total += 1
                raw_pred = pred_pf.get(tc)
                pred_status = None if (not raw_pred or raw_pred == "NULL") else raw_pred
                tc_results[tc] = {"status": None, "pred": pred_status, "correct": None}
                continue

            gt_status = gt_tcs[tc]
            raw_pred = pred_pf.get(tc, "NULL")
            pred_status = None if (not raw_pred or raw_pred == "NULL") else raw_pred

            if pred_status is None:
                null_pred_total += 1
                tc_results[tc] = {"status": gt_status, "pred": None, "correct": None}
                continue

            is_correct = pred_status == gt_status
            stats["overall"]["total"] += 1
            stats["overall"]["correct"] += int(is_correct)
            if diff and diff in stats:
                stats[diff]["total"] += 1
                stats[diff]["correct"] += int(is_correct)

            included_preds.append(pred_status)
            tc_results[tc] = {"status": gt_status, "pred": pred_status, "correct": is_correct}

        pred_overall = pred_overall_from(included_preds)
        overall_correct = (pred_overall == gt_overall) if pred_overall != "NULL" else None

        record = {
            "task_id": task_id,
            "overall_correct": overall_correct,
            "gt_overall": gt_overall,
            "pred_overall": pred_overall,
            "test_cases": tc_results,
        }
        if diff is not None:
            record["difficulty"] = diff
        accuracy_raw.append(record)

    return accuracy_raw, stats, no_gt_total, null_pred_total


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["bcb", "lcb"],
                        help="bcb = BigCodeBench-Hard, lcb = LiveCodeBench")
    parser.add_argument("--pred-file",    required=True, help="Path to test.jsonl")
    parser.add_argument("--truth-file",   required=True, help="Path to ground-truth JSON")
    parser.add_argument("--method",       required=True,
                        choices=["direct_verdict", "verdict_with_analysis", "verdict_with_diagnosis"])
    parser.add_argument("--model",        required=True)
    parser.add_argument("--results-dir",  required=True,
                        help="Base results dir (accuracy/ and accuracy.csv written here)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    raw_out_dir = results_dir / "accuracy" / args.method / args.model
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    accuracy_csv = results_dir / "accuracy" / "accuracy.csv"

    # ── Load predictions ───────────────────────────────────────────────────────
    id_field = "id" if args.dataset == "bcb" else "question_id"
    pred_map = {str(item[id_field]): item for item in load_jsonl(args.pred_file)}

    # ── Load GT + metadata ─────────────────────────────────────────────────────
    if args.dataset == "bcb":
        expected_tcs_map = load_bcb_expected_tcs()
        gt_map = load_bcb_gt(args.truth_file)
        task_ids = list(expected_tcs_map.keys())
        difficulty_map = {}
    else:
        gt_map, difficulty_map = load_lcb_gt(args.truth_file)
        task_ids = list(gt_map.keys())
        expected_tcs_map = None  # computed on-the-fly from union

    # ── Score ──────────────────────────────────────────────────────────────────
    accuracy_raw, stats, no_gt_total, null_pred_total = score(
        task_ids, expected_tcs_map, gt_map, pred_map, difficulty_map
    )

    # ── Write accuracy_raw.jsonl ───────────────────────────────────────────────
    raw_path = raw_out_dir / "accuracy_raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for rec in accuracy_raw:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Write accuracy.csv + print summary ────────────────────────────────────
    if args.dataset == "bcb":
        scored  = stats["overall"]["total"]
        correct = stats["overall"]["correct"]
        header = ["method", "model", "no_gt_tcs", "null_pred_tcs",
                  "scored_tcs", "correct_tcs", "accuracy"]
        row = [args.method, args.model, no_gt_total, null_pred_total,
               scored, correct, pct(correct, scored)]
        print(f"[{args.method}/{args.model}]  "
              f"accuracy: {correct}/{scored} = {pct(correct, scored):.2f}%  "
              f"(excl. no_gt={no_gt_total}, null_pred={null_pred_total})")
    else:
        DIFFICULTIES = ["easy", "medium", "hard"]
        header = ["method", "model", "no_gt_tcs", "null_pred_tcs",
                  "easy_total",   "easy_correct",   "easy_acc",
                  "medium_total", "medium_correct", "medium_acc",
                  "hard_total",   "hard_correct",   "hard_acc",
                  "overall_total", "overall_correct", "overall_acc"]
        row = [args.method, args.model, no_gt_total, null_pred_total]
        for d in DIFFICULTIES + ["overall"]:
            s = stats[d]
            row += [s["total"], s["correct"], pct(s["correct"], s["total"])]
        ov = stats["overall"]
        print(f"[{args.method}/{args.model}]  "
              f"Easy={pct(stats['easy']['correct'], stats['easy']['total']):.2f}%  "
              f"Medium={pct(stats['medium']['correct'], stats['medium']['total']):.2f}%  "
              f"Hard={pct(stats['hard']['correct'], stats['hard']['total']):.2f}%  "
              f"Overall={pct(ov['correct'], ov['total']):.2f}%"
              f" ({ov['correct']}/{ov['total']})  "
              f"(excl. no_gt={no_gt_total}, null_pred={null_pred_total})")

    append_csv(accuracy_csv, header, row)
    print(f"  accuracy_raw -> {raw_path}")


if __name__ == "__main__":
    main()
