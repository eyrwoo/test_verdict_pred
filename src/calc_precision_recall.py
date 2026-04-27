"""
Unified Precision / Recall / F1 calculator for LLM verdict predictions.
Works for both BigCodeBench-Hard and LiveCodeBench.

Outputs metrics for BOTH positive-class conventions:

  Positive class = FAIL  (defect detection)
    TP: pred=FAIL, GT=FAIL
    FP: pred=FAIL, GT=PASS
    FN: pred=PASS, GT=FAIL
    TN: pred=PASS, GT=PASS

  Positive class = PASS  (correctness detection)
    TP: pred=PASS, GT=PASS
    FP: pred=PASS, GT=FAIL
    FN: pred=FAIL, GT=PASS
    TN: pred=FAIL, GT=FAIL

Input:  <results-dir>/accuracy/<method>/<model>/accuracy_raw.jsonl
        (auto-discovered from results-dir)
Output: <results-dir>/precision_recall/precision_recall.csv

Global intersection: only (task_id, tc_key) pairs where
  - GT status is not None (no compile error)
  - pred is not None in ALL (method, model) combinations

All (method, model) combinations share the same denominator.

If accuracy_raw.jsonl contains a "difficulty" field, outputs 4 blocks per run:
  {method}/{model} [easy], [medium], [hard], [overall]
Otherwise outputs 1 block:
  {method}/{model}

Block format (3 rows + blank):
  {label}          GT PASS  GT FAIL  Precision  {p}
  Predicted PASS   {tn}     {fn}     Recall     {r}
  Predicted FAIL   {fp}     {tp}     F1         {f1}
  (blank)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


def prf(tp: int, fp: int, fn: int, tn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall    = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return tp, fp, fn, tn, precision, recall, f1


def fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "N/A"


def frac(count: int, total: int) -> str:
    ratio = count / total if total > 0 else 0.0
    return f"{ratio:.2f} ({count}/{total})"


def append_block(path: Path, label: str,
                 tp: int, fp: int, fn: int, tn: int,
                 precision, recall, f1) -> None:
    total = tp + fp + fn + tn
    rows = [
        [label,     "Pred. PASS",      "Pred. FAIL",      "Precision", fmt(precision)],
        ["GT PASS",  frac(tn, total),   frac(fp, total),   "Recall",    fmt(recall)],
        ["GT FAIL",  frac(fn, total),   frac(tp, total),   "F1",        fmt(f1)],
        [],
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results-dir", required=True,
                        help="Base results dir; auto-discovers accuracy/<method>/<model>/accuracy_raw.jsonl")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    accuracy_dir = results_dir / "accuracy"

    # ── Discover all (method, model, path) triples ─────────────────────────────
    runs: list[tuple[str, str, Path]] = []
    for jsonl_path in sorted(accuracy_dir.glob("*/*/accuracy_raw.jsonl")):
        method = jsonl_path.parent.parent.name
        model  = jsonl_path.parent.name
        runs.append((method, model, jsonl_path))

    if not runs:
        raise FileNotFoundError(f"No accuracy_raw.jsonl found under {accuracy_dir}")

    print(f"Found {len(runs)} run(s): {[(m, mo) for m, mo, _ in runs]}")

    # ── Load all data ──────────────────────────────────────────────────────────
    # all_data[(method, model)] = list of items
    all_data: dict[tuple[str, str], list[dict]] = {}
    for method, model, path in runs:
        all_data[(method, model)] = load_jsonl(path)

    # ── Build global valid intersection ────────────────────────────────────────
    # valid key = (task_id, tc_key) where GT is not None AND pred is not None
    # for every (method, model) combination
    global_valid: Optional[set] = None

    for (method, model), data in all_data.items():
        valid_here: set[tuple[str, str]] = set()
        for item in data:
            task_id = str(item["task_id"])
            for tc_key, tc in item["test_cases"].items():
                if tc["status"] is not None and tc["pred"] is not None:
                    valid_here.add((task_id, tc_key))
        if global_valid is None:
            global_valid = valid_here
        else:
            global_valid &= valid_here

    if global_valid is None:
        global_valid = set()

    print(f"Global valid intersection: {len(global_valid)} test cases")

    # ── Compute metrics per (method, model) ────────────────────────────────────
    has_difficulty = any(
        any("difficulty" in item for item in data)
        for data in all_data.values()
    )
    DIFFICULTIES = ["easy", "medium", "hard", "overall"] if has_difficulty else ["overall"]

    pr_csv = results_dir / "precision_recall" / "precision_recall.csv"
    pr_csv.parent.mkdir(parents=True, exist_ok=True)

    for method, model, _ in runs:
        data = all_data[(method, model)]
        counts = {d: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for d in DIFFICULTIES}

        for item in data:
            task_id = str(item["task_id"])
            diff = item.get("difficulty", "overall")
            for tc_key, tc in item["test_cases"].items():
                if (task_id, tc_key) not in global_valid:
                    continue
                gt   = tc["status"]
                pred = tc["pred"]
                if   pred == "FAIL" and gt == "FAIL": key = "tp"
                elif pred == "FAIL" and gt == "PASS": key = "fp"
                elif pred == "PASS" and gt == "FAIL": key = "fn"
                else:                                 key = "tn"
                counts["overall"][key] += 1
                if has_difficulty and diff in counts and diff != "overall":
                    counts[diff][key] += 1

        base_label = f"{method}/{model}"
        print(f"\n[{base_label}]")
        for d in DIFFICULTIES:
            c = counts[d]
            diff_tag = f" [{d}]" if has_difficulty else ""

            # FAIL as positive class
            tp, fp, fn, tn, precision, recall, f1 = prf(c["tp"], c["fp"], c["fn"], c["tn"])
            label_fail = f"{base_label}{diff_tag} [FAIL+]"
            append_block(pr_csv, label_fail, tp, fp, fn, tn, precision, recall, f1)
            print(f"  [{d}][FAIL+]  Precision={fmt(precision)}  Recall={fmt(recall)}  F1={fmt(f1)}  "
                  f"(TP={tp} FP={fp} FN={fn} TN={tn}  total={tp+fp+fn+tn})")

            # PASS as positive class (swap tp↔tn, fp↔fn)
            tp2, fp2, fn2, tn2, precision2, recall2, f12 = prf(c["tn"], c["fn"], c["fp"], c["tp"])
            label_pass = f"{base_label}{diff_tag} [PASS+]"
            append_block(pr_csv, label_pass, tp2, fp2, fn2, tn2, precision2, recall2, f12)
            print(f"  [{d}][PASS+]  Precision={fmt(precision2)}  Recall={fmt(recall2)}  F1={fmt(f12)}  "
                  f"(TP={tp2} FP={fp2} FN={fn2} TN={tn2}  total={tp2+fp2+fn2+tn2})")

    print(f"\n-> {pr_csv}")


if __name__ == "__main__":
    main()
