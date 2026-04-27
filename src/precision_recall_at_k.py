"""
Compute Precision@k and Recall@k for TCP (Test Case Prioritization).

Directory layout:
  Flat methods   : src/results/{dataset}/prioritization/{method}/BigCodeBench_{id}.json
  VP methods     : src/results/{dataset}/prioritization/{vp_method}/{prompt_style}/{model}/BigCodeBench_{id}.json

tc_order format:
  Flat : ["tc_name", ...]
  VP   : [{"name": "tc_name", ...}, ...]

Metrics (macro-averaged across a common intersection of failing (task, code) pairs):
  Recall@k    = |{j | p_j <= k}| / f
  Precision@k = |{t <= k | test t fails}| / k

  k is a ratio (0.0–1.0] of total TC count.
  k = ceil(n * ratio), where n = total TCs in the pair.
  (task, code) pairs with f=0 are excluded from computation.
  For fair comparison, every method is averaged over the same set of
  failing pairs: the intersection of pair keys present in all methods.
"""

import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent

RESULTS_BASE = SCRIPT_DIR / "results"
DATASETS     = ["bcb", "lcb"]

# Ratio-based k values (proportion of total TC count per pair)
K_RATIOS = [0.25, 0.50, 0.75, 1.00]

# Methods whose files sit directly under prioritization/{method}/
FLAT_METHODS = {"ncd_lzma", "original", "random", "input_len_desc"}

# Display labels for flat methods (method name used when not listed here)
FLAT_METHOD_LABELS: dict[str, str] = {"input_len_desc": "LEN"}

# Methods whose files sit under prioritization/{vp_method}/{prompt_style}/{model}/
# Keep this aligned with the directories that actually exist in the repo.
VP_METHODS = {"vp_original", "vp_length"}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_tc_order(tc_order: list) -> list[str]:
    """Normalise tc_order to a list of TC names regardless of format."""
    if not tc_order:
        return []
    if isinstance(tc_order[0], str):
        return tc_order
    return [item["name"] for item in tc_order]  # VP format: list of dicts


# ──────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────

def compute_ratio_metrics_for_task_code(
    tc_order: list[str],
    tc_results: list[dict],
    ratios: list[float],
) -> dict:
    """
    Compute Precision@r% and Recall@r% for one (task, code) pair.

    k = ceil(n * ratio) where n = total TC count.
    Returns {ratio: {precision, recall, k_used}} where values are None when f == 0.
    """
    status_map = {r["name"]: r["status"].lower() for r in tc_results}
    n = len(tc_order)
    f = sum(1 for s in status_map.values() if s == "fail")

    results = {}
    for ratio in ratios:
        k = min(math.ceil(n * ratio), n)
        top_k_fails = sum(1 for tc in tc_order[:k] if status_map.get(tc) == "fail")

        precision = (top_k_fails / k) if f > 0 and k > 0 else None
        recall    = (top_k_fails / f) if f > 0 else None

        results[ratio] = {"precision": precision, "recall": recall}

    return results


def collect_task_file_pairs(data: dict) -> dict[tuple[str, int], dict]:
    """
    Process one task JSON and return pair-level ratio metrics.

    Key = (task_id, code_index), value = {ratio: {precision, recall}}.
    Only failing pairs (f > 0) are retained.
    """
    tc_order = parse_tc_order(data.get("tc_order", []))
    task_id = str(data.get("task_id"))
    pair_metrics = {}

    for code_result in data.get("codes_results", []):
        tc_results = code_result.get("tc_results", [])
        if not tc_results:
            continue

        ratio_metrics = compute_ratio_metrics_for_task_code(tc_order, tc_results, K_RATIOS)
        # f == 0 pairs have precision/recall = None for every ratio.
        if all(m["precision"] is None for m in ratio_metrics.values()):
            continue

        code_index = int(code_result.get("code_index", 0))
        pair_metrics[(task_id, code_index)] = ratio_metrics

    return pair_metrics


def average_pairs(pair_metrics: dict[tuple[str, int], dict], common_keys: set[tuple[str, int]], keys: list) -> dict:
    averaged = {}
    for ratio in keys:
        precisions = [pair_metrics[key][ratio]["precision"] for key in common_keys]
        recalls = [pair_metrics[key][ratio]["recall"] for key in common_keys]
        averaged[ratio] = {
            "precision": float(np.mean(precisions)) if precisions else 0.0,
            "recall":    float(np.mean(recalls))    if recalls    else 0.0,
            "n_pairs":   len(common_keys),
        }
    return averaged


# ──────────────────────────────────────────────
# Method iterators
# ──────────────────────────────────────────────

def collect_flat_method(prio_root: Path, method: str) -> dict[tuple[str, int], dict]:
    method_dir = prio_root / method
    if not method_dir.exists():
        return {}

    pair_metrics = {}

    for task_file in sorted(method_dir.glob("*.json")):
        pair_metrics.update(collect_task_file_pairs(load_json(task_file)))

    return pair_metrics


def collect_vp_method(prio_root: Path, vp_method: str) -> dict[tuple, dict[tuple[str, int], dict]]:
    """Returns {(vp_method, prompt_style, model): {(task_id, code_index): ratio_metrics}}."""
    vp_dir = prio_root / vp_method
    if not vp_dir.exists():
        return {}

    results = {}
    for prompt_dir in sorted(vp_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        for model_dir in sorted(prompt_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            pair_metrics = {}

            for task_file in sorted(model_dir.glob("*.json")):
                pair_metrics.update(collect_task_file_pairs(load_json(task_file)))

            key = (vp_method, prompt_dir.name, model_dir.name)
            results[key] = pair_metrics

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_dataset(dataset: str) -> dict:
    prio_root = RESULTS_BASE / dataset / "prioritization"
    if not prio_root.exists():
        print(f"[ERROR] Directory not found: {prio_root}")
        return {}

    all_pair_metrics: dict[tuple | str, dict[tuple[str, int], dict]] = {}

    for method in sorted(FLAT_METHODS):
        result = collect_flat_method(prio_root, method)
        if result:
            all_pair_metrics[method] = result

    for vp_method in sorted(VP_METHODS):
        all_pair_metrics.update(collect_vp_method(prio_root, vp_method))

    if not all_pair_metrics:
        return {}

    common_keys = set.intersection(*(set(pairs.keys()) for pairs in all_pair_metrics.values()))
    print(f"  common failing-pair intersection: {len(common_keys)}")

    all_results: dict[tuple | str, dict] = {}
    for key, pair_metrics in all_pair_metrics.items():
        all_results[key] = average_pairs(pair_metrics, common_keys, K_RATIOS)

    return all_results


def label(key) -> str:
    if isinstance(key, str):
        return FLAT_METHOD_LABELS.get(key, key)
    vp, ps, m = key
    return f"{vp}/{ps}/{m}"


def save_csv(dataset: str, all_results: dict) -> None:
    out_dir = RESULTS_BASE / dataset / "precision_recall_@_k"
    out_dir.mkdir(parents=True, exist_ok=True)

    ratio_labels = [f"{int(r*100)}%" for r in K_RATIOS]
    fields = ["method"] + [f for rl in ratio_labels for f in (f"P@{rl}", f"R@{rl}", f"n_fail_pairs@{rl}")]

    with open(out_dir / "ratio_k.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, result in sorted(all_results.items(), key=lambda x: label(x[0])):
            row = {"method": label(key)}
            for r, rl in zip(K_RATIOS, ratio_labels):
                m = result[r]
                row[f"P@{rl}"]            = round(m["precision"], 6)
                row[f"R@{rl}"]            = round(m["recall"], 6)
                row[f"n_fail_pairs@{rl}"] = m["n_pairs"]
            writer.writerow(row)

    print(f"  → {out_dir}/ratio_k.csv")


def print_table(dataset: str, all_results: dict) -> None:
    col_w = max(len(label(k)) for k in all_results) + 2
    ratio_labels = [f"{int(r*100)}%" for r in K_RATIOS]
    header = "".join(f"  P@{rl:<4} R@{rl:<4}" for rl in ratio_labels)

    print(f"\n══ [{dataset}] Ratio-based ══")
    print(f"{'Method':<{col_w}}{header}")
    print("-" * (col_w + len(K_RATIOS) * 16))
    for key, result in sorted(all_results.items(), key=lambda x: label(x[0])):
        row = f"{label(key):<{col_w}}"
        for r in K_RATIOS:
            m = result[r]
            row += f"  {m['precision']:.3f}  {m['recall']:.3f} "
        print(row)


def main():
    for dataset in DATASETS:
        print(f"\n{'='*60}\nDataset: {dataset}\n{'='*60}")
        all_results = run_dataset(dataset)
        if not all_results:
            continue
        print_table(dataset, all_results)
        save_csv(dataset, all_results)


if __name__ == "__main__":
    main()
