#!/usr/bin/env python3
"""
Unified Test Case Prioritization (TCP).
Supports BigCodeBench-Hard (--dataset bcb) and LiveCodeBench (--dataset lcb).

Methods (--method):
  original             original TC order (no shuffle)
  random               random order (reproducible seed)
  ncd                  NCD greedy dissimilarity (default: lzma compressor)
  input_len_desc       longer TC source text first (descending text length)
  vp_original          FAIL-first; ties → original order; null TCs excluded
  vp_random            FAIL-first; ties → random; null TCs excluded
  vp_length            FAIL-first; ties → longer TC source text first; null TCs excluded
  vp_token_input_desc  FAIL-first; ties → more input tokens first; null TCs excluded
  vp_token_output_desc FAIL-first; ties → more output tokens first; null TCs excluded
  vp_token_output_asc  FAIL-first; ties → fewer output tokens first; null TCs excluded

Null filtering (VP methods only):
  TCs where the LLM returned NULL are excluded from the TC order and from
  codes_results entirely.  e.g. 5 TCs but tc_2 is null → 4 TCs ordered.

Token-based methods prerequisite:
  vp_token_* methods require --token-report-file pointing to a tokens_count.json
  produced by calc_tokens.py (or calc_tokens.sh).

Output:
  <results-dir>/prioritization/original/                       {task_id}.json
  <results-dir>/prioritization/random/                         {task_id}.json
  <results-dir>/prioritization/ncd_{compressor}/               {task_id}.json
  <results-dir>/prioritization/input_len_desc/                 {task_id}.json
  <results-dir>/prioritization/{vp_method}/{vp_type}/{model}/  {task_id}.json
"""
from __future__ import annotations

import argparse
import bz2
import json
import logging
import lzma
import random
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

# ── Constants ──────────────────────────────────────────────────────────────────

SIMPLE_METHODS: Set[str] = {"original", "random", "ncd", "input_len_desc"}
VP_METHODS: Set[str] = {"vp_original", "vp_random", "vp_length", "vp_token_input_desc", "vp_token_output_desc", "vp_token_output_asc"}
LENGTH_METHODS: Set[str] = {"vp_length"}
TOKEN_METHODS: Set[str] = {"vp_token_input_desc", "vp_token_output_desc", "vp_token_output_asc"}
ALL_METHODS = sorted(SIMPLE_METHODS | VP_METHODS)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Dataset-specific defaults ──────────────────────────────────────────────────

def _dataset_defaults(dataset: str) -> Dict[str, str]:
    if dataset == "bcb":
        return {
            "truth_file": str(
                SRC_DIR / "BigCodeBench_Hard/actual_exec/results/2025"
                          "/qwen3-coder-30B-A3B-instruct/nucleus_eval_all.json"
            ),
            "results_dir": str(SRC_DIR / "results/bcb"),
            "tc_source_cache": "",
        }
    else:  # lcb
        return {
            "truth_file": str(
                SRC_DIR / "LiveCodeBench/actual_exec/results/filtered_output_eval_all.json"
            ),
            "results_dir": str(SRC_DIR / "results/lcb"),
            "tc_source_cache": str(SRC_DIR / "LiveCodeBench/results/fetch_tc_source/tc_source_cache.json"),
        }


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _sanitize_id(task_id: str) -> str:
    """'BigCodeBench/13' → 'BigCodeBench_13'  (LCB ids unchanged)."""
    return task_id.replace("/", "_")


def _get_tc_names(metadata: List[Dict]) -> List[str]:
    """Union of TC names from time_breakdown across all code samples (insertion order)."""
    seen: Dict[str, bool] = {}
    for m in metadata:
        for name in m.get("time_breakdown", {}):
            seen[name] = True
    return list(seen.keys())


def _tc_actual_result(tc_name: str, meta: Dict, dataset: str) -> Dict[str, Any]:
    """Actual execution result for one TC from harness metadata."""
    time_breakdown = meta.get("time_breakdown", {})
    details = meta.get("details", {})
    fallback_key = "ALL" if dataset == "bcb" else "compile"

    if not time_breakdown or tc_name not in time_breakdown:
        return {
            "status": "fail",
            "time": 0.0,
            "details": details.get(fallback_key, "TC did not execute"),
        }

    result: Dict[str, Any] = {"time": round(time_breakdown[tc_name], 6)}
    if tc_name in details:
        result["status"] = "fail"
        result["details"] = details[tc_name]
    else:
        result["status"] = "pass"
    return result


def _build_codes_results(
    tc_order: List[str],
    metadata: List[Dict],
    dataset: str,
) -> List[Dict[str, Any]]:
    codes_results = []
    for code_idx, meta in enumerate(metadata):
        tc_results = []
        for tc_name in tc_order:
            entry = {"name": tc_name}
            entry.update(_tc_actual_result(tc_name, meta, dataset))
            tc_results.append(entry)
        codes_results.append({
            "code_index": code_idx,
            "overall_status": meta.get("status", "fail"),
            "runtime": meta.get("runtime", 0.0),
            "tc_results": tc_results,
        })
    return codes_results


def _compile_failure_result(task_id: str, method: str, metadata: List[Dict]) -> Dict[str, Any]:
    """Placeholder result for LCB problems where all samples failed to compile."""
    return {
        "task_id": task_id,
        "method": method,
        "compile_failure": True,
        "tc_order": [],
        "num_test_cases": 0,
        "num_codes": len(metadata),
        "codes_results": [],
    }


# ── NCD helpers ────────────────────────────────────────────────────────────────

def _compress_len(data: bytes, compressor: str) -> int:
    if compressor == "zlib":
        return len(zlib.compress(data, level=9))
    if compressor == "bz2":
        return len(bz2.compress(data))
    if compressor == "lzma":
        return len(lzma.compress(data))
    raise ValueError(f"Unknown compressor: {compressor!r}")


def _ncd(x: str, y: str, compressor: str) -> float:
    bx, by = x.encode(), y.encode()
    cx = _compress_len(bx, compressor)
    cy = _compress_len(by, compressor)
    cxy = _compress_len(bx + by, compressor)
    return (cxy - min(cx, cy)) / max(cx, cy)


def _build_sim_ncd(tc_names: List[str], source_map: Dict[str, str], compressor: str) -> np.ndarray:
    n = len(tc_names)
    texts = [source_map.get(tc, tc) for tc in tc_names]
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = _ncd(texts[i], texts[j], compressor)
            dist[i, j] = d
            dist[j, i] = d
    return 1.0 - dist


def _greedy_dissimilarity_order(sim: np.ndarray) -> List[int]:
    """Greedy max-dissimilarity ordering given an NxN similarity matrix."""
    n = sim.shape[0]
    if n == 1:
        return [0]
    sim_no_diag = sim.copy()
    np.fill_diagonal(sim_no_diag, -np.inf)
    first = int(np.argmin(sim_no_diag.max(axis=1)))
    selected = [first]
    remaining = list(range(n))
    remaining.remove(first)
    max_sim = sim[:, first].copy()
    while remaining:
        best = min(remaining, key=lambda i: max_sim[i])
        selected.append(best)
        remaining.remove(best)
        np.maximum(max_sim, sim[:, best], out=max_sim)
    return selected


# ── TC source loaders (for NCD) ────────────────────────────────────────────────

def _load_bcb_tc_source(dataset_path: Optional[str]) -> Dict[str, Dict[str, str]]:
    """Returns {task_id: {tc_name: source_code}} for BCB."""
    from BigCodeBench_Hard.actual_exec.utils import load_bigcodebench_hard, split_test_cases
    dataset = load_bigcodebench_hard(dataset_path)
    return {
        record["task_id"]: {name: code for name, code in split_test_cases(record["test"])}
        for record in dataset
    }


def _load_lcb_tc_source(tc_source_cache: str) -> Dict[str, Dict[str, str]]:
    """Returns {question_id: {test_case_1: text, ...}} from cache file."""
    with open(tc_source_cache, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): v for k, v in raw.items()}


# ── VP result loaders ──────────────────────────────────────────────────────────

def _normalize_pred(v: Any) -> Optional[str]:
    """Normalize a raw VP prediction value to 'PASS', 'FAIL', or None (null)."""
    if v is None or not str(v).strip():
        return None
    upper = str(v).upper()
    return None if upper == "NULL" else upper


def load_bcb_vp(vp_file: str) -> Tuple[Dict[str, Dict[str, Optional[str]]], str, str]:
    """
    Load BCB VP results from test.jsonl.

    Returns
    -------
    pred_map  : {task_id: {tc_name: "PASS"|"FAIL"|None}}
    vp_type   : inferred from path (e.g. "direct_verdict")
    model_name: inferred from path (e.g. "claude-haiku-4-5-20251001")
    """
    vp_path = Path(vp_file)
    model_name = vp_path.parent.name
    vp_type = vp_path.parent.parent.name

    pred_map: Dict[str, Dict[str, Optional[str]]] = {}
    with open(vp_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task_id = str(rec["id"])
            pfl = rec.get("pass_fail_list", {})
            pred_map[task_id] = {tc: _normalize_pred(v) for tc, v in pfl.items()}

    return pred_map, vp_type, model_name


def load_lcb_vp(vp_file: str) -> Tuple[Dict[str, Dict[str, Optional[str]]], str, str]:
    """
    Load LCB VP results from test.jsonl.

    Returns
    -------
    pred_map  : {question_id: {test_case_N: "PASS"|"FAIL"|None}}
    vp_type   : inferred from path
    model_name: inferred from path
    """
    vp_path = Path(vp_file)
    model_name = vp_path.parent.name
    vp_type = vp_path.parent.parent.name

    pred_map: Dict[str, Dict[str, Optional[str]]] = {}
    with open(vp_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec["question_id"])
            pfl = rec.get("pass_fail_list", {})
            pred_map[qid] = {tc: _normalize_pred(v) for tc, v in pfl.items()}

    return pred_map, vp_type, model_name


# ── Token report loaders ───────────────────────────────────────────────────────

def load_token_report(report_file: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Load token_report.json: {task_id: {tc_name: {input_tokens, output_tokens}}}."""
    with open(report_file, encoding="utf-8") as f:
        return json.load(f)


# ── VP ordering (null filtering) ───────────────────────────────────────────────

def _vp_order(
    tc_names: List[str],
    pred: Dict[str, Optional[str]],
    method: str,
    seed: int,
    tc_tokens: Optional[Dict[str, Dict[str, int]]] = None,
    source_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Sort tc_names for VP-based methods.

    1. Filter out any TC whose prediction is None (null → excluded entirely).
    2. Among remaining TCs, FAIL-predicted come first; PASS-predicted follow.
    3. Ties within each group broken by method.
    """
    valid = [tc for tc in tc_names if pred.get(tc) is not None]

    def fail_key(tc: str) -> int:
        return 0 if pred.get(tc) == "FAIL" else 1

    if method == "vp_original":
        return sorted(valid, key=fail_key)

    if method == "vp_random":
        rng = random.Random(seed)
        shuffled = valid[:]
        rng.shuffle(shuffled)
        return sorted(shuffled, key=fail_key)

    if method == "vp_length":
        src = source_map or {}
        return sorted(
            valid,
            key=lambda tc: (fail_key(tc), -len(src.get(tc, tc))),
        )

    tokens = tc_tokens or {}

    if method == "vp_token_input_desc":
        return sorted(
            valid,
            key=lambda tc: (fail_key(tc), -tokens.get(tc, {}).get("input_tokens", 0)),
        )

    if method == "vp_token_output_desc":
        return sorted(
            valid,
            key=lambda tc: (fail_key(tc), -tokens.get(tc, {}).get("output_tokens", 0)),
        )

    if method == "vp_token_output_asc":
        return sorted(
            valid,
            key=lambda tc: (fail_key(tc), tokens.get(tc, {}).get("output_tokens", 0)),
        )

    raise ValueError(f"Unknown VP method: {method!r}")


# ── Per-problem processors ─────────────────────────────────────────────────────

def _process_original(problem: Dict, dataset: str) -> Dict[str, Any]:
    task_id = str(problem["question_id"])
    metadata = problem["metadata"]
    tc_names = _get_tc_names(metadata)
    if not tc_names:
        log.warning("%s: no TC names found, skipping", task_id)
        return {} if dataset == "bcb" else _compile_failure_result(task_id, "original", metadata)
    return {
        "task_id": task_id,
        "method": "original",
        "tc_order": tc_names,
        "num_test_cases": len(tc_names),
        "num_codes": len(metadata),
        "codes_results": _build_codes_results(tc_names, metadata, dataset),
    }


def _process_random(problem: Dict, seed: int, dataset: str) -> Dict[str, Any]:
    task_id = str(problem["question_id"])
    metadata = problem["metadata"]
    tc_names = _get_tc_names(metadata)
    if not tc_names:
        log.warning("%s: no TC names found, skipping", task_id)
        return {} if dataset == "bcb" else _compile_failure_result(task_id, "random", metadata)
    rng = random.Random(seed)
    shuffled = tc_names[:]
    rng.shuffle(shuffled)
    return {
        "task_id": task_id,
        "method": "random",
        "seed": seed,
        "tc_order": shuffled,
        "num_test_cases": len(shuffled),
        "num_codes": len(metadata),
        "codes_results": _build_codes_results(shuffled, metadata, dataset),
    }


def _process_ncd(
    problem: Dict,
    source_map: Dict[str, str],
    compressor: str,
    dataset: str,
) -> Dict[str, Any]:
    task_id = str(problem["question_id"])
    metadata = problem["metadata"]
    tc_names = _get_tc_names(metadata)
    if not tc_names:
        log.warning("%s: no TC names found, skipping", task_id)
        return {} if dataset == "bcb" else _compile_failure_result(task_id, f"ncd_{compressor}", metadata)
    if len(tc_names) == 1:
        tc_order = tc_names
    else:
        log.info("  Computing NCD (%s) for %d TCs ...", compressor, len(tc_names))
        sim = _build_sim_ncd(tc_names, source_map, compressor)
        tc_order = [tc_names[i] for i in _greedy_dissimilarity_order(sim)]
    return {
        "task_id": task_id,
        "method": "ncd",
        "compressor": compressor,
        "tc_order": tc_order,
        "num_test_cases": len(tc_order),
        "num_codes": len(metadata),
        "codes_results": _build_codes_results(tc_order, metadata, dataset),
    }


def _process_input_len_desc(
    problem: Dict,
    source_map: Dict[str, str],
    dataset: str,
) -> Dict[str, Any]:
    task_id = str(problem["question_id"])
    metadata = problem["metadata"]
    tc_names = _get_tc_names(metadata)
    if not tc_names:
        log.warning("%s: no TC names found, skipping", task_id)
        return {} if dataset == "bcb" else _compile_failure_result(task_id, "input_len_desc", metadata)
    tc_order = sorted(tc_names, key=lambda tc: len(source_map.get(tc, tc)), reverse=True)
    return {
        "task_id": task_id,
        "method": "input_len_desc",
        "tc_order": tc_order,
        "num_test_cases": len(tc_order),
        "num_codes": len(metadata),
        "codes_results": _build_codes_results(tc_order, metadata, dataset),
    }


def _process_vp(
    problem: Dict,
    pred: Optional[Dict[str, Optional[str]]],
    method: str,
    seed: int,
    dataset: str,
    tc_tokens: Optional[Dict[str, Dict[str, int]]] = None,
    source_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    task_id = str(problem["question_id"])
    metadata = problem["metadata"]
    tc_names = _get_tc_names(metadata)

    if not tc_names:
        log.warning("%s: no TC names found, skipping", task_id)
        return {} if dataset == "bcb" else _compile_failure_result(task_id, method, metadata)

    if pred is None:
        log.warning("%s: no VP record found, falling back to original order (no null filtering)", task_id)
        tc_order = tc_names
        pred_for_annotation: Dict[str, Optional[str]] = {}
    else:
        tc_order = _vp_order(tc_names, pred, method, seed, tc_tokens, source_map)
        pred_for_annotation = pred

    tc_order_annotated: List[Dict[str, Any]] = []
    for tc in tc_order:
        entry: Dict[str, Any] = {
            "name": tc,
            "vp_prediction": pred_for_annotation.get(tc),
        }
        if method in TOKEN_METHODS and tc_tokens:
            tok = tc_tokens.get(tc, {})
            entry["input_tokens"] = tok.get("input_tokens")
            entry["output_tokens"] = tok.get("output_tokens")
        if method in LENGTH_METHODS and source_map:
            entry["source_length"] = len(source_map.get(tc, tc))
        tc_order_annotated.append(entry)

    out: Dict[str, Any] = {
        "task_id": task_id,
        "method": method,
        "tc_order": tc_order_annotated,
        "num_test_cases": len(tc_order),
        "num_codes": len(metadata),
        "codes_results": _build_codes_results(tc_order, metadata, dataset),
    }
    if method == "vp_random":
        out["seed"] = seed
    return out


# ── Run loops ──────────────────────────────────────────────────────────────────

def _run_simple(
    problems: List[Dict],
    method: str,
    output_dir: Path,
    dataset: str,
    seed: int,
    resume: bool,
    task_filter: Optional[Set[str]],
    tc_source_all: Optional[Dict[str, Dict[str, str]]] = None,
    compressor: str = "lzma",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if task_filter:
        problems = [p for p in problems if str(p["question_id"]) in task_filter]
    log.info("%d problems to process  (method=%s)", len(problems), method)

    for idx, problem in enumerate(problems):
        task_id = str(problem["question_id"])
        out_file = output_dir / f"{_sanitize_id(task_id)}.json"

        if resume and out_file.exists():
            log.info("[%d/%d] %s — skip (exists)", idx + 1, len(problems), task_id)
            continue

        log.info("[%d/%d] %s", idx + 1, len(problems), task_id)

        if method == "original":
            result = _process_original(problem, dataset)
        elif method == "random":
            result = _process_random(problem, seed, dataset)
        elif method == "input_len_desc":
            source_map = (tc_source_all or {}).get(task_id, {})
            result = _process_input_len_desc(problem, source_map, dataset)
        else:  # ncd
            source_map = (tc_source_all or {}).get(task_id, {})
            result = _process_ncd(problem, source_map, compressor, dataset)

        if not result:
            continue

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    log.info("Done. Results in: %s", output_dir)


def _run_vp(
    problems: List[Dict],
    pred_map: Dict[str, Dict[str, Optional[str]]],
    method: str,
    vp_type: str,
    model_name: str,
    priori_dir: Path,
    dataset: str,
    seed: int,
    resume: bool,
    task_filter: Optional[Set[str]],
    token_map: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
    tc_source_all: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    out_path = priori_dir / method / vp_type / model_name
    out_path.mkdir(parents=True, exist_ok=True)

    if task_filter:
        problems = [p for p in problems if str(p["question_id"]) in task_filter]
    log.info(
        "%d problems to process  (method=%s, vp_type=%s, model=%s)",
        len(problems), method, vp_type, model_name,
    )

    for idx, problem in enumerate(problems):
        task_id = str(problem["question_id"])
        out_file = out_path / f"{_sanitize_id(task_id)}.json"

        if resume and out_file.exists():
            log.info("[%d/%d] %s — skip (exists)", idx + 1, len(problems), task_id)
            continue

        log.info("[%d/%d] %s", idx + 1, len(problems), task_id)
        pred = pred_map.get(task_id)
        task_tokens = (token_map or {}).get(task_id)
        task_source = (tc_source_all or {}).get(task_id)
        result = _process_vp(problem, pred, method, seed, dataset, task_tokens, task_source)

        if not result:
            continue

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    log.info("Done. Results in: %s", out_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Test Case Prioritization for BigCodeBench-Hard and LiveCodeBench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True, choices=["bcb", "lcb"],
        help="bcb = BigCodeBench-Hard, lcb = LiveCodeBench",
    )
    parser.add_argument(
        "--method", required=True, choices=ALL_METHODS,
        help="Prioritization strategy",
    )
    parser.add_argument(
        "--truth-file", default=None,
        help="Ground-truth eval JSON (auto-derived from --dataset if omitted)",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Base results directory (default: src/results/{dataset})",
    )

    vp_group = parser.add_argument_group("VP methods")
    vp_group.add_argument(
        "--vp-file", default=None,
        help="[vp_*] Path to test.jsonl (required for all vp_* methods)",
    )
    vp_group.add_argument(
        "--token-report-file", default=None, metavar="PATH",
        help="[vp_token_*] Explicit path to token report JSON (overrides auto-derived path)",
    )

    ncd_group = parser.add_argument_group("NCD method")
    ncd_group.add_argument(
        "--compressor", default="lzma", choices=["zlib", "bz2", "lzma"],
        help="[ncd] Compression algorithm for NCD",
    )
    ncd_group.add_argument(
        "--dataset-path", default=None,
        help="[ncd/bcb] Local BigCodeBench-Hard dataset path (downloads from HF if omitted)",
    )
    ncd_group.add_argument(
        "--tc-source-cache", default=None,
        help="[ncd/lcb] Path to tc_source_cache.json (auto-derived if omitted)",
    )

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for random and vp_random methods")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-run even if output file already exists")
    parser.add_argument("--tasks", nargs="+", default=None, metavar="ID",
                        help="Limit to specific task IDs")
    args = parser.parse_args()

    # Resolve defaults from dataset
    defaults = _dataset_defaults(args.dataset)
    truth_file = args.truth_file or defaults["truth_file"]
    results_dir = Path(args.results_dir or defaults["results_dir"])
    priori_dir = results_dir / "prioritization"
    resume = not args.no_resume
    task_filter: Optional[Set[str]] = set(args.tasks) if args.tasks else None

    # ── Validation ─────────────────────────────────────────────────────────────

    if args.method in VP_METHODS and not args.vp_file:
        parser.error(f"--vp-file is required for --method {args.method}")

    if args.method in {"ncd", "input_len_desc", "vp_length"} and args.dataset == "lcb":
        tc_cache = args.tc_source_cache or defaults["tc_source_cache"]
        if not tc_cache or not Path(tc_cache).exists():
            parser.error(
                f"TC source cache not found: {tc_cache}\n"
                "Run: python src/LiveCodeBench/fetch_tc_source.py"
            )

# ── Load eval (ground-truth) data ──────────────────────────────────────────

    log.info("Loading eval file: %s", truth_file)
    with open(truth_file, encoding="utf-8") as f:
        problems: List[Dict] = json.load(f)
    log.info("%d problems loaded", len(problems))

    # ── Simple methods ─────────────────────────────────────────────────────────

    if args.method == "original":
        _run_simple(
            problems, "original",
            priori_dir / "original",
            args.dataset, args.seed, resume, task_filter,
        )
        return

    if args.method == "random":
        _run_simple(
            problems, "random",
            priori_dir / "random",
            args.dataset, args.seed, resume, task_filter,
        )
        return

    if args.method == "ncd":
        log.info("Loading TC source code for NCD ...")
        if args.dataset == "bcb":
            tc_source_all = _load_bcb_tc_source(args.dataset_path)
        else:
            tc_cache = args.tc_source_cache or defaults["tc_source_cache"]
            tc_source_all = _load_lcb_tc_source(tc_cache)
        log.info("TC sources loaded for %d tasks", len(tc_source_all))
        _run_simple(
            problems, "ncd",
            priori_dir / f"ncd_{args.compressor}",
            args.dataset, args.seed, resume, task_filter,
            tc_source_all=tc_source_all,
            compressor=args.compressor,
        )
        return

    if args.method == "input_len_desc":
        log.info("Loading TC source code for input_len_desc ...")
        if args.dataset == "bcb":
            tc_source_all = _load_bcb_tc_source(args.dataset_path)
        else:
            tc_cache = args.tc_source_cache or defaults["tc_source_cache"]
            tc_source_all = _load_lcb_tc_source(tc_cache)
        log.info("TC sources loaded for %d tasks", len(tc_source_all))
        _run_simple(
            problems, "input_len_desc",
            priori_dir / "input_len_desc",
            args.dataset, args.seed, resume, task_filter,
            tc_source_all=tc_source_all,
        )
        return

    # ── VP methods ─────────────────────────────────────────────────────────────

    log.info("Loading VP file: %s", args.vp_file)
    if args.dataset == "bcb":
        pred_map, vp_type, model_name = load_bcb_vp(args.vp_file)
    else:
        pred_map, vp_type, model_name = load_lcb_vp(args.vp_file)
    log.info("VP records loaded: %d  (vp_type=%s, model=%s)", len(pred_map), vp_type, model_name)

    # Load TC source for length-based VP method
    vp_source_all: Optional[Dict[str, Dict[str, str]]] = None
    if args.method in LENGTH_METHODS:
        log.info("Loading TC source code for vp_length ...")
        if args.dataset == "bcb":
            vp_source_all = _load_bcb_tc_source(args.dataset_path)
        else:
            tc_cache = args.tc_source_cache or defaults["tc_source_cache"]
            vp_source_all = _load_lcb_tc_source(tc_cache)
        log.info("TC sources loaded for %d tasks", len(vp_source_all))

    # Load token map for token-based methods
    token_map: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None
    if args.method in TOKEN_METHODS:
        if args.token_report_file:
            report_file = Path(args.token_report_file)
        else:
            report_file = priori_dir / "token_count" / args.token_mode / vp_type / model_name / "token_report.json"
        if not report_file.exists():
            print(
                f"[ERROR] Token report not found: {report_file}\n"
                f"  Run calc_tokens first or pass --token-report-file.",
                file=sys.stderr,
            )
            sys.exit(1)
        log.info("Loading token report: %s", report_file)
        token_map = load_token_report(str(report_file))
        log.info("Token map loaded for %d tasks", len(token_map))

    _run_vp(
        problems=problems,
        pred_map=pred_map,
        method=args.method,
        vp_type=vp_type,
        model_name=model_name,
        priori_dir=priori_dir,
        dataset=args.dataset,
        seed=args.seed,
        resume=resume,
        task_filter=task_filter,
        token_map=token_map,
        tc_source_all=vp_source_all,
    )


if __name__ == "__main__":
    main()
