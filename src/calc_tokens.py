"""
Unified token counting for BigCodeBench-Hard and LiveCodeBench VP experiments.
Supports --dataset bcb|lcb, matching the pattern in calc_accuracy.py.

Input tokens  = tokens in the entire formatted LLM prompt
                (template + generated code + test-case text).
Output tokens = tokens in the raw LLM response.

Tokenizer auto-detected from model name in --raw-file path:
  qwen3-coder-*  → HuggingFace AutoTokenizer (Qwen/Qwen3-Coder-30B-A3B-Instruct)
  gpt-*          → tiktoken o200k_base
  claude-*       → tiktoken cl100k_base
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

log = logging.getLogger(__name__)

_MODEL_TOKENIZER_MAP = [
    ("qwen3",  "hf:Qwen/Qwen3-Coder-30B-A3B-Instruct"),
    ("gpt",    "tiktoken:o200k_base"),
    ("claude", "tiktoken:cl100k_base"),
]


# ── Tokenizer ──────────────────────────────────────────────────────────────────

def _resolve_tokenizer_spec(raw_file: str) -> str:
    model_name = Path(raw_file).parent.name.lower()
    for keyword, spec in _MODEL_TOKENIZER_MAP:
        if keyword in model_name:
            return spec
    log.warning("Could not auto-detect tokenizer for '%s'; falling back to tiktoken:cl100k_base", model_name)
    return "tiktoken:cl100k_base"


def _load_tokenizer(spec: str):
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer
        model_id = spec[3:]
        log.info("Loading HuggingFace tokenizer: %s", model_id)
        return ("hf", AutoTokenizer.from_pretrained(model_id))
    elif spec.startswith("tiktoken:"):
        import tiktoken
        enc_name = spec[9:]
        log.info("Loading tiktoken encoding: %s", enc_name)
        return ("tiktoken", tiktoken.get_encoding(enc_name))
    else:
        raise ValueError(f"Unknown tokenizer spec: {spec!r}")


def count_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    kind, tok = tokenizer
    if kind == "hf":
        return len(tok.encode(text, add_special_tokens=False))
    else:
        return len(tok.encode(text))


# ── Dataset-specific loaders ───────────────────────────────────────────────────

def _load_bcb_tc_texts(dataset_path: Optional[str]) -> Dict[str, Dict[str, str]]:
    """Returns {task_id: {tc_name: source_code}} from BigCodeBench-Hard dataset."""
    from BigCodeBench_Hard.verdict_prediction.utils import load_bigcodebench_hard, split_test_cases
    problems = load_bigcodebench_hard(dataset_path)
    return {
        p["task_id"]: {name: code for name, code in split_test_cases(p["test"])}
        for p in problems
    }


def _load_bcb_template_map() -> Dict[str, str]:
    from BigCodeBench_Hard.verdict_prediction.vp_template import (
        PRED_PROMPT_TMPL, BUG_REPORT_PROMPT_TMPL, BUG_LOCAL_PROMPT_TMPL
    )
    return {
        "direct_verdict":   PRED_PROMPT_TMPL,
        "reasoned_verdict": BUG_REPORT_PROMPT_TMPL,
        "failure_analysis": BUG_LOCAL_PROMPT_TMPL,
    }


def _load_bcb_codes(generated_code_path: str) -> Dict[str, str]:
    """Returns {task_id: first_code_str} from nucleus_code_generate.json."""
    with open(generated_code_path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("raw") or data.get("code") or data
    result: Dict[str, str] = {}
    for task_id, codes in raw.items():
        if isinstance(codes, list) and codes:
            result[task_id] = codes[0]
        elif isinstance(codes, str):
            result[task_id] = codes
    return result


def _load_lcb_tc_texts(tc_source_cache: str) -> Dict[str, Dict[str, str]]:
    """Returns {question_id: {test_case_1: text, ...}} from cache file."""
    with open(tc_source_cache, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): v for k, v in raw.items()}


def _load_lcb_template_map() -> Dict[str, str]:
    from LiveCodeBench.verdict_prediction.vp_template import METHOD_PROMPTS
    return dict(METHOD_PROMPTS)


def _load_lcb_codes(eval_file: str, code_index: int = 0) -> Dict[str, str]:
    """Returns {question_id: code_str} for the given code_index."""
    with open(eval_file, encoding="utf-8") as f:
        data = json.load(f)
    result: Dict[str, str] = {}
    for item in data:
        qid = str(item["question_id"])
        codes: List[str] = item.get("code_list", [])
        if code_index < len(codes):
            result[qid] = codes[code_index]
    return result


# ── Core ───────────────────────────────────────────────────────────────────────

def build_token_map(
    dataset: str,
    raw_file: str,
    tokenizer_spec: Optional[str] = None,
    # BCB-only
    dataset_path: Optional[str] = None,
    generated_code_path: Optional[str] = None,
    # LCB-only
    tc_source_cache: Optional[str] = None,
    eval_file: Optional[str] = None,
    code_index: int = 0,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Build a per-task, per-TC token count map.

    Input tokens  = tokens in the full LLM prompt (template + code + TC text).
    Output tokens = tokens in the raw LLM response.

    Returns
    -------
    {task_id: {tc_name: {"input_tokens": int, "output_tokens": int}}}
    """
    if dataset not in ("bcb", "lcb"):
        raise ValueError(f"Unknown dataset: {dataset!r}")

    spec = tokenizer_spec or _resolve_tokenizer_spec(raw_file)
    log.info("Tokenizer spec: %s", spec)
    tokenizer = _load_tokenizer(spec)

    # Load TC texts
    if dataset == "bcb":
        log.info("Loading BCB dataset for TC texts ...")
        tc_text_map = _load_bcb_tc_texts(dataset_path)
        if not generated_code_path:
            raise ValueError("--generated-code-path is required for bcb")
        template_map = _load_bcb_template_map()
        log.info("Loading BCB generated codes: %s", generated_code_path)
        code_map = _load_bcb_codes(generated_code_path)
    else:
        log.info("Loading LCB TC source cache: %s", tc_source_cache)
        tc_text_map = _load_lcb_tc_texts(tc_source_cache)
        if not eval_file:
            raise ValueError("--eval-file is required for lcb")
        template_map = _load_lcb_template_map()
        log.info("Loading LCB generated codes (index=%d): %s", code_index, eval_file)
        code_map = _load_lcb_codes(eval_file, code_index)
    log.info("TC texts loaded for %d tasks, codes for %d tasks", len(tc_text_map), len(code_map))

    vp_type = Path(raw_file).parent.parent.name
    template = template_map.get(vp_type)
    if template is None:
        raise ValueError(f"No template found for vp_type: {vp_type!r}")
    log.info("Using template for vp_type: %s", vp_type)

    # Read test_raw.jsonl
    raw_records: Dict[str, Dict[str, Optional[str]]] = {}
    with open(raw_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if dataset == "lcb" and rec.get("code_index", 0) != code_index:
                continue
            task_id = str(rec.get("id") or rec.get("question_id", ""))
            if task_id:
                raw_records[task_id] = rec.get("raw_responses", {})
    log.info("Loaded raw responses for %d tasks", len(raw_records))

    # Assemble token map
    token_map: Dict[str, Dict[str, Dict[str, int]]] = {}
    for task_id, raw_responses in raw_records.items():
        tc_texts = tc_text_map.get(task_id, {})
        token_map[task_id] = {}
        for tc_key, response in raw_responses.items():
            tc_text = tc_texts.get(tc_key, "")
            code = code_map.get(task_id, "")
            full_prompt = template.format(code=code, testcase=tc_text)
            input_tok = count_tokens(full_prompt, tokenizer)
            output_tok = count_tokens(response or "", tokenizer)
            token_map[task_id][tc_key] = {
                "input_tokens": input_tok,
                "output_tokens": output_tok,
            }

    return token_map


def summarize_token_map(token_map: Dict[str, Dict[str, Dict[str, int]]]) -> Dict:
    all_input: List[int] = []
    all_output: List[int] = []
    per_task = {}

    for task_id, tcs in token_map.items():
        t_in  = [v["input_tokens"]  for v in tcs.values()]
        t_out = [v["output_tokens"] for v in tcs.values()]
        all_input.extend(t_in)
        all_output.extend(t_out)
        per_task[task_id] = {
            "num_tcs":              len(tcs),
            "total_input_tokens":   sum(t_in),
            "total_output_tokens":  sum(t_out),
            "avg_input_tokens":     round(sum(t_in)  / len(t_in),  2) if t_in  else 0,
            "avg_output_tokens":    round(sum(t_out) / len(t_out), 2) if t_out else 0,
            "max_input_tokens":     max(t_in)  if t_in  else 0,
            "max_output_tokens":    max(t_out) if t_out else 0,
        }

    def _avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0

    return {
        "num_tasks":     len(token_map),
        "num_tcs_total": len(all_input),
        "global": {
            "total_input_tokens":       sum(all_input),
            "total_output_tokens":      sum(all_output),
            "total_tokens":             sum(all_input) + sum(all_output),
            "avg_input_tokens_per_tc":  _avg(all_input),
            "avg_output_tokens_per_tc": _avg(all_output),
            "max_input_tokens":         max(all_input)  if all_input  else 0,
            "max_output_tokens":        max(all_output) if all_output else 0,
        },
        "per_task": per_task,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    LCB_DIR = SRC_DIR / "LiveCodeBench"

    parser = argparse.ArgumentParser(
        description="Count input/output tokens per TC from a test_raw.jsonl file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=["bcb", "lcb"],
                        help="bcb = BigCodeBench-Hard, lcb = LiveCodeBench")
    parser.add_argument("--raw-file", required=True, help="Path to test_raw.jsonl")
    parser.add_argument("--tokenizer", default=None, metavar="SPEC",
                        help="Override: 'hf:<model_id>' or 'tiktoken:<enc>'. Auto-detected if omitted.")
    parser.add_argument("--output", default=None, metavar="PATH",
                        help="Save token map as JSON to this path")

    bcb_group = parser.add_argument_group("BCB options")
    bcb_group.add_argument("--dataset-path", default=None,
                           help="[bcb] Local BigCodeBench-Hard dataset path (HF Hub if omitted)")
    bcb_group.add_argument("--generated-code-path", default=None,
                           help="[bcb] Path to nucleus_code_generate.json (required)")

    lcb_group = parser.add_argument_group("LCB options")
    lcb_group.add_argument("--tc-source-cache", default=None,
                           help="[lcb] Path to tc_source_cache.json")
    lcb_group.add_argument("--eval-file", default=None,
                           help="[lcb] Path to filtered_output_eval_all.json (required)")
    lcb_group.add_argument("--code-index", type=int, default=0,
                           help="[lcb] Code index to use for generated code")

    args = parser.parse_args()

    # Apply dataset-specific defaults
    if args.dataset == "lcb":
        if args.tc_source_cache is None:
            args.tc_source_cache = str(LCB_DIR / "results/tc_source_cache.json")
        if args.eval_file is None:
            args.eval_file = str(LCB_DIR / "actual_exec/results/filtered_output_eval_all.json")

    spec = args.tokenizer or _resolve_tokenizer_spec(args.raw_file)
    token_map = build_token_map(
        dataset=args.dataset,
        raw_file=args.raw_file,
        tokenizer_spec=args.tokenizer,
        dataset_path=getattr(args, "dataset_path", None),
        generated_code_path=getattr(args, "generated_code_path", None),
        tc_source_cache=getattr(args, "tc_source_cache", None),
        eval_file=getattr(args, "eval_file", None),
        code_index=getattr(args, "code_index", 0),
    )
    summary = summarize_token_map(token_map)

    g = summary["global"]
    print(f"\n=== Token Report ===")
    print(f"  Dataset          : {args.dataset}")
    print(f"  Raw file         : {args.raw_file}")
    print(f"  Tokenizer        : {spec}")
    print(f"  Tasks            : {summary['num_tasks']}")
    print(f"  Total TCs        : {summary['num_tcs_total']}")
    print(f"  Input  tokens    : {g['total_input_tokens']:,}  (avg {g['avg_input_tokens_per_tc']:.1f}/TC, max {g['max_input_tokens']})")
    print(f"  Output tokens    : {g['total_output_tokens']:,}  (avg {g['avg_output_tokens_per_tc']:.1f}/TC, max {g['max_output_tokens']})")
    print(f"  Total  tokens    : {g['total_tokens']:,}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(token_map, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to: {out}")
