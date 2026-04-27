#!/usr/bin/env python3
"""
Fetch and cache TC input/output source texts from HuggingFace LiveCodeBench dataset.

Usage (standalone):
  python src/fetch_tc_source.py [--output PATH] [--eval-file PATH]

Builds a JSON cache: question_id -> {test_case_N: json_text}
  Only fetches problems whose question_id appears in the eval file.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import pickle
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Set

log = logging.getLogger(__name__)

LCB_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = str(LCB_DIR / "results/tc_source_cache.json")
DEFAULT_EVAL_FILE = str(LCB_DIR / "actual_exec/results/filtered_output_eval_all.json")

HF_REPO = "livecodebench/code_generation_lite"
HF_FILES = [
    "test.jsonl", "test2.jsonl", "test3.jsonl",
    "test4.jsonl", "test5.jsonl", "test6.jsonl",
]


def decode_private_test_cases(s: str) -> List[Dict]:
    if not s:
        return []
    return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8")))))


def load_target_qids(eval_file: str) -> Set[str]:
    """Load question_ids from the eval JSON to use as an allowlist."""
    with open(eval_file, encoding="utf-8") as f:
        problems = json.load(f)
    return {str(p["question_id"]) for p in problems}


def fetch_tc_source(
    output: str = DEFAULT_OUTPUT,
    eval_file: str = DEFAULT_EVAL_FILE,
) -> Dict[str, Dict[str, str]]:
    """
    Download LCB JSONL files from HuggingFace and build a cache mapping:
      question_id -> {test_case_N: json.dumps({"input": ..., "output": ...})}

    Only includes problems whose question_id appears in eval_file.
    Saves the result to `output` and returns it.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("huggingface_hub is required: pip install huggingface-hub")

    target_qids = load_target_qids(eval_file)
    log.info(
        "Downloading LCB dataset from HuggingFace (%s), filtering to %d qids from %s ...",
        HF_REPO,
        len(target_qids),
        eval_file,
    )

    qid_to_source: Dict[str, Dict[str, str]] = {}
    skipped = 0

    for filename in HF_FILES:
        log.info("  Fetching %s ...", filename)
        try:
            path = hf_hub_download(HF_REPO, filename, repo_type="dataset")
        except Exception as e:
            log.warning("  Failed to fetch %s: %s", filename, e)
            continue

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = str(row.get("question_id", ""))
                if not qid or qid not in target_qids:
                    skipped += 1
                    continue

                pub_raw = row.get("public_test_cases", "[]")
                public_cases = json.loads(pub_raw) if isinstance(pub_raw, str) else pub_raw
                private_cases = decode_private_test_cases(row.get("private_test_cases", ""))
                all_cases = public_cases + private_cases

                qid_to_source[qid] = {
                    f"test_case_{i + 1}": json.dumps(
                        {"input": case.get("input"), "output": case.get("output")},
                        ensure_ascii=False,
                    )
                    for i, case in enumerate(all_cases)
                }

    log.info("Fetched %d / %d problems (skipped %d)", len(qid_to_source), len(target_qids), skipped)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qid_to_source, f, ensure_ascii=False)
    log.info("Saved to %s", output)

    return qid_to_source


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Fetch and cache LCB TC source texts from HuggingFace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to write the TC source cache JSON",
    )
    parser.add_argument(
        "--eval-file",
        default=DEFAULT_EVAL_FILE,
        help="Path to filtered_output_eval_all.json (defines which qids to fetch)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if cache already exists",
    )
    args = parser.parse_args()

    if not args.force and Path(args.output).exists():
        log.info("Cache already exists at %s (use --force to re-fetch)", args.output)
        return

    fetch_tc_source(output=args.output, eval_file=args.eval_file)


if __name__ == "__main__":
    main()
