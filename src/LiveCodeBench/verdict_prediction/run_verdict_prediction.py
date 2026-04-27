"""
Verdict Prediction Generation for LiveCodeBench.

Methods:   direct_verdict, reasoned_verdict, failure_analysis
Models:    qwen3-coder-30B-A3B-instruct, claude-haiku-4-5-20251001, gpt-5-mini-2025-08-07
Code idx:  0 only

Output (per method/model):
  results/verdict_pred/{method}/{model_name}/
    test.jsonl      - parsed PASS/FAIL + latency per TC + overall_pass
    test_raw.jsonl  - raw LLM responses per TC
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import base64
import pickle
import zlib
import tiktoken
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from tqdm import tqdm

from vp_template import METHOD_PROMPTS

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="log_vp_generation.txt",
    filemode="a",
)

MAX_CONCURRENT_TASKS = 5
TIMEOUT_SECONDS = 120
MAX_RETRIES = 2
CODE_INDEX = 0

# Per-model input token limits
MODEL_MAX_INPUT_TOKENS: dict = {
    "claude": 185_000,
    "gpt-5-mini": 258_000,
}

_TOKENIZER: Optional[tiktoken.Encoding] = None


def get_tokenizer() -> tiktoken.Encoding:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = tiktoken.get_encoding("o200k_base")
    return _TOKENIZER


def get_max_input_tokens(model: str) -> Optional[int]:
    for prefix, limit in MODEL_MAX_INPUT_TOKENS.items():
        if prefix in model:
            return limit
    return None


# ─── helpers ──────────────────────────────────────────────────────────────────

def extract_block(s: str) -> str:
    if not s:
        return ""
    m = re.search(r"```plaintext[a-zA-Z0-9_-]*\n(.*?)```", s, re.DOTALL)
    return (m.group(1) if m else s).strip()


def parse_result(s: Optional[str]) -> str:
    if not s or not s.strip():
        return "NULL"
    s_upper = s.upper()
    m = re.search(r"\[RESULTS?\]([\s\S]*?)(?:\[BUG LOCALIZATION\]|\[EXPLANATION\]|$)", s_upper)
    if m:
        sec = m.group(1)
        if "[PASS]" in sec or re.search(r"\bPASS\b", sec):
            return "PASS"
        if "[FAIL]" in sec or re.search(r"\bFAIL\b", sec):
            return "FAIL"
    candidates = re.findall(r"\b(PASS|FAIL)\b", s_upper)
    return candidates[-1] if candidates else "NULL"


def decode_private_test_cases(s: str) -> list:
    if not s:
        return []
    return json.loads(
        pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8"))))
    )


def build_testcase_strings(row: dict) -> List[str]:
    public_raw = row["public_test_cases"]
    public_cases = json.loads(public_raw) if isinstance(public_raw, str) else public_raw
    private_cases = decode_private_test_cases(row["private_test_cases"])
    all_cases = public_cases + private_cases
    return [
        json.dumps({"input": c.get("input"), "output": c.get("output")}, ensure_ascii=False)
        for c in all_cases
    ]


def load_actual_exec_data(actual_exec_path: str) -> Dict[str, dict]:
    with open(actual_exec_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    index: Dict[str, dict] = {}
    for item in data:
        qid = str(item.get("question_id", "")).strip()
        if qid:
            index[qid] = item
    return index


def load_lcb_index(target_qids: Set[str]) -> Dict[str, dict]:
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag="release_latest",
    )
    index: Dict[str, dict] = {}
    for row in ds:
        qid = str(row.get("question_id") or row.get("id") or row.get("problem_id") or "")
        if qid in target_qids:
            index[qid] = row
    missing = len(target_qids - set(index.keys()))
    logging.info(f"LCB index: target={len(target_qids)}, matched={len(index)}, missing={missing}")
    return index


# ─── LLM call ─────────────────────────────────────────────────────────────────

async def call_llm(
    prompt: str,
    client,
    model_id: str,
    is_anthropic: bool,
    temperature: Optional[float],
) -> Tuple[Optional[str], float]:
    """Returns (raw_content, latency_seconds)."""
    for attempt in range(1, MAX_RETRIES + 2):
        t0 = time.perf_counter()
        try:
            if is_anthropic:
                kwargs = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8192,
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                resp = await asyncio.wait_for(
                    client.messages.create(**kwargs),
                    timeout=TIMEOUT_SECONDS,
                )
                latency = time.perf_counter() - t0
                content = resp.content[0].text if resp.content else ""
                return content, latency
            else:
                kwargs = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 8192,
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                resp = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=TIMEOUT_SECONDS,
                )
                latency = time.perf_counter() - t0
                content = resp.choices[0].message.content or ""
                return content, latency
        except asyncio.TimeoutError:
            logging.warning(f"call_llm timeout (attempt {attempt})")
        except Exception as e:
            err_str = str(e)
            if "context_length_exceeded" in err_str or "prompt is too long" in err_str:
                logging.warning(f"call_llm context length exceeded, skipping.")
                return None, 0.0
            if "rate_limit_error" in err_str or "429" in err_str:
                logging.warning(f"call_llm rate limit (attempt {attempt}), sleeping 60s")
                await asyncio.sleep(60)
            else:
                logging.error(f"call_llm exception (attempt {attempt}): {e}")
    return None, 0.0


# ─── per-problem handler ──────────────────────────────────────────────────────

async def handle_problem(
    qid: str,
    exec_item: dict,
    lcb_row: dict,
    client,
    is_anthropic: bool,
    model_id: str,
    sem: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    raw_path: Path,
    test_path: Path,
    method: str,
    temperature: Optional[float],
    pbar: tqdm,
) -> bool:
    async with sem:
        try:
            code = (exec_item.get("code_list") or [])[CODE_INDEX]
            testcase_strs = build_testcase_strings(lcb_row)
            if not testcase_strs:
                logging.warning(f"No testcases for qid={qid}")
                return False

            prompt_tmpl = METHOD_PROMPTS[method]

            raw_responses: Dict[str, Optional[str]] = {}
            pass_fail_dict: Dict[str, str] = {}
            latency_dict: Dict[str, float] = {}

            for tc_idx, tc_str in enumerate(testcase_strs):
                tc_key = f"test_case_{tc_idx + 1}"
                prompt = prompt_tmpl.format(code=code, testcase=tc_str)
                content, latency = await call_llm(
                    prompt=prompt,
                    client=client,
                    model_id=model_id,
                    is_anthropic=is_anthropic,
                    temperature=temperature,
                )
                pbar.update(1)

                raw_responses[tc_key] = content
                pass_fail_dict[tc_key] = parse_result(content)
                latency_dict[tc_key] = latency

            pf_values = list(pass_fail_dict.values())
            if any(v == "FAIL" for v in pf_values):
                overall_pass = "FAIL"
            elif all(v == "PASS" for v in pf_values):
                overall_pass = "PASS"
            else:
                overall_pass = "NULL"

            raw_record = {"question_id": qid, "raw_responses": raw_responses}
            test_record = {
                "question_id": qid,
                "num_testcases": len(testcase_strs),
                "code_index": CODE_INDEX,
                "overall_pass": overall_pass,
                "pass_fail_list": pass_fail_dict,
                "latency_list": latency_dict,
            }

            async with file_lock:
                with open(raw_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                with open(test_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(test_record, ensure_ascii=False) + "\n")

            return True

        except Exception as e:
            logging.error(f"handle_problem qid={qid} exception: {e}", exc_info=True)
            return False


# ─── resume helpers ───────────────────────────────────────────────────────────

def load_done_qids(path: Path) -> Set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = str(obj.get("question_id", "")).strip()
                if qid:
                    done.add(qid)
            except json.JSONDecodeError:
                pass
    return done


# ─── main entry ───────────────────────────────────────────────────────────────

async def run(
    actual_exec_path: str,
    out_dir: Path,
    method: str,
    model_id: str,
    model_name: str,
    is_anthropic: bool,
    temperature: Optional[float] = None,
    limit: Optional[int] = None,
    base_url: Optional[str] = None,
) -> int:
    if method not in METHOD_PROMPTS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(METHOD_PROMPTS.keys())}")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "test_raw.jsonl"
    test_path = out_dir / "test.jsonl"

    exec_index = load_actual_exec_data(actual_exec_path)
    all_qids = set(exec_index.keys())
    logging.info(f"Total problems in actual_exec: {len(all_qids)}")

    done_qids = load_done_qids(test_path)
    remaining_qids = all_qids - done_qids
    logging.info(f"Done: {len(done_qids)}, Remaining: {len(remaining_qids)}")

    if not remaining_qids:
        logging.info("All questions already processed.")
        return 0

    lcb_index = load_lcb_index(remaining_qids)

    qids_to_run = [
        qid for qid in remaining_qids
        if qid in lcb_index and CODE_INDEX < len(exec_index[qid].get("code_list") or [])
    ]
    if limit is not None:
        qids_to_run = qids_to_run[:limit]
    logging.info(f"Problems to run: {len(qids_to_run)}")

    if is_anthropic:
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY") or "EMPTY"
        client = AsyncAnthropic(api_key=api_key)
    else:
        api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    sem = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    file_lock = asyncio.Lock()

    total_tcs = sum(
        len(build_testcase_strings(lcb_index[qid]))
        for qid in qids_to_run
    )

    ok_cnt = 0
    pbar = tqdm(total=total_tcs, desc=f"[{method}/{model_name}]", unit="tc")
    try:
        coros = [
            handle_problem(
                qid=qid,
                exec_item=exec_index[qid],
                lcb_row=lcb_index[qid],
                client=client,
                is_anthropic=is_anthropic,
                model_id=model_id,
                sem=sem,
                file_lock=file_lock,
                raw_path=raw_path,
                test_path=test_path,
                method=method,
                temperature=temperature,
                pbar=pbar,
            )
            for qid in qids_to_run
        ]
        for fut in asyncio.as_completed(coros):
            if await fut:
                ok_cnt += 1
    finally:
        pbar.close()

    logging.info(f"Completed: {len(coros)}, Saved: {ok_cnt}")
    return ok_cnt


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _THIS_DIR = Path(__file__).resolve().parent
    _LCB_DIR = _THIS_DIR.parent
    DEFAULT_ACTUAL_EXEC = str(_LCB_DIR / "actual_exec/results/filtered_output.json")
    DEFAULT_OUT_BASE = str(_THIS_DIR / "results/verdict_pred")

    parser = argparse.ArgumentParser(
        description="Run verdict prediction LLM inference for LiveCodeBench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", required=True, choices=list(METHOD_PROMPTS.keys()))
    parser.add_argument("--model", required=True, help="Model ID for API (e.g. claude-haiku-4-5-20251001)")
    parser.add_argument("--model-name", default=None, help="Short name for output dir (defaults to --model)")
    parser.add_argument("--actual-exec", default=DEFAULT_ACTUAL_EXEC)
    parser.add_argument("--out-base", default=DEFAULT_OUT_BASE)
    parser.add_argument("--base-url", default=None, help="API base URL for local vLLM")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model_name = args.model_name or args.model
    is_anthropic = args.model.startswith("claude")
    out_dir = Path(args.out_base) / args.method / model_name

    print(f"Method:     {args.method}")
    print(f"Model:      {args.model}")
    print(f"Model name: {model_name}")
    print(f"Output:     {out_dir}")

    saved = asyncio.run(
        run(
            actual_exec_path=args.actual_exec,
            out_dir=out_dir,
            method=args.method,
            model_id=args.model,
            model_name=model_name,
            is_anthropic=is_anthropic,
            temperature=args.temperature,
            limit=args.limit,
            base_url=args.base_url,
        )
    )
    print(f"Saved: {saved} problems")


if __name__ == "__main__":
    main()
