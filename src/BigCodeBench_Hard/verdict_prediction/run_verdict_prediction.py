"""
Verdict Prediction Generation for BigCodeBench-Hard.

Methods: direct_verdict, verdict_with_analysis, verdict_with_diagnosis
Models:  qwen3-coder-30B-A3B-instruct, claude-haiku-4-5-20251001, gpt-5-mini-2025-08-07
Code idx: 0 only

Output (per method/model):
  verdict_prediction/results/verdict_pred/{method}/{model_name}/
    test.jsonl      - parsed PASS/FAIL + latency per TC + overall_pass
    test_raw.jsonl  - raw LLM responses per TC
"""
import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from tqdm.asyncio import tqdm

import sys
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_THIS.parent))

from vp_template import METHOD_PROMPTS
from utils import load_generated_codes, split_test_cases, load_bigcodebench_hard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(_THIS / "log_vp_generation.txt"),
        logging.StreamHandler(),
    ],
)

# ─── configs ──────────────────────────────────────────────────────────────────

METHOD_CONFIGS = {
    "direct_verdict": {
        "template": METHOD_PROMPTS["direct_verdict"],
        "max_concurrent": 5,
        "timeout": 60,
    },
    "verdict_with_analysis": {
        "template": METHOD_PROMPTS["verdict_with_analysis"],
        "max_concurrent": 5,
        "timeout": 60,
    },
    "verdict_with_diagnosis": {
        "template": METHOD_PROMPTS["verdict_with_diagnosis"],
        "max_concurrent": 3,
        "timeout": 120,
    },
}

MODEL_CONFIGS = {
    "qwen3-coder-30B-A3B-instruct": {
        "api_model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "api_key": "EMPTY",
        "base_url": "http://localhost:8008/v1",
        "extra_body": {"repetition_penalty": 1.05},
        "max_tokens": 8192,
        "temperature": 0.0,
    },
    "gpt-5-mini-2025-08-07": {
        "api_model": "gpt-5-mini-2025-08-07",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": None,
        "max_completion_tokens": 8192,
        "temperature": 1,
    },
    "claude-haiku-4-5-20251001": {
        "api_model": "claude-haiku-4-5-20251001",
        "api_key": os.getenv("CLAUDE_API_KEY"),
        "base_url": None,
        "max_tokens": 8192,
        "temperature": 0.0,
        "client_type": "anthropic",
    },
}

# ─── LLM call ─────────────────────────────────────────────────────────────────

async def call_llm(
    client,
    prompt: str,
    semaphore: asyncio.Semaphore,
    model_config: Dict,
    timeout: int,
) -> tuple:
    async with semaphore:
        for attempt in range(3):
            try:
                start_time = time.time()
                if model_config.get("client_type") == "anthropic":
                    kwargs = {
                        "model": model_config["api_model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": model_config.get("max_tokens", 1024),
                    }
                    if model_config.get("top_p") is not None:
                        kwargs["top_p"] = model_config["top_p"]
                    if model_config.get("temperature") is not None:
                        kwargs["temperature"] = model_config["temperature"]
                    response = await asyncio.wait_for(
                        client.messages.create(**kwargs),
                        timeout=timeout,
                    )
                    latency = time.time() - start_time
                    content = response.content[0].text if response.content else ""
                    logging.info(f"Anthropic call finished in {latency:.2f}s")
                    return content, latency
                else:
                    kwargs = {
                        "model": model_config["api_model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": model_config["temperature"],
                        "extra_body": model_config.get("extra_body"),
                    }
                    if "max_completion_tokens" in model_config:
                        kwargs["max_completion_tokens"] = model_config["max_completion_tokens"]
                    else:
                        kwargs["max_tokens"] = model_config["max_tokens"]
                    response = await asyncio.wait_for(
                        client.chat.completions.create(**kwargs),
                        timeout=timeout,
                    )
                    latency = time.time() - start_time
                    content = response.choices[0].message.content
                    if not content or not content.strip():
                        finish_reason = response.choices[0].finish_reason
                        refusal = getattr(response.choices[0].message, "refusal", None)
                        logging.warning(f"Empty content! finish_reason={finish_reason}, refusal={refusal}")
                        content = f"[DEBUG_EMPTY] Reason: {finish_reason} | Refusal: {refusal}"
                    logging.info(f"OpenAI call finished in {latency:.2f}s")
                    return content, latency
            except asyncio.TimeoutError:
                logging.warning(f"Attempt {attempt + 1} timed out after {timeout}s")
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)
        return None, 0.0


# ─── response parsing ──────────────────────────────────────────────────────────

def parse_result(response: Optional[str]) -> str:
    if not response or not response.strip():
        return "NULL"

    response_upper = response.upper()

    # 1. Targeted extraction from [Result] section
    m = re.search(
        r"\[RESULTS?\]([\s\S]*?)(?:\[BUG LOCALIZATION\]|\[EXPLANATION\]|$)",
        response_upper,
    )
    if m:
        sec = m.group(1)
        if "[PASS]" in sec or re.search(r"\bPASS\b", sec):
            return "PASS"
        if "[FAIL]" in sec or re.search(r"\bFAIL\b", sec):
            return "FAIL"

    # 2. Code block fallback
    m = re.search(r"```(?:plaintext|text)?\s*[\r\n]+(.*?)(?:```|$)", response, re.DOTALL | re.IGNORECASE)
    if m:
        block = m.group(1).upper()
        if "[PASS]" in block or "PASS" in block.split():
            return "PASS"
        if "[FAIL]" in block or "FAIL" in block.split():
            return "FAIL"

    # 3. Global explicit tags
    if "[PASS]" in response_upper:
        return "PASS"
    if "[FAIL]" in response_upper:
        return "FAIL"

    # 4. Fallback: last PASS/FAIL word
    candidates = re.findall(r"\b(PASS|FAIL)\b", response_upper)
    if candidates:
        return candidates[-1]

    return "NULL"


# ─── per-task evaluation ───────────────────────────────────────────────────────

async def evaluate_task(
    task_id: str,
    code: str,
    test_cases: List[tuple],
    client,
    semaphore: asyncio.Semaphore,
    model_config: Dict,
    template: str,
    timeout: int,
    code_index: int = 0,
) -> Dict:
    tasks = [
        call_llm(client, template.format(code=code, testcase=tc_code), semaphore, model_config, timeout)
        for _, tc_code in test_cases
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    pass_fail_dict: Dict[str, str] = {}
    latency_dict: Dict[str, float] = {}
    raw_response_dict: Dict[str, Optional[str]] = {}

    for i, (tc_name, _) in enumerate(test_cases):
        res = results[i]
        if isinstance(res, Exception):
            logging.error(f"TC {tc_name} raised exception: {res}")
            content, latency = None, 0.0
        else:
            content, latency = res
        pass_fail_dict[tc_name] = parse_result(content)
        latency_dict[tc_name] = latency
        raw_response_dict[tc_name] = content

    pf_values = list(pass_fail_dict.values())
    if any(v == "FAIL" for v in pf_values):
        overall_pass = "FAIL"
    elif all(v == "PASS" for v in pf_values):
        overall_pass = "PASS"
    else:
        overall_pass = "NULL"

    return {
        "id": task_id,
        "num_testcases": len(test_cases),
        "code_index": code_index,
        "overall_pass": overall_pass,
        "pass_fail_list": pass_fail_dict,
        "latency_list": latency_dict,
        "raw_responses": raw_response_dict,
    }


# ─── main ──────────────────────────────────────────────────────────────────────

async def run(
    generated_code_path: str,
    method: str,
    model_name: str,
    limit: Optional[int] = None,
) -> None:
    model_config = MODEL_CONFIGS[model_name]
    method_config = METHOD_CONFIGS[method]
    template = method_config["template"]
    timeout = method_config["timeout"]

    logging.info(f"Method: {method}, Model: {model_name}")

    gen_data = load_generated_codes(generated_code_path)
    code_map = gen_data.get("raw") or gen_data.get("code") or gen_data

    problems_data = load_bigcodebench_hard()
    problems = []
    for item in problems_data:
        t_id = item["task_id"]
        codes = code_map.get(t_id)
        if codes is None:
            continue
        if isinstance(codes, list) and codes:
            selected_code = codes[0]
        elif isinstance(codes, str):
            selected_code = codes
        else:
            continue
        problems.append({
            "task_id": t_id,
            "code": selected_code,
            "test_code": item["test"],
        })

    if limit:
        problems = problems[:limit]

    # Client setup
    client_type = model_config.get("client_type", "openai")
    if client_type == "anthropic":
        api_key = model_config.get("api_key")
        if not api_key:
            raise ValueError(f"CLAUDE_API_KEY missing for {model_name}")
        client = AsyncAnthropic(api_key=api_key)
    else:
        api_key = model_config.get("api_key") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        client = AsyncOpenAI(api_key=api_key, base_url=model_config.get("base_url"))

    # Output paths
    out_dir = _THIS / "results" / "verdict_pred" / method / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / "test.jsonl"
    raw_path = out_dir / "test_raw.jsonl"

    # Resume support
    done_ids: set = set()
    if test_path.exists():
        with open(test_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    if done_ids:
        logging.info(f"Resuming: {len(done_ids)} already done")

    problems_to_run = [p for p in problems if p["task_id"] not in done_ids]
    if not problems_to_run:
        logging.info("All problems already processed.")
        return

    logging.info(f"Running {len(problems_to_run)} problems (skipped {len(done_ids)})...")

    semaphore = asyncio.Semaphore(method_config["max_concurrent"])
    coros = [
        evaluate_task(
            p["task_id"], p["code"],
            split_test_cases(p["test_code"]),
            client, semaphore, model_config, template, timeout,
        )
        for p in problems_to_run
    ]

    with open(test_path, "a", encoding="utf-8") as f_out, \
         open(raw_path, "a", encoding="utf-8") as f_raw:
        pbar = tqdm(total=len(coros), desc=f"[{method}/{model_name}]")
        for coro in asyncio.as_completed(coros):
            result = await coro
            raw_data = {"id": result["id"], "raw_responses": result.pop("raw_responses")}
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()
            f_raw.write(json.dumps(raw_data, ensure_ascii=False) + "\n")
            f_raw.flush()
            pbar.update(1)
        pbar.close()

    logging.info(f"Done. test.jsonl → {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run verdict prediction LLM inference for BigCodeBench-Hard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generated-code", required=True,
        help="Path to nucleus_code_generate.json",
    )
    parser.add_argument(
        "--method", required=True,
        choices=list(METHOD_CONFIGS.keys()),
        help="Verdict prediction method",
    )
    parser.add_argument(
        "--model", required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model name",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N problems (for testing)",
    )
    args = parser.parse_args()

    asyncio.run(run(args.generated_code, args.method, args.model, args.limit))
