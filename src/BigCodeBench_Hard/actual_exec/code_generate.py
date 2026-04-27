import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

_ACTUAL_EXEC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ACTUAL_EXEC_DIR))

load_dotenv(_ACTUAL_EXEC_DIR.parents[2] / ".env")

from prompt_template import GENERATION_PROMPT_TMPL
from utils import extract_code_blocks, load_bigcodebench_hard

MAX_CONCURRENT_TASKS = 5
TIMEOUT_SECONDS = 120
MAX_RETRIES = 2

MODEL_BASE_URL = os.environ["MODEL_BASE_URL"]

DEFAULT_MODELS: Dict[str, Dict] = {
    "qwen3-coder-30B-A3B-instruct": {
        "api_model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "api_key": "EMPTY",
        "base_url": MODEL_BASE_URL,
        "extra_body": {"repetition_penalty": 1.05},
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=Path(__file__).parent / "log_code_generate.txt",
    filemode="a",
)


async def call_llm(
    prompt: str,
    llm: AsyncOpenAI,
    model_name: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: Optional[Dict] = None,
) -> Optional[str]:
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                kwargs = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body

                resp = await asyncio.wait_for(
                    llm.chat.completions.create(**kwargs),
                    timeout=TIMEOUT_SECONDS,
                )
                return resp.choices[0].message.content or ""
            except asyncio.TimeoutError:
                logging.warning("call_llm timeout (attempt %s)", attempt)
            except Exception as exc:
                logging.error("call_llm error (attempt %s): %s", attempt, exc)
        return None


def build_client(model_cfg: Dict) -> AsyncOpenAI:
    api_key = model_cfg.get("api_key")
    return AsyncOpenAI(
        api_key=api_key,
        base_url=model_cfg.get("base_url"),
    )


async def generate_for_model(
    model_name: str,
    model_cfg: Dict,
    problems: List[Dict[str, str]],
    output_dir: Path,
    sampling_subset: Optional[List[str]] = None,
    n_samples: int = 1,
) -> None:
    client = build_client(model_cfg)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    strategies = {
        "greedy": {"temperature": 0.2, "top_p": 0.95},
        "nucleus": {"temperature": 0.7, "top_p": 0.95},
    }
    if sampling_subset:
        strategies = {k: v for k, v in strategies.items() if k in sampling_subset}

    for strategy, sampling in strategies.items():
        out_dir = output_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        samples_path = out_dir / f"{strategy}_samples.jsonl"
        raw_path = out_dir / f"{strategy}_samples_raw.jsonl"

        # Resume: count already-written samples per task_id
        done: Dict[str, int] = {}
        if samples_path.exists():
            with open(samples_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        tid = json.loads(line)["task_id"]
                        done[tid] = done.get(tid, 0) + 1
                    except Exception:
                        pass
            logging.info("Resuming: %d tasks partially/fully done", len(done))

        # Create tasks (only for samples still needed)
        tasks = []
        for problem in problems:
            task_id = problem["task_id"]
            already_done = done.get(task_id, 0)
            if already_done >= n_samples:
                continue
            prompt = GENERATION_PROMPT_TMPL.format(prompt=problem["prompt"])
            for i in range(n_samples - already_done):
                tasks.append(asyncio.create_task(
                    process_problem(
                        task_id,
                        already_done + i,
                        prompt,
                        client,
                        semaphore,
                        sampling["temperature"],
                        sampling["top_p"],
                        model_cfg,
                    )
                ))

        if not tasks:
            logging.info("All tasks already completed for strategy %s", strategy)
            continue

        logging.info("Starting %d tasks for strategy %s", len(tasks), strategy)

        write_lock = asyncio.Lock()
        completed = 0

        for coro in asyncio.as_completed(tasks):
            task_id, sample_idx, raw_resp, extracted_codes = await coro
            identifier = f"{task_id}_{sample_idx}"
            code = extracted_codes[0] if extracted_codes else ""

            async with write_lock:
                with open(samples_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "task_id": task_id,
                        "solution": code,
                        "_identifier": identifier,
                    }) + "\n")
                if raw_resp:
                    with open(raw_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "task_id": task_id,
                            "_identifier": identifier,
                            "raw": raw_resp,
                        }) + "\n")

            completed += 1
            if completed % 50 == 0:
                logging.info("Progress: %d/%d tasks completed", completed, len(tasks))

        logging.info("Done strategy=%s → %s", strategy, samples_path)


async def process_problem(
    task_id: str,
    sample_idx: int,
    prompt: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    model_cfg: Dict,
) -> Tuple[str, int, Optional[str], List[str]]:
    raw_resp = await call_llm(
        prompt=prompt,
        llm=client,
        model_name=model_cfg["api_model"],
        semaphore=semaphore,
        temperature=temperature,
        top_p=top_p,
        max_tokens=model_cfg.get("max_tokens", 4096),
        extra_body=model_cfg.get("extra_body"),
    )

    extracted = extract_code_blocks(raw_resp or "")
    return task_id, sample_idx, raw_resp, extracted


async def main(
    dataset_path: Optional[str],
    output_root: str,
    limit: Optional[int],
    models: Dict[str, Dict],
    sampling_subset: Optional[List[str]] = None,
    n_samples: int = 1,
) -> None:
    problems = load_bigcodebench_hard(dataset_path)
    if not problems:
        raise RuntimeError("No problems found to generate code for.")
    if limit is not None:
        problems = problems[:limit]

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, cfg in models.items():
        await generate_for_model(model_name, cfg, problems, output_dir, sampling_subset, n_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSON/JSONL. If omitted, load from HuggingFace bigcodebench-hard.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(_ACTUAL_EXEC_DIR / "results"),
        help="Root output directory for generated code.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of problems to generate (e.g., 3 for dry run).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Subset of models to run (default: all defaults).",
    )
    parser.add_argument(
        "--local_free",
        action="store_true",
        help="Use an additional local/free OpenAI-compatible endpoint for dry run.",
    )
    parser.add_argument(
        "--local_model_name",
        type=str,
        default="local-free",
        help="Name key for the local/free model.",
    )
    parser.add_argument(
        "--local_api_model",
        type=str,
        default="qwen2.5-coder-3b-instruct",
        help="Model ID served by the local/free endpoint.",
    )
    parser.add_argument(
        "--local_base_url",
        type=str,
        default="http://localhost:11434/v1",
        help="Base URL of the local/free OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--local_api_key",
        type=str,
        default="EMPTY",
        help="API key for the local/free endpoint (if required).",
    )
    parser.add_argument(
        "--local_max_tokens",
        type=int,
        default=4096,
        help="max_tokens for the local/free endpoint.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        nargs="*",
        default=None,
        help="Subset of strategies to run (greedy, nucleus).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of samples to generate per problem (default: 1).",
    )
    args = parser.parse_args()

    models = dict(DEFAULT_MODELS)
    # Optionally inject a local/free model for dry runs
    if args.local_free:
        models[args.local_model_name] = {
            "api_model": args.local_api_model,
            "api_key": args.local_api_key,
            "base_url": args.local_base_url,
            "max_tokens": args.local_max_tokens,
        }

    # Allow selecting subset
    if args.models:
        selected = {}
        for m in args.models:
            if m in models:
                selected[m] = models[m]
            else:
                raise ValueError(f"Unknown model '{m}'. Available: {list(models.keys())}")
        models = selected

    asyncio.run(main(args.dataset, args.out_dir, args.limit, models, args.sampling, args.n_samples))
