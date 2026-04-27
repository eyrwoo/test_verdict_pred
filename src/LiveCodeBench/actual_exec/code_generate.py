import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

_ACTUAL_EXEC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ACTUAL_EXEC_DIR))

load_dotenv(_ACTUAL_EXEC_DIR.parents[2] / ".env")

from prompt_template import GENERATION_PROMPT_TMPL, STARTER_CODE_SECTION_TMPL

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


def extract_code_blocks(text: str) -> List[str]:
    """Extract all ```python ... ``` blocks from text."""
    return re.findall(r"```python\s*(.*?)```", text, re.DOTALL)


def load_lcb_problems(cache_dir: Optional[str] = None) -> List[Dict]:
    """Load LiveCodeBench problems from HF local cache (dynamic snapshot search)."""
    if cache_dir:
        snapshot_dir = Path(cache_dir)
    else:
        hub_dir = Path.home() / ".cache/huggingface/hub/datasets--livecodebench--code_generation_lite/snapshots"
        snapshots = sorted(hub_dir.iterdir()) if hub_dir.exists() else []
        if not snapshots:
            raise RuntimeError(
                "LCB dataset not found in HF cache. "
                "Run: python3 -c \"from datasets import load_dataset; load_dataset('livecodebench/code_generation_lite', trust_remote_code=True)\""
            )
        snapshot_dir = snapshots[-1]  # use latest snapshot

    jsonl_files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
    problems = []
    for fname in jsonl_files:
        fpath = snapshot_dir / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))

    logging.info("Loaded %d LCB problems from %s", len(problems), snapshot_dir)
    return problems


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
    return AsyncOpenAI(
        api_key=model_cfg.get("api_key"),
        base_url=model_cfg.get("base_url"),
    )


async def process_problem(
    question_id: str,
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
    return question_id, sample_idx, raw_resp, extracted


async def generate_for_model(
    model_name: str,
    model_cfg: Dict,
    problems: List[Dict],
    output_dir: Path,
    n_samples: int = 10,
) -> None:
    """Generate n_samples codes per problem and write one JSONL line per problem.

    Output line format (one per problem):
      {"question_id": ..., "code_list": [...], "question_content": ...,
       "starter_code": ..., "platform": ...}
    """
    client = build_client(model_cfg)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    out_dir = output_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "nucleus_samples.jsonl"
    raw_path = out_dir / "nucleus_samples_raw.jsonl"

    # Resume: load already-complete question_ids (all n_samples done)
    done: Dict[str, List[str]] = {}  # question_id -> code_list (length == n_samples means complete)
    if samples_path.exists():
        with open(samples_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    qid = rec["question_id"]
                    if len(rec.get("code_list", [])) >= n_samples:
                        done[qid] = rec["code_list"]
                except Exception:
                    pass
        logging.info("Resuming: %d problems already complete", len(done))

    # Collect pending problems
    pending = [p for p in problems if p["question_id"] not in done]
    if not pending:
        logging.info("All problems already complete")
        return

    logging.info("Generating %d problems x %d samples each", len(pending), n_samples)

    # Build all individual sample tasks
    tasks = []
    for problem in pending:
        qid = problem["question_id"]
        starter = problem.get("starter_code", "").strip()
        starter_section = STARTER_CODE_SECTION_TMPL.format(starter_code=starter) if starter else ""
        prompt = GENERATION_PROMPT_TMPL.format(
            question_content=problem["question_content"],
            starter_code_section=starter_section,
        )
        for i in range(n_samples):
            tasks.append(asyncio.create_task(
                process_problem(
                    qid, i, prompt, client, semaphore,
                    temperature=0.7, top_p=0.95,
                    model_cfg=model_cfg,
                )
            ))

    # Accumulate results per question_id
    # Using a dict to collect samples as they complete
    in_progress: Dict[str, List[Optional[str]]] = {p["question_id"]: [None] * n_samples for p in pending}
    problem_meta: Dict[str, Dict] = {p["question_id"]: p for p in pending}

    completed = 0
    write_lock = asyncio.Lock()

    for coro in asyncio.as_completed(tasks):
        qid, sample_idx, raw_resp, extracted = await coro
        code = extracted[0] if extracted else ""
        in_progress[qid][sample_idx] = code

        async with write_lock:
            if raw_resp:
                with open(raw_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "question_id": qid,
                        "_sample_idx": sample_idx,
                        "raw": raw_resp,
                    }) + "\n")

        completed += 1
        if completed % 50 == 0:
            logging.info("Progress: %d/%d tasks completed", completed, len(tasks))

    # Write completed records to JSONL (one line per problem)
    with open(samples_path, "a", encoding="utf-8") as f:
        for problem in pending:
            qid = problem["question_id"]
            code_list = [c or "" for c in in_progress[qid]]
            f.write(json.dumps({
                "question_id": qid,
                "code_list": code_list,
                "question_content": problem["question_content"],
                "starter_code": problem.get("starter_code", ""),
                "platform": problem.get("platform", ""),
            }) + "\n")

    logging.info("Done → %s (%d problems)", samples_path, len(pending))


async def main(
    output_root: str,
    limit: Optional[int],
    models: Dict[str, Dict],
    n_samples: int,
    cache_dir: Optional[str] = None,
) -> None:
    problems = load_lcb_problems(cache_dir)
    if not problems:
        raise RuntimeError("No problems found.")
    if limit is not None:
        problems = problems[:limit]

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, cfg in models.items():
        await generate_for_model(model_name, cfg, problems, output_dir, n_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(_ACTUAL_EXEC_DIR / "results"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of problems to generate (e.g., 3 for dry run).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples to generate per problem (default: 10).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Subset of models to run (default: all defaults).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to HF dataset snapshot directory (auto-detected if omitted).",
    )
    args = parser.parse_args()

    models = dict(DEFAULT_MODELS)
    if args.models:
        selected = {}
        for m in args.models:
            if m in models:
                selected[m] = models[m]
            else:
                raise ValueError(f"Unknown model '{m}'. Available: {list(models.keys())}")
        models = selected

    asyncio.run(main(args.out_dir, args.limit, models, args.n_samples, args.cache_dir))
