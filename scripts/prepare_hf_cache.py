#!/usr/bin/env python3
"""Prepare HuggingFace dataset snapshots used by the reproduction container."""

from __future__ import annotations

import argparse
from typing import Iterable

from datasets import load_dataset


def preload_lcb(version_tag: str) -> None:
    print(f"[hf-cache] Loading LiveCodeBench code_generation_lite ({version_tag})...")
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag=version_tag,
        trust_remote_code=True,
    )
    print(f"[hf-cache] LiveCodeBench rows: {len(ds)}")


def preload_bcb(split: str) -> None:
    print(f"[hf-cache] Loading BigCodeBench-Hard ({split})...")
    ds = load_dataset("bigcode/bigcodebench-hard", split=split)
    print(f"[hf-cache] BigCodeBench-Hard rows: {len(ds)}")


def main(datasets: Iterable[str], lcb_version: str, bcb_split: str) -> None:
    selected = set(datasets)
    if "lcb" in selected:
        preload_lcb(lcb_version)
    if "bcb" in selected:
        preload_bcb(bcb_split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download/cache HF datasets required by LCB/BCB experiments."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["lcb", "bcb"],
        default=["lcb", "bcb"],
        help="Datasets to preload into HF_HOME.",
    )
    parser.add_argument(
        "--lcb-version",
        default="release_latest",
        help="LiveCodeBench release tag passed to load_dataset.",
    )
    parser.add_argument(
        "--bcb-split",
        default="v0.1.4",
        help="BigCodeBench-Hard split passed to load_dataset.",
    )
    args = parser.parse_args()
    main(args.datasets, args.lcb_version, args.bcb_split)
