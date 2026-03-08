#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference on both GSM8K and DROP using Qwen 2.5 14B Instruct.

- Model: Qwen/Qwen2.5-14B-Instruct
- Temperature: 0
- Four prompts (conditions): direct, short, medium, long
- Outputs: 8 JSONL files total
  - 4 for GSM8K: results/large/gsm8k/<model_tag>_<condition>_test.jsonl
  - 4 for DROP:  results/large/drop/<model_tag>_<condition>_validation.jsonl

Usage:
  python run_large_inference.py
  python run_large_inference.py --max-samples 100   # limit examples per run
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Project root (parent of script)
ROOT = Path(__file__).resolve().parent

MODEL = "Qwen/Qwen2.5-14B-Instruct"
CONDITIONS = ["direct", "short", "medium", "long"]
TEMPERATURE = 0


def run_gsm8k(condition: str, results_dir: Path, max_samples: int | None = None) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "gsm8k" / "eval_gsm8k_step.py"),
        "--model", MODEL,
        "--condition", condition,
        "--split", "test",
        "--results-dir", str(results_dir),
        "--temperature", str(TEMPERATURE),
    ]
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])
    return subprocess.call(cmd)


def run_drop(condition: str, results_dir: Path, max_samples: int | None = None) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "drop" / "eval_drop.py"),
        "--model", MODEL,
        "--condition", condition,
        "--split", "validation",
        "--results-dir", str(results_dir),
        "--temperature", str(TEMPERATURE),
    ]
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])
    return subprocess.call(cmd)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run Qwen 2.5 14B inference on GSM8K and DROP (all 4 conditions).")
    p.add_argument("--max-samples", type=int, default=None, help="Limit examples per dataset (for testing).")
    p.add_argument("--gsm8k-only", action="store_true", help="Run only GSM8K.")
    p.add_argument("--drop-only", action="store_true", help="Run only DROP.")
    args = p.parse_args()

    gsm8k_dir = ROOT / "results" / "large" / "gsm8k"
    drop_dir = ROOT / "results" / "large" / "drop"
    gsm8k_dir.mkdir(parents=True, exist_ok=True)
    drop_dir.mkdir(parents=True, exist_ok=True)

    failed = []

    if not args.drop_only:
        for cond in CONDITIONS:
            print(f"\n{'='*60}\n[GSM8K] condition={cond}\n{'='*60}")
            ret = run_gsm8k(cond, gsm8k_dir, args.max_samples)
            if ret != 0:
                failed.append(f"gsm8k_{cond}")

    if not args.gsm8k_only:
        for cond in CONDITIONS:
            print(f"\n{'='*60}\n[DROP] condition={cond}\n{'='*60}")
            ret = run_drop(cond, drop_dir, args.max_samples)
            if ret != 0:
                failed.append(f"drop_{cond}")

    if failed:
        print(f"\nFailed runs: {failed}")
        sys.exit(1)
    print("\nAll runs finished. 8 files total: 4 in results/large/gsm8k/, 4 in results/large/drop/")


if __name__ == "__main__":
    main()
