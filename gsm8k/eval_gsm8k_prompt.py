#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_gsm8k.py  (Prompt-length experiment version, levels 0-4)

This script evaluates HF causal/instruct LLMs on GSM8K while ONLY varying
the *prompt/input length level* (0-4). It does NOT control CoT step count.

Prompt levels (0-4):
  0: no prompt, only question
  1: short
  2: medium
  3: long
  4: very long

All levels require the model to output:
  Final Answer: <number>

Defaults are set to minimize required flags:
  python eval_gsm8k.py
  python eval_gsm8k.py --model Qwen/Qwen2.5-7B-Instruct --prompt-level 3

Output:
  results/gsm8k_promptlen/<model_tag>_p{level}_{split}_n{max}.jsonl

Deps:
  pip install -U transformers datasets accelerate torch click
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

# -----------------------------
# Prompt-length templates (0-4)
# -----------------------------

SYSTEM_PROMPT = "You are a careful math solver. Follow instructions exactly."

PROMPT_LEVELS = {0, 1, 2, 3, 4}

PROMPT_L0 = """{question}
"""

PROMPT_L1 = """Answer this question. Think step by step.
{question}

Final Answer:
"""

PROMPT_L2 = """You are a careful math solver.

Instructions:
- Think step by step to solve the problem.
- Output the final numeric answer on a single line in the exact format:
  Final Answer: <number>
- Do not add any extra text after the final answer line.

Question:
{question}
"""

PROMPT_L3 = """You are a careful math solver. Your goal is to produce the correct final numeric answer.

Guidelines:
1) Read the question carefully and identify what is being asked.
2) Use step-by-step reasoning and keep track of units (if any).
3) Avoid common mistakes:
   - copying numbers incorrectly
   - mixing up addition/subtraction
   - forgetting to apply multiplication/division
   - rounding too early
4) Double-check your final result before answering.

Output format (required):
- Provide the final answer on a single line:
  Final Answer: <number>
- Do not include any other text after that line.

Question:
{question}
"""

PROMPT_L4 = """You are a careful math problem solver. Follow the process below.

Process:
A. Understand
- Restate the problem in your own words.
- Identify the quantities involved and what needs to be computed.

B. Plan
- Decide which operations are needed (addition, subtraction, multiplication, division).
- If the problem involves multiple steps, keep intermediate values consistent.

C. Execute
- Compute step by step.
- Keep track of intermediate results.
- If there are units (dollars, minutes, items), ensure consistency.

D. Verify
- Check whether the answer is reasonable (e.g., not negative when it should be positive).
- Re-check arithmetic quickly.
- Ensure you answered the question being asked (not a related quantity).

Important constraints:
- Do not use external tools.
- Do not rely on unstated assumptions.
- Do not include explanations in the final output.

Required output format:
Final Answer: <number>

Question:
{question}
"""

_LEVEL_TO_TEMPLATE = {
    0: PROMPT_L0,
    1: PROMPT_L1,
    2: PROMPT_L2,
    3: PROMPT_L3,
    4: PROMPT_L4,
}


def build_user_prompt(question: str, level: int) -> str:
    if level not in _LEVEL_TO_TEMPLATE:
        raise ValueError(f"Unknown prompt level: {level}")
    return _LEVEL_TO_TEMPLATE[level].format(question=question)


def format_chat_if_possible(tokenizer, system_prompt: str, user_prompt: str, use_chat_template: bool) -> str:
    """
    If tokenizer has apply_chat_template, use it (recommended for instruct models).
    Otherwise, fall back to plain concatenation.
    """
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"{system_prompt}\n\n{user_prompt}".strip()


# -----------------------------
# Answer extraction
# -----------------------------
_FINAL_ANSWER_RE = re.compile(
    r"Final Answer\s*:\s*([-\+]?\$?\s*\d[\d,]*\.?\d*)",
    flags=re.IGNORECASE,
)
_LAST_NUMBER_RE = re.compile(r"([-\+]?\d[\d,]*\.?\d*)")


def _normalize_num(s: str) -> str:
    s = s.strip()
    s = s.replace("$", "").strip()
    s = s.replace(",", "")
    s = s.rstrip(".")
    return s


def extract_number(text: str) -> Optional[str]:
    """
    Priority:
      1) explicit 'Final Answer:'
      2) fallback to last number in text
    """
    if not text:
        return None
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return _normalize_num(m.group(1))
    nums = _LAST_NUMBER_RE.findall(text)
    if not nums:
        return None
    return _normalize_num(nums[-1])


def correct(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return float(pred) == float(gold)
    except Exception:
        return pred == gold


# -----------------------------
# Dataset
# -----------------------------
def load_gsm8k(split: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [dict(x) for x in ds]


def pick_indices(n: int, max_samples: Optional[int], seed: int, shuffle: bool) -> List[int]:
    import random

    idx = list(range(n))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(idx)
    if max_samples is not None and max_samples > 0:
        idx = idx[:max_samples]
    return idx


# -----------------------------
# Model loading & generation
# -----------------------------
@dataclass
class RuntimeCfg:
    temperature: float
    top_p: float
    max_new_tokens: int
    use_chat_template: bool


def load_model(model_name: str, device: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Ensure pad token exists
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype.lower())
    if torch_dtype is None:
        raise ValueError("dtype must be float16|bfloat16|float32")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "auto" else None,
    )
    if device != "auto":
        model.to(device)
    model.eval()
    return model, tok


def generate(model, tokenizer, prompt: str, cfg: RuntimeCfg) -> Tuple[str, int, int]:
    """
    Returns: (generated_text, input_tokens, output_tokens)
    """
    import torch

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, padding=False)
    input_ids = enc["input_ids"].to(model.device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(model.device)

    input_tokens = int(input_ids.shape[-1])

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=(cfg.temperature > 0),
            temperature=cfg.temperature if cfg.temperature > 0 else None,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0, input_tokens:]
    output_tokens = int(gen_ids.shape[-1])
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, input_tokens, output_tokens


def model_tag(model_name: str) -> str:
    return model_name.split("/")[-1].lower().replace(" ", "_")


# -----------------------------
# CLI
# -----------------------------
@click.command()
@click.option("--model", "model_name", default="Qwen/Qwen2.5-1.5B-Instruct", show_default=True, type=str)
@click.option("--split", default="test", show_default=True, type=click.Choice(["train", "test"], case_sensitive=False))
@click.option("--prompt-level", default=2, show_default=True, type=click.IntRange(0, 4), help="Prompt length level 0..4")
@click.option("--results-dir", default="results/gsm8k_promptlen", show_default=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--max-samples", default=None, type=int, help="Run only first N examples (after shuffle).")
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--shuffle/--no-shuffle", default=True, show_default=True)
@click.option("--device", default="auto", show_default=True, type=str, help="auto|cuda|cpu|cuda:0 ...")
@click.option("--dtype", default="bfloat16", show_default=True,
              type=click.Choice(["float16", "bfloat16", "float32"], case_sensitive=False))
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--top-p", default=1.0, show_default=True, type=float)
@click.option("--max-new-tokens", default=256, show_default=True, type=int)
@click.option("--use-chat-template/--no-chat-template", default=True, show_default=True)
def main(
    model_name: str,
    split: str,
    prompt_level: int,
    results_dir: Path,
    max_samples: Optional[int],
    seed: int,
    shuffle: bool,
    device: str,
    dtype: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    use_chat_template: bool,
):
    if prompt_level not in PROMPT_LEVELS:
        raise click.ClickException(f"prompt_level must be in {sorted(PROMPT_LEVELS)}")

    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"{model_tag(model_name)}_p{prompt_level}_{split}"
    if max_samples is not None:
        out_path = out_path.with_name(out_path.name + f"_n{max_samples}")
    out_path = out_path.with_suffix(".jsonl")

    click.echo(f"[cfg] model={model_name}")
    click.echo(f"[cfg] split={split} prompt_level={prompt_level}")
    click.echo(f"[cfg] temperature={temperature} top_p={top_p} max_new_tokens={max_new_tokens} device={device} dtype={dtype}")
    click.echo(f"[out] {out_path}")

    rows = load_gsm8k(split)
    idxs = pick_indices(len(rows), max_samples, seed, shuffle)
    model, tok = load_model(model_name, device=device, dtype=dtype)

    run_cfg = RuntimeCfg(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        use_chat_template=use_chat_template,
    )

    t0 = time.time()
    correct_cnt = 0

    with out_path.open("w", encoding="utf-8") as f:
        for k, i in enumerate(idxs, start=1):
            q = rows[i]["question"]
            gold_raw = rows[i]["answer"]
            gold_num = extract_number(gold_raw)

            user_prompt = build_user_prompt(q, prompt_level)
            prompt = format_chat_if_possible(tok, SYSTEM_PROMPT, user_prompt, use_chat_template)

            start = time.time()
            gen_text, in_tok, out_tok = generate(model, tok, prompt, run_cfg)
            latency = time.time() - start

            pred_num = extract_number(gen_text)
            ok = correct(pred_num, gold_num)
            correct_cnt += int(ok)

            rec = {
                "dataset": "gsm8k",
                "split": split,
                "qid": i,
                "model_name": model_name,
                "prompt_level": prompt_level,
                "question": q,
                "gold_raw": gold_raw,
                "gold_num": gold_num,
                "pred_raw": gen_text,
                "pred_num": pred_num,
                "correct": ok,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_sec": latency,
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if k % 25 == 0 or k == len(idxs):
                acc = correct_cnt / k
                click.echo(f"[{k}/{len(idxs)}] acc={acc:.3f} last_pred={pred_num} gold={gold_num} in_tok={in_tok} out_tok={out_tok}")

    total = time.time() - t0
    click.echo(f"[done] n={len(idxs)} acc={correct_cnt/len(idxs):.4f} time={total:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()