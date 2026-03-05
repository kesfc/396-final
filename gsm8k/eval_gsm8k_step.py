#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_gsm8k.py

GSM8K evaluation focused on *output reasoning length* (steps / output tokens),
NOT prompt input length.

We keep the prompt template fixed across conditions and only change:
  - required number of steps (for CoT conditions)
  - max_new_tokens budget (to enforce length separation)

Conditions:
  - direct: no reasoning, only "Final Answer: <number>"
  - short : 2–3 steps (we use 3)
  - medium: 6–8 steps (we use 8)
  - long  : 12–16 steps (we use 16)

Defaults are set so you can run with minimal flags:
  python eval_gsm8k.py
  python eval_gsm8k.py --model Qwen/Qwen2.5-7B-Instruct --condition long

Outputs JSONL to: results/gsm8k/<model_tag>_<condition>_<split>_n<max>.jsonl

Install:
  pip install -U transformers datasets accelerate torch click

Notes:
- Uses HF Transformers generation.
- Uses chat template if tokenizer supports it (recommended for Instruct models).
- Records both input token count and output token count.
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
# Condition configs (core knob)
# -----------------------------
# We pick a fixed target steps and token budget per condition to make
# length differences real and repeatable.
COND_CFG: Dict[str, Dict[str, int]] = {
    "direct": {"steps": 0, "max_new_tokens": 64},
    "short": {"steps": 2, "max_new_tokens": 128},
    "medium": {"steps": 4, "max_new_tokens": 256},
    "long": {"steps": 8, "max_new_tokens": 512},
}

# You can tweak these if your models are slower / faster.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0


# -----------------------------
# Prompt templates
# -----------------------------
SYSTEM_PROMPT = "You are a careful math solver. Follow formatting rules exactly."

USER_PROMPT_DIRECT = """Solve the following math problem.

Rules:
- Do NOT show any reasoning.
- Output exactly ONE line in the format:
  Final Answer: <number>
- Do not output anything else.

Problem:
{question}
"""

USER_PROMPT_COT = """Solve the following math problem.

Rules:
- Write EXACTLY {n_steps} reasoning steps.
- Each step MUST be a single line and MUST start with "Step k:" (k = 1..{n_steps}).
- Each step MUST be one short sentence (max 15 words).
- After the last step, output exactly ONE final line:
  Final Answer: <number>
- Do not output anything else.

Problem:
{question}
"""


def build_user_prompt(question: str, condition: str, n_steps: int) -> str:
    if condition == "direct":
        return USER_PROMPT_DIRECT.format(question=question)
    return USER_PROMPT_COT.format(question=question, n_steps=n_steps)


def format_chat_if_possible(tokenizer, system_prompt: str, user_prompt: str, use_chat_template: bool) -> str:
    """
    If tokenizer has apply_chat_template, we use it. Otherwise, fall back to plain text.
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


def estimate_steps(output_text: str) -> int:
    if not output_text:
        return 0
    return len(re.findall(r"\bStep\s+\d+\s*:", output_text))


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
    # Turn "Qwen/Qwen2.5-7B-Instruct" into "qwen2.5-7b-instruct"
    return model_name.split("/")[-1].lower().replace(" ", "_")


@click.command()
# Model list: Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B-Instruct, meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B-Instruct, meta-llama/Llama-3.2-7B-Instruct
@click.option("--model", "model_name", default="Qwen/Qwen2.5-1.5B-Instruct", show_default=True, type=str)
@click.option("--split", default="test", show_default=True, type=click.Choice(["train", "test"], case_sensitive=False))
@click.option("--condition", default="direct", show_default=True,
              type=click.Choice(["direct", "short", "medium", "long"], case_sensitive=False))
@click.option("--results-dir", default="results/gsm8k", show_default=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--max-samples", default=None, type=int, help="Run only first N examples (after shuffle).")
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--shuffle/--no-shuffle", default=True, show_default=True)
@click.option("--device", default="auto", show_default=True, type=str, help="auto|cuda|cpu|cuda:0 ...")
@click.option("--dtype", default="bfloat16", show_default=True,
              type=click.Choice(["float16", "bfloat16", "float32"], case_sensitive=False))
@click.option("--temperature", default=DEFAULT_TEMPERATURE, show_default=True, type=float)
@click.option("--top-p", default=DEFAULT_TOP_P, show_default=True, type=float)
@click.option("--use-chat-template/--no-chat-template", default=True, show_default=True)
def main(
    model_name: str,
    split: str,
    condition: str,
    results_dir: Path,
    max_samples: Optional[int],
    seed: int,
    shuffle: bool,
    device: str,
    dtype: str,
    temperature: float,
    top_p: float,
    use_chat_template: bool,
):
    condition = condition.lower().strip()
    if condition not in COND_CFG:
        raise click.ClickException(f"Unknown condition: {condition}")

    # Condition-specific "reasoning length" controls
    n_steps = COND_CFG[condition]["steps"]
    max_new_tokens = COND_CFG[condition]["max_new_tokens"]

    # Build default out path (no required flags)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_tag(model_name)}_{condition}_{split}"
    if max_samples is not None:
        out_path = out_path.with_name(out_path.name + f"_n{max_samples}")
    out_path = out_path.with_suffix(".jsonl")

    click.echo(f"[cfg] model={model_name}")
    click.echo(f"[cfg] split={split} condition={condition} steps={n_steps} max_new_tokens={max_new_tokens}")
    click.echo(f"[cfg] temperature={temperature} top_p={top_p} device={device} dtype={dtype}")
    click.echo(f"[out] {out_path}")

    # Load data & model
    rows = load_gsm8k(split)
    idxs = pick_indices(len(rows), max_samples, seed, shuffle)
    model, tok = load_model(model_name, device=device, dtype=dtype)

    run_cfg = RuntimeCfg(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        use_chat_template=use_chat_template,
    )

    # Loop
    t0 = time.time()
    correct_cnt = 0

    with out_path.open("w", encoding="utf-8") as f:
        for k, i in enumerate(idxs, start=1):
            q = rows[i]["question"]
            gold_raw = rows[i]["answer"]
            gold_num = extract_number(gold_raw)

            user_prompt = build_user_prompt(q, condition, n_steps=n_steps)
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
                "condition": condition,
                "steps_target": n_steps,
                "max_new_tokens": max_new_tokens,
                "question": q,
                "gold_raw": gold_raw,
                "gold_num": gold_num,
                "pred_raw": gen_text,
                "pred_num": pred_num,
                "correct": ok,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "steps_est": estimate_steps(gen_text),
                "latency_sec": latency,
                "seed": seed,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if k % 25 == 0 or k == len(idxs):
                acc = correct_cnt / k
                click.echo(f"[{k}/{len(idxs)}] acc={acc:.3f} last_pred={pred_num} gold={gold_num} out_tok={out_tok}")

    total = time.time() - t0
    click.echo(f"[done] n={len(idxs)} acc={correct_cnt/len(idxs):.4f} time={total:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()