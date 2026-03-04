#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_drop.py

DROP evaluation (minimal viable):
- Input: passage + question
- Output: MUST contain `Final Answer: <answer>`
- Metric: normalize + Exact Match (EM) only

We mirror GSM8K step-length setup:
  conditions: direct / short / medium / long
  - direct: no reasoning, only final answer line
  - others: require EXACTLY n_steps lines "Step k:" then final answer line
And we enforce separation via max_new_tokens budgets.

Output:
  results/drop/<model_tag>_<condition>_<split>_n<max>.jsonl

Deps:
  pip install -U transformers datasets accelerate torch click

Usage:

1️⃣ Qwen2.5-1.5B
direct
Mac/Linux:
python eval_drop.py \
--model Qwen/Qwen2.5-1.5B-Instruct \
--condition short \
--split validation \
--max-samples 200 \
--temperature 0 \
--device auto

Windows PowerShell (adjust use ` instead of \):
python eval_drop.py `
--model Qwen/Qwen2.5-1.5B-Instruct `
--condition direct `
--split validation `
--max-samples 200 `
--temperature 0 `
--device auto

short
python eval_drop.py `
--model Qwen/Qwen2.5-1.5B-Instruct `
--condition short `
--split validation `
--max-samples 200 `
--temperature 0 `
--device auto

2️⃣ Qwen2.5-7B-Instruct
--model Qwen/Qwen2.5-7B-Instruct
其它参数一样。

3️⃣ Llama-3.2-1B
--model meta-llama/Llama-3.2-1B-Instruct
4️⃣ Llama-3.2-3B
--model meta-llama/Llama-3.2-3B-Instruct
"""

from __future__ import annotations

import json
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from drop_scoring import extract_final_answer, em_over_ground_truths


# -----------------------------
# Condition configs (core knob)
# -----------------------------
COND_CFG: Dict[str, Dict[str, int]] = {
    "direct": {"steps": 0, "max_new_tokens": 64},
    "short":  {"steps": 2, "max_new_tokens": 128},
    "medium": {"steps": 4, "max_new_tokens": 256},
    "long":   {"steps": 8, "max_new_tokens": 512},
}

DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0


# -----------------------------
# Prompt templates
# -----------------------------
SYSTEM_PROMPT = "You are a careful reading comprehension solver. Follow formatting rules exactly."

USER_PROMPT_DIRECT = """Answer the question based ONLY on the given passage.

Rules:
- Do NOT show any reasoning.
- Output exactly ONE line in the format:
  Final Answer: <answer>
- <answer> should be a short span or a number, copied from or computed from the passage.
- Do not output anything else.

Passage:
{passage}

Question:
{question}
"""

USER_PROMPT_COT = """Answer the question based ONLY on the given passage.

Rules:
- Write EXACTLY {n_steps} reasoning steps.
- Each step MUST be a single line and MUST start with "Step k:" (k = 1..{n_steps}).
- Each step MUST be one short sentence (max 18 words).
- After the last step, output exactly ONE final line:
  Final Answer: <answer>
- <answer> should be a short span or a number, copied from or computed from the passage.
- Do not output anything else.

Passage:
{passage}

Question:
{question}
"""


def build_user_prompt(passage: str, question: str, condition: str, n_steps: int) -> str:
    if condition == "direct":
        return USER_PROMPT_DIRECT.format(passage=passage, question=question)
    return USER_PROMPT_COT.format(passage=passage, question=question, n_steps=n_steps)


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


def estimate_steps(output_text: str) -> int:
    if not output_text:
        return 0
    return len(re.findall(r"\bStep\s+\d+\s*:", output_text))


# -----------------------------
# Dataset loading (robust)
# -----------------------------
def _extract_gold_answers(ex: Dict[str, Any]) -> List[str]:
    """
    Robustly extract gold answers from HF DROP-like examples.
    Returns a list of acceptable gold answer strings.
    """
    def _nonempty_str_list(xs):
        return [str(x).strip() for x in xs if str(x).strip()]

    def _from_answer_dict(a: Dict[str, Any]) -> List[str]:
        out: List[str] = []

        # spans/text variants
        for k in ("spans", "text", "answers"):
            if k in a and isinstance(a[k], list):
                out.extend(_nonempty_str_list(a[k]))

        # number
        num = a.get("number")
        if isinstance(num, str) and num.strip():
            out.append(num.strip())

        # date
        date = a.get("date")
        if isinstance(date, dict):
            parts = []
            # choose a consistent order; month day year is common
            for k in ("month", "day", "year"):
                v = date.get(k)
                if v is not None and str(v).strip():
                    parts.append(str(v).strip())
            if parts:
                out.append(" ".join(parts))
        elif isinstance(date, str) and date.strip():
            out.append(date.strip())

        # dedupe while keeping order
        seen = set()
        deduped = []
        for x in out:
            if x not in seen:
                deduped.append(x)
                seen.add(x)
        return deduped

    # (0) Common: "answers" as dict (HF QA-style)
    if "answers" in ex and isinstance(ex["answers"], dict):
        out = _from_answer_dict(ex["answers"])
        if out:
            return out

    # (A) Some versions: "answers_spans"
    if "answers_spans" in ex:
        v = ex["answers_spans"]
        if isinstance(v, list):
            out = _nonempty_str_list(v)
            if out:
                return out
        if isinstance(v, dict):
            out = _from_answer_dict(v)
            if out:
                return out

    # (B) Some versions: "answer" as dict (DROP original style)
    if "answer" in ex and isinstance(ex["answer"], dict):
        out = _from_answer_dict(ex["answer"])
        if out:
            return out

    # (C) Already-prepared fields (your current run shows gold_answers exists)
    for key in ("gold_answers", "answer_text"):
        if key in ex:
            v = ex[key]
            if isinstance(v, list):
                out = _nonempty_str_list(v)
                if out:
                    return out
            if isinstance(v, str) and v.strip():
                return [v.strip()]

    # final fallback: if "answer" is string/list
    if "answer" in ex:
        v = ex["answer"]
        if isinstance(v, list):
            out = _nonempty_str_list(v)
            if out:
                return out
        if isinstance(v, str) and v.strip():
            return [v.strip()]

    return []


def load_drop(split: str) -> List[Dict[str, Any]]:
    """
    DROP doesn't have public test labels in the original leaderboard setting;
    HF commonly provides train/validation.
    """
    from datasets import load_dataset
    ds = load_dataset("drop", split=split)
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
    tag = model_name.split("/")[-1].lower()
    tag = tag.replace(" ", "_")
    tag = tag.replace(".", "_")
    return tag

# -----------------------------
# CLI
# -----------------------------
@click.command()
@click.option("--model", "model_name", default="Qwen/Qwen2.5-1.5B-Instruct", show_default=True, type=str)
@click.option("--split", default="validation", show_default=True,
              type=click.Choice(["train", "validation"], case_sensitive=False))
@click.option("--condition", default="direct", show_default=True,
              type=click.Choice(["direct", "short", "medium", "long"], case_sensitive=False))
@click.option("--results-dir", default="results/drop", show_default=True,
              type=click.Path(file_okay=False, path_type=Path))
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

    n_steps = COND_CFG[condition]["steps"]
    max_new_tokens = COND_CFG[condition]["max_new_tokens"]

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_tag(model_name)}_{condition}_{split}"
    if max_samples is not None:
        out_path = out_path.with_name(out_path.name + f"_n{max_samples}")
    out_path = out_path.with_suffix(".jsonl")

    click.echo(f"[cfg] model={model_name}")
    click.echo(f"[cfg] split={split} condition={condition} steps={n_steps} max_new_tokens={max_new_tokens}")
    click.echo(f"[cfg] temperature={temperature} top_p={top_p} device={device} dtype={dtype}")
    click.echo(f"[out] {out_path}")

    rows = load_drop(split)
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
            ex = rows[i]

            passage = ex.get("passage") or ex.get("context") or ""
            question = ex.get("question") or ""
            gold_answers = _extract_gold_answers(ex)

            user_prompt = build_user_prompt(passage, question, condition, n_steps=n_steps)
            prompt = format_chat_if_possible(tok, SYSTEM_PROMPT, user_prompt, use_chat_template)

            start = time.time()
            gen_text, in_tok, out_tok = generate(model, tok, prompt, run_cfg)
            steps_est = estimate_steps(gen_text)

            # retry once if step count is wrong
            if condition != "direct" and steps_est != n_steps:
                gen_text, in_tok, out_tok = generate(model, tok, prompt, run_cfg)
                steps_est = estimate_steps(gen_text)
            latency = time.time() - start

            pred_ans = extract_final_answer(gen_text) or ""
            ok = em_over_ground_truths(pred_ans, gold_answers)
            correct_cnt += int(ok)

            rec = {
                "dataset": "drop",
                "split": split,
                "qid": i,
                "model_name": model_name,
                "condition": condition,
                "steps_target": n_steps,
                "max_new_tokens": max_new_tokens,
                "passage": passage,
                "question": question,
                "gold_answers": gold_answers,
                "pred_raw": gen_text,
                "pred_answer": pred_ans,
                "em": bool(ok),
                "correct": bool(ok),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "steps_est": estimate_steps(gen_text),
                "latency_sec": latency,
                "seed": seed,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if k % 25 == 0 or k == len(idxs):
                acc = correct_cnt / k
                click.echo(f"[{k}/{len(idxs)}] em={acc:.3f} last_pred='{pred_ans}' gold0='{(gold_answers[0] if gold_answers else '')}' out_tok={out_tok}")

    total = time.time() - t0
    click.echo(f"[done] n={len(idxs)} em={correct_cnt/len(idxs):.4f} time={total:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()