# drop_scoring.py
# Minimal DROP scoring: normalize + Exact Match (EM)
from __future__ import annotations

import re
import string
from typing import List, Optional


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans({p: " " for p in string.punctuation})


def _strip_quotes(s: str) -> str:
    s = s.strip()
    # common quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def normalize_answer(text: str) -> str:
    if text is None:
        return ""

    text = _strip_quotes(text)
    text = text.lower()

    # ---------- 先处理数字 ----------
    # 如果是纯数字或小数，直接标准化
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", text.strip()):
        x = float(text)
        if x.is_integer():
            return str(int(x))
        return str(x).rstrip("0").rstrip(".")

    # ---------- 再去标点 ----------
    text = text.translate(_PUNCT_TABLE)

    # remove articles
    text = _ARTICLES_RE.sub(" ", text)

    # collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def em_over_ground_truths(pred: str, golds: List[str]) -> bool:
    """
    EM is True if pred matches ANY gold answer after normalization.
    """
    if pred is None:
        pred = ""
    if not golds:
        return False
    p = normalize_answer(pred)
    for g in golds:
        if p == normalize_answer(g):
            return True
    return False


def clean_final_answer_span(ans: str) -> str:
    """
    Post-process model extracted answer:
      - take first line
      - strip trailing spaces
    """
    if ans is None:
        return ""
    ans = ans.strip()
    # use only the first line (DROP answers should be short)
    ans = ans.splitlines()[0].strip()
    return ans


def extract_final_answer(text: str) -> Optional[str]:
    """
    Prefer 'Final Answer:' line. If missing, fallback to last non-empty line.
    Works for text answers (not only numbers).
    """
    if not text:
        return None

    # capture content after 'Final Answer:' up to line end
    m = re.search(r"Final Answer\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return clean_final_answer_span(m.group(1))

    # fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    return clean_final_answer_span(lines[-1])