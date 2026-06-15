from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List

from transformers import AutoTokenizer


@dataclass
class TokenizedText:
    ids: List[int]
    tokens: List[str]


@lru_cache(maxsize=16)
def get_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


def token_count(text: str, model_id: str) -> int:
    tokenizer = get_tokenizer(model_id)
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_to_ratio(text: str, model_id: str, ratio: float, mode: str = "head_tail") -> str:
    tokenizer = get_tokenizer(model_id)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return text
    budget = max(1, int(len(ids) * ratio))
    if budget >= len(ids):
        return text
    if mode == "head":
        kept = ids[:budget]
    elif mode == "tail":
        kept = ids[-budget:]
    elif mode == "head_tail":
        head = max(1, int(budget * 0.35))
        tail = budget - head
        kept = ids[:head] + ids[-tail:]
    else:
        raise ValueError(f"Unknown truncation mode: {mode}")
    return tokenizer.decode(kept, skip_special_tokens=True)


def split_token_strings(text: str, model_id: str) -> TokenizedText:
    tokenizer = get_tokenizer(model_id)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return TokenizedText(ids=ids, tokens=tokenizer.convert_ids_to_tokens(ids))
