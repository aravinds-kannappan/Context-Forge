"""Unified tokenizer layer.

Context Forge measures *tokens saved* under the tokenizer a real deployment
would actually pay for. That means production tokenizers, not just GPT-2:

- ``gpt-4o``        -> tiktoken ``o200k_base`` (GPT-4o / GPT-4.1 family)
- ``gpt-4``         -> tiktoken ``cl100k_base`` (GPT-4 / GPT-3.5-turbo / text-embedding-3)
- ``gpt-2``         -> Hugging Face ``gpt2`` (BPE reference baseline)
- ``flan-t5``       -> Hugging Face ``google/flan-t5-small`` (SentencePiece)
- ``mistral``       -> Hugging Face ``mistralai/Mistral-7B-v0.1`` if available

Both backends are exposed through one interface (``token_count``,
``truncate_to_ratio``, ``split_token_strings``) so the rest of the benchmark
never has to know whether a model is tiktoken- or transformers-backed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional


@dataclass
class TokenizedText:
    ids: List[int]
    tokens: List[str]


@dataclass(frozen=True)
class ModelSpec:
    """A friendly model name mapped to a concrete tokenizer backend."""

    name: str
    backend: str  # "tiktoken" | "huggingface"
    ref: str  # tiktoken encoding name or HF repo id
    family: str  # for grouping / display


# Registry of recognizable production tokenizers. Order matters for defaults.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "gpt-4o": ModelSpec("gpt-4o", "tiktoken", "o200k_base", "openai-o200k"),
    "gpt-4": ModelSpec("gpt-4", "tiktoken", "cl100k_base", "openai-cl100k"),
    "gpt-2": ModelSpec("gpt-2", "huggingface", "gpt2", "bpe"),
    "flan-t5": ModelSpec("flan-t5", "huggingface", "google/flan-t5-small", "sentencepiece"),
    "mistral": ModelSpec("mistral", "huggingface", "mistralai/Mistral-7B-v0.1", "sentencepiece"),
}

DEFAULT_MODELS = ["gpt-4o", "gpt-4", "gpt-2"]


def resolve_spec(model_id: str) -> ModelSpec:
    """Resolve a friendly name or a raw backend ref to a ModelSpec."""
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id]
    # Allow passing a raw tiktoken encoding name.
    if model_id in {"o200k_base", "cl100k_base", "p50k_base", "r50k_base"}:
        return ModelSpec(model_id, "tiktoken", model_id, "tiktoken")
    # Otherwise treat it as a Hugging Face repo id.
    return ModelSpec(model_id, "huggingface", model_id, "huggingface")


# --------------------------------------------------------------------------- #
# Backend loaders
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=32)
def _load_backend(model_id: str):
    spec = resolve_spec(model_id)
    if spec.backend == "tiktoken":
        import tiktoken

        return ("tiktoken", tiktoken.get_encoding(spec.ref))
    from transformers import AutoTokenizer

    return ("huggingface", AutoTokenizer.from_pretrained(spec.ref, use_fast=True))


def _encode(model_id: str, text: str) -> List[int]:
    backend, enc = _load_backend(model_id)
    if backend == "tiktoken":
        return enc.encode(text, disallowed_special=())
    return enc.encode(text, add_special_tokens=False)


def _decode(model_id: str, ids: List[int]) -> str:
    backend, enc = _load_backend(model_id)
    if backend == "tiktoken":
        return enc.decode(ids)
    return enc.decode(ids, skip_special_tokens=True)


def _id_to_string(model_id: str, token_id: int) -> str:
    backend, enc = _load_backend(model_id)
    if backend == "tiktoken":
        return enc.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")
    return enc.convert_ids_to_tokens(token_id)


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #
def token_count(text: str, model_id: str) -> int:
    if not text:
        return 0
    return len(_encode(model_id, text))


def truncate_to_ratio(text: str, model_id: str, ratio: float, mode: str = "head_tail") -> str:
    ids = _encode(model_id, text)
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
    return _decode(model_id, kept)


def split_token_strings(text: str, model_id: str) -> TokenizedText:
    ids = _encode(model_id, text)
    tokens = [_id_to_string(model_id, tid) for tid in ids]
    return TokenizedText(ids=ids, tokens=tokens)


def model_families(model_ids) -> Dict[str, str]:
    return {m: resolve_spec(m).family for m in model_ids}
