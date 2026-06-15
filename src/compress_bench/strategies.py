from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import CompressionResult, TaskExample
from .tokenizers import token_count, truncate_to_ratio


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass
class Strategy:
    name: str
    run: Callable[[TaskExample, str, float], CompressionResult]


def available_strategies(include_llmlingua: bool = True) -> List[Strategy]:
    strategies = [
        Strategy("hard_prompt_pruning", hard_prompt_pruning),
        Strategy("embedding_chunk_drop", embedding_chunk_drop),
        Strategy("kv_cache_eviction", kv_cache_eviction_proxy),
    ]
    if include_llmlingua:
        strategies.insert(0, Strategy("llmlingua", llmlingua_compress))
    return strategies


def strategy_by_name(names: Iterable[str]) -> List[Strategy]:
    all_strategies = {strategy.name: strategy for strategy in available_strategies(True)}
    selected = []
    for name in names:
        if name not in all_strategies:
            raise ValueError(f"Unknown strategy {name}; choices: {sorted(all_strategies)}")
        selected.append(all_strategies[name])
    return selected


def hard_prompt_pruning(example: TaskExample, model_id: str, ratio: float) -> CompressionResult:
    start = time.perf_counter()
    compressed = truncate_to_ratio(example.prompt, model_id, ratio, mode="head_tail")
    return CompressionResult(
        strategy="hard_prompt_pruning",
        ratio=ratio,
        compressed_prompt=compressed,
        latency_ms=(time.perf_counter() - start) * 1000,
        metadata={"mode": "head_tail_token_budget"},
    )


def embedding_chunk_drop(example: TaskExample, model_id: str, ratio: float) -> CompressionResult:
    start = time.perf_counter()
    chunks = _chunk_text(example.prompt)
    if len(chunks) <= 2:
        compressed = truncate_to_ratio(example.prompt, model_id, ratio, mode="head_tail")
        method = "token_truncate_small_input"
    else:
        query = f"{example.metadata.get('question', '')} {example.target}".strip() or example.prompt[:2000]
        scored = _score_chunks_tfidf(chunks, query)
        compressed = _pack_scored_chunks(scored, model_id, example.prompt, ratio)
        method = "tfidf_cosine_chunk_embedding"
    return CompressionResult(
        strategy="embedding_chunk_drop",
        ratio=ratio,
        compressed_prompt=compressed,
        latency_ms=(time.perf_counter() - start) * 1000,
        metadata={"method": method},
    )


def kv_cache_eviction_proxy(example: TaskExample, model_id: str, ratio: float) -> CompressionResult:
    """Text-level proxy for cache eviction: keep prompt prefix, recent suffix, and salient middle chunks."""
    start = time.perf_counter()
    chunks = _chunk_text(example.prompt)
    if len(chunks) <= 3:
        compressed = truncate_to_ratio(example.prompt, model_id, ratio, mode="head_tail")
    else:
        prefix = chunks[: max(1, int(len(chunks) * 0.08))]
        suffix = chunks[-max(1, int(len(chunks) * 0.20)) :]
        middle = chunks[len(prefix) : len(chunks) - len(suffix)]
        scored_middle = _score_chunks_tfidf(middle, example.target or " ".join(suffix))
        ordered = [(10.0, i, text) for i, text in enumerate(prefix)]
        offset = len(prefix)
        ordered += [(score, i + offset, text) for score, i, text in scored_middle]
        tail_offset = len(chunks) - len(suffix)
        ordered += [(10.0, i + tail_offset, text) for i, text in enumerate(suffix)]
        compressed = _pack_scored_chunks(ordered, model_id, example.prompt, ratio)
    return CompressionResult(
        strategy="kv_cache_eviction",
        ratio=ratio,
        compressed_prompt=compressed,
        latency_ms=(time.perf_counter() - start) * 1000,
        metadata={"mode": "prefix_recent_salience_proxy"},
    )


def llmlingua_compress(example: TaskExample, model_id: str, ratio: float) -> CompressionResult:
    start = time.perf_counter()
    try:
        from llmlingua import PromptCompressor
    except Exception as exc:  # pragma: no cover - exercised in optional environments
        raise RuntimeError(
            "LLMLingua is not installed. Install with `pip install -e .[llmlingua]`."
        ) from exc

    target_token = max(1, int(token_count(example.prompt, model_id) * ratio))
    compressor = _get_llmlingua_compressor()
    output = compressor.compress_prompt(
        example.prompt,
        instruction="Preserve task instructions, entities, answers, tool calls, and conclusions.",
        question=example.metadata.get("question", ""),
        target_token=target_token,
    )
    compressed = output.get("compressed_prompt", "")
    return CompressionResult(
        strategy="llmlingua",
        ratio=ratio,
        compressed_prompt=compressed,
        latency_ms=(time.perf_counter() - start) * 1000,
        metadata={"target_token": str(target_token)},
    )


_LLMLINGUA_COMPRESSOR = None


def _get_llmlingua_compressor():
    global _LLMLINGUA_COMPRESSOR
    if _LLMLINGUA_COMPRESSOR is None:
        from llmlingua import PromptCompressor

        _LLMLINGUA_COMPRESSOR = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    return _LLMLINGUA_COMPRESSOR


def _chunk_text(text: str) -> List[str]:
    chunks = [chunk.strip() for chunk in _SENTENCE_RE.split(text) if chunk.strip()]
    if len(chunks) < 4 and len(text) > 1200:
        chunks = [text[i : i + 600].strip() for i in range(0, len(text), 600)]
    return chunks or [text]


def _score_chunks_tfidf(chunks: List[str], query: str) -> List[tuple]:
    corpus = chunks + [query]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    try:
        matrix = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(matrix[:-1], matrix[-1:]).reshape(-1)
    except ValueError:
        sims = np.ones(len(chunks))
    return [(float(score), i, chunk) for i, (score, chunk) in enumerate(zip(sims, chunks))]


def _pack_scored_chunks(scored: List[tuple], model_id: str, original_prompt: str, ratio: float) -> str:
    budget = max(1, int(token_count(original_prompt, model_id) * ratio))
    selected = []
    used = 0
    for _, idx, chunk in sorted(scored, key=lambda item: (-item[0], item[1])):
        count = token_count(chunk, model_id)
        if used + count <= budget:
            selected.append((idx, chunk))
            used += count
    if not selected:
        return truncate_to_ratio(original_prompt, model_id, ratio, mode="head_tail")
    return "\n".join(chunk for _, chunk in sorted(selected))
