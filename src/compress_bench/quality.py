from __future__ import annotations

import math
import string
from collections import Counter
from typing import Iterable, Set

import regex as re


_WORD_RE = re.compile(r"[\p{L}\p{N}]+")


def normalize_words(text: str) -> list:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def quality_retained(original_target: str, compressed_prompt: str, task: str) -> float:
    target_words = normalize_words(original_target)
    compressed_words = normalize_words(compressed_prompt)
    if not target_words:
        return 0.0

    if task == "rag_qa":
        target = " ".join(target_words)
        prompt = " ".join(compressed_words)
        exact = 1.0 if target and target in prompt else 0.0
        recall = _multiset_recall(target_words, compressed_words)
        return max(exact, recall)

    target_keywords = top_keywords(target_words, max_terms=80)
    if not target_keywords:
        return _multiset_recall(target_words, compressed_words)
    retained = len(target_keywords.intersection(set(compressed_words))) / len(target_keywords)
    coverage = _multiset_recall(target_words, compressed_words)
    return 0.7 * retained + 0.3 * coverage


def exact_target_retained(original_target: str, compressed_prompt: str) -> float:
    target = " ".join(normalize_words(original_target))
    prompt = " ".join(normalize_words(compressed_prompt))
    if not target:
        return 0.0
    return 1.0 if target in prompt else 0.0


def top_keywords(words: Iterable[str], max_terms: int = 64) -> Set[str]:
    stop = {
        "the", "and", "or", "of", "to", "in", "a", "an", "for", "on", "with", "by", "is",
        "are", "was", "were", "be", "been", "that", "this", "as", "from", "at", "it", "its",
        "which", "we", "their", "has", "have", "had", "not", "but", "can", "will", "would",
    }
    counts = Counter(w.strip(string.punctuation) for w in words)
    scored = []
    total = sum(counts.values()) or 1
    for word, count in counts.items():
        if len(word) < 4 or word in stop:
            continue
        score = count * math.log(1 + total / count)
        scored.append((score, word))
    return {word for _, word in sorted(scored, reverse=True)[:max_terms]}


def _multiset_recall(target_words: list, compressed_words: list) -> float:
    target_counts = Counter(target_words)
    compressed_counts = Counter(compressed_words)
    overlap = sum(min(count, compressed_counts[word]) for word, count in target_counts.items())
    return overlap / max(1, sum(target_counts.values()))
