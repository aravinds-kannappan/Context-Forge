from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TaskExample:
    uid: str
    task: str
    prompt: str
    context: str
    target: str
    metadata: Dict[str, str]


@dataclass(frozen=True)
class CompressionResult:
    strategy: str
    ratio: float
    compressed_prompt: str
    latency_ms: float
    metadata: Dict[str, str]


@dataclass(frozen=True)
class EvalRecord:
    run_id: str
    task: str
    model: str
    strategy: str
    ratio: float
    example_id: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved_pct: float
    quality_retained: float
    latency_ms: float
    exact_target_retained: Optional[float] = None


@dataclass(frozen=True)
class ParetoPoint:
    task: str
    model: str
    strategy: str
    ratio: float
    tokens_saved_pct: float
    quality_retained: float
    latency_ms: float
    n_examples: int


def dataclass_list_to_dicts(items: List[object]) -> List[dict]:
    return [item.__dict__ for item in items]
