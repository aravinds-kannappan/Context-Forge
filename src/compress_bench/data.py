from __future__ import annotations

import json
from itertools import islice
from typing import Iterable, List

import yaml
from datasets import load_dataset

from .schemas import TaskExample


def read_manifest(path: str = "data/manifest.yml") -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _stream_dataset(source: str, split: str, config: str = None):
    if config:
        return load_dataset(source, config, split=split, streaming=True)
    return load_dataset(source, split=split, streaming=True)


def load_rag_qa(source: str, split: str, limit: int) -> List[TaskExample]:
    rows = _stream_dataset(source, split)
    examples = []
    for row in islice(rows, limit):
        answers = row.get("answers", {})
        answer_texts = answers.get("text") or []
        target = answer_texts[0] if answer_texts else ""
        question = _safe_text(row.get("question"))
        context = _safe_text(row.get("context"))
        prompt = (
            "Answer the question using only the context.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
        )
        examples.append(
            TaskExample(
                uid=_safe_text(row.get("id")),
                task="rag_qa",
                prompt=prompt,
                context=context,
                target=target,
                metadata={"title": _safe_text(row.get("title")), "question": question},
            )
        )
    return examples


def load_gov_report(source: str, split: str, limit: int) -> List[TaskExample]:
    rows = _stream_dataset(source, split)
    examples = []
    for i, row in enumerate(islice(rows, limit)):
        context = _safe_text(row.get("report") or row.get("document"))
        target = _safe_text(row.get("summary"))
        prompt = (
            "Summarize the following long government report while preserving "
            "the central claims, entities, and conclusions.\n\n"
            f"Report:\n{context}\n\nSummary:"
        )
        examples.append(
            TaskExample(
                uid=_safe_text(row.get("id") or f"govreport-{i}"),
                task="long_context_summarization",
                prompt=prompt,
                context=context,
                target=target,
                metadata={"source": source},
            )
        )
    return examples


def _flatten_agent_row(row: dict) -> str:
    preferred = [
        row.get("messages"),
        row.get("conversations"),
        row.get("conversation"),
        row.get("trajectory"),
        row.get("trace"),
    ]
    for value in preferred:
        if value:
            return _safe_text(value)
    return _safe_text(row)


def load_agent_traces(source: str, split: str, config: str, limit: int) -> List[TaskExample]:
    rows = _stream_dataset(source, split, config=config)
    examples = []
    for i, row in enumerate(islice(rows, limit)):
        trace = _flatten_agent_row(dict(row))
        prompt = (
            "Compress this agent trajectory for audit and replay. Keep user goals, "
            "tool calls, tool results, decisions, and final outcome.\n\n"
            f"Trace:\n{trace}\n\nCompressed trace:"
        )
        target_fields = []
        for key in ("final_answer", "answer", "output", "result"):
            if row.get(key):
                target_fields.append(_safe_text(row.get(key)))
        target = "\n".join(target_fields) if target_fields else trace[-4000:]
        examples.append(
            TaskExample(
                uid=_safe_text(row.get("id") or row.get("trace_id") or f"agent-{i}"),
                task="agent_traces",
                prompt=prompt,
                context=trace,
                target=target,
                metadata={"source": source, "config": config or ""},
            )
        )
    return examples


def load_task_examples(manifest: dict, tasks: Iterable[str], limit_per_task: int) -> List[TaskExample]:
    examples: List[TaskExample] = []
    dataset_cfg = manifest["datasets"]
    for task in tasks:
        cfg = dataset_cfg[task]
        loader = cfg["loader"]
        if loader == "squad":
            examples.extend(load_rag_qa(cfg["source"], cfg["split"], limit_per_task))
        elif loader == "gov_report":
            examples.extend(load_gov_report(cfg["source"], cfg["split"], limit_per_task))
        elif loader == "hermes_agent_traces":
            examples.extend(
                load_agent_traces(
                    cfg["source"], cfg["split"], cfg.get("config"), limit_per_task
                )
            )
        else:
            raise ValueError(f"Unknown loader: {loader}")
    return examples
