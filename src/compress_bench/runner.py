from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm

from .data import load_task_examples, read_manifest
from .quality import exact_target_retained, quality_retained
from .schemas import EvalRecord, ParetoPoint
from .strategies import strategy_by_name
from .tokenizers import token_count


def run_benchmark(
    manifest_path: str,
    tasks: Iterable[str],
    models: Iterable[str],
    strategies: Iterable[str],
    ratios: Iterable[float],
    limit_per_task: int,
    out_dir: str,
    run_id: str = None,
    skip_missing_strategies: bool = True,
) -> dict:
    manifest = read_manifest(manifest_path)
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    examples = load_task_examples(manifest, tasks, limit_per_task)
    selected_strategies = strategy_by_name(strategies)
    records: List[EvalRecord] = []
    errors = []

    for model_id in models:
        for example in tqdm(examples, desc=f"benchmark:{model_id}"):
            original_tokens = token_count(example.prompt, model_id)
            if original_tokens == 0:
                continue
            for strategy in selected_strategies:
                for ratio in ratios:
                    try:
                        compressed = strategy.run(example, model_id, ratio)
                    except Exception as exc:
                        if skip_missing_strategies:
                            errors.append(
                                {
                                    "strategy": strategy.name,
                                    "model": model_id,
                                    "example_id": example.uid,
                                    "error": str(exc),
                                }
                            )
                            continue
                        raise
                    compressed_tokens = token_count(compressed.compressed_prompt, model_id)
                    quality = quality_retained(
                        example.target, compressed.compressed_prompt, example.task
                    )
                    exact = exact_target_retained(example.target, compressed.compressed_prompt)
                    records.append(
                        EvalRecord(
                            run_id=run_id,
                            task=example.task,
                            model=model_id,
                            strategy=compressed.strategy,
                            ratio=ratio,
                            example_id=example.uid,
                            original_tokens=original_tokens,
                            compressed_tokens=compressed_tokens,
                            tokens_saved_pct=max(
                                0.0, 100.0 * (original_tokens - compressed_tokens) / original_tokens
                            ),
                            quality_retained=quality,
                            latency_ms=compressed.latency_ms,
                            exact_target_retained=exact,
                        )
                    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "records.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    pareto = aggregate_pareto(records)
    summary = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_path": manifest_path,
        "n_records": len(records),
        "n_examples": len(examples),
        "models": list(models),
        "tasks": list(tasks),
        "ratios": list(ratios),
        "strategies_requested": list(strategies),
        "errors": errors[:50],
        "records_path": str(raw_path),
        "pareto": [asdict(point) for point in pareto],
    }
    latest_path = out / "latest_results.json"
    latest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def aggregate_pareto(records: List[EvalRecord]) -> List[ParetoPoint]:
    grouped = defaultdict(list)
    for record in records:
        key = (record.task, record.model, record.strategy, record.ratio)
        grouped[key].append(record)

    candidates = []
    for (task, model, strategy, ratio), rows in grouped.items():
        candidates.append(
            ParetoPoint(
                task=task,
                model=model,
                strategy=strategy,
                ratio=ratio,
                tokens_saved_pct=float(pd.Series([r.tokens_saved_pct for r in rows]).mean()),
                quality_retained=float(pd.Series([r.quality_retained for r in rows]).mean()),
                latency_ms=float(pd.Series([r.latency_ms for r in rows]).median()),
                n_examples=len(rows),
            )
        )

    frontier = []
    for point in candidates:
        dominated = False
        for other in candidates:
            same_slice = point.task == other.task and point.model == other.model
            if not same_slice or point == other:
                continue
            better_or_equal = (
                other.tokens_saved_pct >= point.tokens_saved_pct
                and other.quality_retained >= point.quality_retained
                and other.latency_ms <= point.latency_ms
            )
            strictly_better = (
                other.tokens_saved_pct > point.tokens_saved_pct
                or other.quality_retained > point.quality_retained
                or other.latency_ms < point.latency_ms
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda p: (p.task, p.model, -p.tokens_saved_pct))


def copy_latest_to_public(out_dir: str, public_dir: str = "public/results") -> None:
    os.makedirs(public_dir, exist_ok=True)
    src = Path(out_dir) / "latest_results.json"
    dst = Path(public_dir) / "latest_results.json"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
