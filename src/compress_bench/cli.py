from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import shutil
from pathlib import Path

from .classifier import train_droppable_classifier
from .plots import plot_pareto
from .runner import copy_latest_to_public, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark real-data context compression.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    bench = sub.add_parser("run", help="Run compression benchmark")
    bench.add_argument("--manifest", default="data/manifest.yml")
    bench.add_argument("--tasks", nargs="+", default=["rag_qa", "agent_traces", "long_context_summarization"])
    bench.add_argument("--models", nargs="+", default=["gpt-4o", "gpt-4", "gpt-2"])
    bench.add_argument("--strategies", nargs="+", default=["hard_prompt_pruning", "embedding_chunk_drop", "kv_cache_eviction"])
    bench.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.35, 0.5, 0.65, 0.8])
    bench.add_argument("--limit-per-task", type=int, default=8)
    bench.add_argument("--out-dir", default="data/results")
    bench.add_argument("--no-public-copy", action="store_true")
    bench.add_argument("--fail-on-missing-strategy", action="store_true")

    train = sub.add_parser("train-classifier", help="Train token droppability classifier")
    train.add_argument("--manifest", default="data/manifest.yml")
    train.add_argument("--tasks", nargs="+", default=["rag_qa", "agent_traces", "long_context_summarization"])
    train.add_argument("--model", default="gpt-4o")
    train.add_argument("--limit-per-task", type=int, default=20)
    train.add_argument("--out-dir", default="data/results")
    train.add_argument("--no-public-copy", action="store_true")

    plots = sub.add_parser("plots", help="Generate PNG Pareto plots")
    plots.add_argument("--results", default="data/results/latest_results.json")
    plots.add_argument("--out-dir", default="outputs")

    args = parser.parse_args()
    if args.cmd == "run":
        summary = run_benchmark(
            manifest_path=args.manifest,
            tasks=args.tasks,
            models=args.models,
            strategies=args.strategies,
            ratios=args.ratios,
            limit_per_task=args.limit_per_task,
            out_dir=args.out_dir,
            skip_missing_strategies=not args.fail_on_missing_strategy,
        )
        if not args.no_public_copy:
            copy_latest_to_public(args.out_dir)
        print(json.dumps(summary, indent=2))
    elif args.cmd == "train-classifier":
        report = train_droppable_classifier(
            manifest_path=args.manifest,
            tasks=args.tasks,
            model_id=args.model,
            limit_per_task=args.limit_per_task,
            out_dir=args.out_dir,
        )
        if not args.no_public_copy:
            dst = Path("public/results/classifier_report.json")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(args.out_dir) / "classifier_report.json", dst)
        print(json.dumps(asdict(report), indent=2))
    elif args.cmd == "plots":
        paths = plot_pareto(args.results, args.out_dir)
        print(json.dumps({"plots": paths}, indent=2))


if __name__ == "__main__":
    main()
