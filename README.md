# Context Forge

An open-source benchmark and static Vercel report for comparing context compression strategies on real public data:

- RAG / grounded QA: SQuAD-style question, context, and answer examples.
- Agent traces: public Hugging Face agent trajectories.
- Long-context summarization: GovReport government reports and summaries.

The benchmark reports the Pareto frontier for tokens saved, quality retained, and compression latency per tokenizer/model. It also trains a small supervised classifier that predicts token droppability from real benchmark examples.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
compress-bench run --limit-per-task 2
compress-bench train-classifier --limit-per-task 4
compress-bench plots --out-dir outputs
python -m http.server 4173 -d public
```

Open `http://localhost:4173` to view the report.

## Full LLMLingua Run

LLMLingua is optional because it downloads a compression model. Install and include it explicitly:

```bash
python -m pip install -e ".[llmlingua]"
compress-bench run \
  --strategies llmlingua hard_prompt_pruning embedding_chunk_drop kv_cache_eviction \
  --limit-per-task 25
```

## Vercel

This repo is frameworkless static Vercel. After a benchmark run has written `public/results/latest_results.json`, deploy with:

```bash
vercel deploy --prod
```

The deployment contains no baked-in benchmark numbers; it renders the generated result artifact.

## Output Files

- `data/results/records.jsonl`: raw per-example measurements.
- `data/results/latest_results.json`: aggregate run metadata and Pareto frontier.
- `data/results/classifier_report.json`: classifier metrics.
- `data/results/droppable_classifier.pkl`: trained classifier.
- `outputs/pareto_*.png`: writeup-ready plots.

## Quality Metrics

The benchmark uses task-specific retention metrics so it can run without paid inference:

- QA: exact answer retention or answer-token recall.
- Summarization: target keyword and token coverage against the reference summary.
- Agent traces: final-output or tail-trace keyword retention.

These are compression-preservation metrics, not a replacement for human or model-based answer quality. The code is structured so LLM evaluators can be added later without changing the static report format.
