# Context Forge: Short Writeup

## Goal

This repository benchmarks compression strategies over a fixed real-data task set: grounded QA, agent traces, and long-context summarization. The output is a per-model Pareto frontier over tokens saved, quality retained, and compression latency.

## Data

- RAG QA uses a public SQuAD-style extractive QA dataset with real questions, contexts, and answer spans.
- Agent traces use `lambda/hermes-agent-reasoning-traces`, a public Hugging Face dataset of multi-turn agent conversations and tool execution traces.
- Long-context summarization uses GovReport, a dataset of U.S. government reports and expert-written summaries.

No benchmark examples are synthetic or hardcoded. Dataset ids and splits live in `data/manifest.yml`.

## Compression Strategies

- `llmlingua`: wraps the open-source LLMLingua prompt compressor when the optional dependency is installed.
- `hard_prompt_pruning`: deterministic head/tail token budget pruning.
- `embedding_chunk_drop`: TF-IDF cosine chunk relevance dropping, using task query/target text as the relevance anchor.
- `kv_cache_eviction`: a text-level proxy for KV-cache eviction that preserves prompt prefix, recent suffix, and salient middle chunks.

## Evaluation

The benchmark measures compression latency directly and computes model-specific token savings with each model tokenizer. Quality retained is a preservation score: answer-span/token retention for QA, reference-summary keyword coverage for summarization, and final/tail trace coverage for agent traces.

The Pareto frontier removes points dominated by another point for the same task and model: at least as much token saving, at least as much quality, and no higher latency.

## Token Droppability Classifier

The classifier is trained on real examples loaded from the same benchmark manifest. Labels are weakly supervised from target retention: tokens that overlap target-critical words are labeled non-droppable; other tokens are labeled droppable. The baseline model is logistic regression over token position and lexical features, saved as `data/results/droppable_classifier.pkl`.

## Reproduce

```bash
python -m pip install -e ".[llmlingua]"
compress-bench run --strategies llmlingua hard_prompt_pruning embedding_chunk_drop kv_cache_eviction --limit-per-task 25
compress-bench train-classifier --limit-per-task 40
compress-bench plots --out-dir outputs
```

The Vercel app reads `public/results/latest_results.json` and renders the latest frontier.
