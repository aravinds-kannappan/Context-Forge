<div align="center">

# ◆ Context Forge

### Spend fewer tokens. Keep the answer.

**An open, reproducible benchmark for prompt/context compression** — it reports the
**Pareto frontier** of *tokens saved* vs. *quality retained* vs. *latency* across the
tokenizers production apps actually pay for, on **real public data**.

[![Python](https://img.shields.io/badge/python-3.9%2B-2dd4bf)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-818cf8)](LICENSE)
[![Tokenizers](https://img.shields.io/badge/tokenizers-GPT--4o%20%7C%20GPT--4%20%7C%20GPT--2-f59e0b)](data/manifest.yml)
[![Deploy](https://img.shields.io/badge/deploy-Vercel%20static-f472b6)](vercel.json)
[![Data](https://img.shields.io/badge/data-100%25%20real%2C%20public-34d399)](docs/writeup.md)

![Pareto overview](docs/figures/overview.png)

</div>

---

## Why this exists

Every token you send to an LLM costs money and latency, and long contexts blow up
quadratically. **Context compression** trims the prompt before inference — but the
question that matters is the *tradeoff*: how much can you cut before the answer breaks,
and what does it cost you in compute?

Context Forge answers that question with **hard numbers**. It runs a fixed task set
through several compression strategies and multiple tokenizers, then plots the
**Pareto frontier** — the set of configurations where you can't get more savings
without losing quality (or paying more latency). Nothing is hard-coded; the static
report renders whatever the last benchmark run produced.

> **Live report:** the `public/` folder is a frameworkless static site (deploy to Vercel)
> that renders `public/results/latest_results.json` into an interactive Pareto explorer.

---

## Headline results

From the bundled run (`1,620` measurements · `36` examples · `3` tokenizers · `5` ratios):

| Strategy | Quality retained | Tokens saved | Latency (median) | One-liner |
|---|---:|---:|---:|---|
| **kv_cache_eviction** | **0.919** | 52.2% | 9.3 ms | prefix + recent suffix + salient middle |
| **embedding_chunk_drop** | **0.914** | 50.7% | 10.2 ms | drop least-relevant chunks (TF-IDF cosine) |
| hard_prompt_pruning | 0.709 | 49.1% | **2.7 ms** | head/tail token-budget truncation |

**Takeaway:** relevance-aware strategies (chunk drop / cache-eviction proxy) retain
**~92% of task signal at ~51% fewer tokens**, while naive head/tail truncation collapses
to **71%** — because it routinely cuts the very span that answers the question. For RAG QA
specifically, truncation drops quality to **0.0** at high compression (the answer span is
gone), whereas chunk-relevance dropping keeps it at **1.0**.

<div align="center">

![Strategy comparison](docs/figures/strategy_comparison.png)

</div>

---

## What's measured

**Three task families, all from real public datasets** (see [`data/manifest.yml`](data/manifest.yml)):

| Task | Dataset | Signal preserved |
|---|---|---|
| RAG / grounded QA | SQuAD (validation) | answer-span recall |
| Agent traces | `lambda/hermes-agent-reasoning-traces` | tail-trace / outcome coverage |
| Long-context summarization | GovReport | reference-keyword coverage |

**Three production tokenizers** — token savings are counted with the tokenizer a real
deployment pays for, not a stand-in:

| Model | Backend | Encoding |
|---|---|---|
| `gpt-4o` | tiktoken | `o200k_base` (GPT-4o / GPT-4.1) |
| `gpt-4` | tiktoken | `cl100k_base` (GPT-4 / GPT-3.5-turbo / `text-embedding-3`) |
| `gpt-2` | Hugging Face | BPE reference baseline |

Adding `gpt-4`/`gpt-4o` was the point of the multi-backend rewrite — the original
benchmark only counted GPT-2 tokens. Any tiktoken encoding or HF repo id also works
(see [`tokenizers.py`](src/compress_bench/tokenizers.py)).

**Four strategies:**

- `hard_prompt_pruning` — deterministic head/tail token-budget truncation (near-zero latency baseline).
- `embedding_chunk_drop` — TF-IDF/cosine chunk relevance dropping against the task query.
- `kv_cache_eviction` — keeps prompt prefix + recent suffix + salient middle (a text-level proxy for cache eviction).
- `llmlingua` — optional wrapper around Microsoft **LLMLingua-2** (extra dependency; downloads a model).

---

## Bonus: a learned token-droppability classifier

The seed of an actual *learned* compressor: a small model that predicts, **per token**,
whether it can be dropped. Trained on weak labels (tokens overlapping the target are
"keep", the rest "droppable") using **character n-gram + positional features**.

From the bundled run (1.6M labeled tokens, `gpt-4o` tokenizer):

| ROC-AUC | F1 (droppable) | Accuracy | Precision / Recall |
|---:|---:|---:|---:|
| **0.964** | 0.917 | 88.1% | 0.98 / 0.86 |

It learns intuitive signal — morphological fragments and inflections
(`-ating`, `-ated`, `-ative`, `izer`) are droppable, while salient content nouns
(`healthcare`, `claims`, `agency`, `relationships`) are kept.

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e .

# 1) Benchmark compression strategies across tokenizers
compress-bench run --limit-per-task 12

# 2) Train the token-droppability classifier
compress-bench train-classifier --limit-per-task 30

# 3) Render the figures for the writeup / README
compress-bench plots

# 4) View the interactive report locally
python -m http.server 4173 -d public   # → http://localhost:4173
```

Both `run` and `train-classifier` write their artifacts into `public/results/`
automatically, so the static site is ready to deploy immediately after.

### Optional: full LLMLingua-2 run

```bash
python -m pip install -e ".[llmlingua]"   # downloads a compression model
compress-bench run \
  --strategies llmlingua hard_prompt_pruning embedding_chunk_drop kv_cache_eviction \
  --limit-per-task 25
```

---

## Architecture

```
src/compress_bench/
├── cli.py          # `compress-bench {run,train-classifier,plots}`
├── data.py         # streams real datasets per data/manifest.yml (no synthetic data)
├── tokenizers.py   # unified tiktoken + Hugging Face backend (GPT-4o/GPT-4/GPT-2/…)
├── strategies.py   # the four compression strategies
├── quality.py      # task-specific retention metrics
├── runner.py       # benchmark loop + Pareto-frontier aggregation
├── classifier.py   # learned token-droppability model
├── plots.py        # dark-themed Pareto / comparison figures
└── schemas.py      # typed records

public/             # frameworkless static dashboard (HTML/CSS/vanilla JS canvas)
└── results/        # latest_results.json + classifier_report.json (what the site renders)

data/manifest.yml   # dataset ids, splits, tokenizers, ratios — the single source of truth
docs/writeup.md     # short methodology writeup
```

---

## Methodology notes

- **Pareto frontier** — within each `(task, tokenizer)` slice, a configuration is kept
  if no other configuration beats it on *all three* axes (≥ tokens saved, ≥ quality,
  ≤ latency). The dashboard rings frontier points and connects them.
- **Quality retained is a preservation proxy**, not a model-graded answer score
  (answer-span recall for QA, reference-keyword coverage for summarization, tail-trace
  coverage for agents). This is deliberate: the benchmark runs end-to-end with **no paid
  inference**. The code is structured so an LLM judge can be slotted in without changing
  the report format. See [`docs/writeup.md`](docs/writeup.md).
- **Reproducibility** — every dataset id, split, tokenizer, and ratio lives in
  `data/manifest.yml`. The deployed site contains no baked-in numbers; it renders the
  generated artifact.

---

## Output files

| File | Contents |
|---|---|
| `public/results/latest_results.json` | run metadata, per-config aggregates (with frontier flag), per-strategy rollups |
| `public/results/classifier_report.json` | classifier metrics + most informative tokens |
| `data/results/records.jsonl` | raw per-example measurements |
| `data/results/droppable_classifier.pkl` | the trained classifier |
| `outputs/*.png`, `docs/figures/*.png` | Pareto / comparison / overview figures |

---

## Roadmap

- [ ] Optional LLM-as-judge quality scoring (drop-in, behind a flag).
- [ ] More tokenizers (Llama-3, Mistral, Claude) and an `embeddings` strategy via `sentence-transformers`.
- [ ] Use the trained classifier as a fifth, *learned* compression strategy in the benchmark loop.
- [ ] Cost axis (USD per 1M tokens) alongside latency.

## License

MIT — see [LICENSE](LICENSE).
