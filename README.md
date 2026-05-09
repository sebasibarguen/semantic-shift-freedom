# semantic-shift-freedom

Computational analysis of how the word "freedom" shifted meaning across 500 years of English text.

Current hypothesis test: whether the proportion of positive-liberty versus
negative-liberty uses changes over time. The project no longer treats surface
preposition grammar as an active measurement method.

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/sebasibarguen/semantic-shift-freedom
cd semantic-shift-freedom
uv sync
cp .env.example .env  # fill in API keys
```

## Data

The `data/` directory is not included (too large). Download each dataset and place it under `data/`. Each script has an `ABOUTME` comment at the top describing the expected path structure.

| Dataset | Source | Scripts |
|---------|--------|---------|
| HistWords (COHA + Google Books) | [Stanford NLP](https://nlp.stanford.edu/projects/histwords/) | `embeddings.py`, `freedom_liberty_analysis.py`, `modern_embeddings.py` |
| EEBO (Early English Books Online) | [Text Creation Partnership](https://textcreationpartnership.org/) | `tier2_analysis.py`, `tier2_fulltext_analysis.py` |
| Hansard Parliamentary Debates | [Historic Hansard API](https://api.parliament.uk/historic-hansard/), [parlparse](https://github.com/mysociety/parlparse) | `hansard_*.py`, `parlparse_extractor.py` |
| Wikipedia dump | [Wikimedia Downloads](https://dumps.wikimedia.org/) | `wiki_*.py` |

## Running analyses

`src/` is a Python package — run scripts as modules:

```bash
# Embedding trajectory analysis
uv run python -m src.freedom_liberty_analysis

# Robustness and control checks
uv run python -m src.robustness
uv run python -m src.control_words

# Sentence-label proportion trends and corpus coverage audit
uv run python -m src.liberty_trends
uv run python -m src.corpus_manifest

# Google Trends (2004-present or COVID-era)
uv run python -m src.trends --range full
uv run python -m src.trends --range 2020s

# Hansard / EEBO analysis scripts
uv run python -m src.hansard_analysis
uv run python -m src.tier2_fulltext_analysis
```

Each script's `ABOUTME` comment at the top describes inputs and outputs.

## LLM sentence classification

Classifies Hansard sentences as positive/negative/ambiguous/other liberty using Claude Haiku. One request per sentence via the [Message Batches API](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) (50% cheaper, async) with forced tool-use output (rationale + label). Runs local — no cloud infra.

```bash
# Classify every sentence in a decade file (in place)
uv run python -m src.classify_liberty --input web/data/sentences_1980s.json

# Evaluate against the 100-sentence Opus comparison set
uv run python -m src.classify_liberty --eval

# Create a deterministic stratified sample for human annotation
uv run python -m src.sample_annotation_set --per-bucket 6
```

See `docs/annotation_protocol.md` for the human validation workflow. LLM labels
are useful for large-scale trend exploration, but publication-grade claims
should be checked against adjudicated human labels.

## LLM council (gold-standard labeling)

A 3-model council — Claude Opus 4.7, GPT-5.5, Gemini 3.1 Preview — labels a
strategic 5K sentence sample. Where all three agree → gold; where two agree →
silver; no-majority cases are kept aside (`disputed.json`) for human review.
Each verdict carries a one-line rationale and a self-reported confidence.

```bash
# 1. Build the strategic sample (deterministic seed, ~5K sentences)
uv run python -m src.classify_council --build-sample

# 2. Submit batches to all three providers and adjudicate
uv run python -m src.classify_council --run

# Or both at once for a 239-sentence pilot to validate the pipeline
uv run python -m src.classify_council --pilot

# Resume after a crash without re-submitting batches
uv run python -m src.classify_council --collect [--pilot-dir]
```

Outputs land in `outputs/council/{full,pilot}/`:
- `labels_{claude,gpt,gemini}.json` — raw per-provider verdicts
- `gold.json` — 3/3 agreement, all rationales preserved
- `silver.json` — 2/3 agreement
- `disputed.json` — kept aside for human review
- `summary.json` — counts and label distribution

Sample is built with `src/council/sample.py`: stratified by decade × Haiku
label, oversampled on method-disagreement and "freedom of X" patterns,
anchored on the existing 100-sentence Opus comparison set.

## Prompt arena (hill-climb the small classifier)

Iterate on the production-classifier prompt against the council gold set.
Train/dev/test splits are deterministic by sentence id (70/15/15) — the
test split is locked, only touched at the end.

```bash
# See split sizes and gold availability
uv run python -m src.iterate_prompt status [--pilot-gold]

# Evaluate a candidate prompt on dev
uv run python -m src.iterate_prompt eval \
    --prompt prompts/haiku_v3.txt --split dev

# Show all logged evaluations sorted by accuracy
uv run python -m src.iterate_prompt history --split dev
```

Each eval logs to `outputs/prompts/history.jsonl` with metrics + sample
errors. Prompt files live in `prompts/`. The arena uses caching by
`prompt_hash × split × model`, so re-evaluating an unchanged prompt is free.

The metric is **agreement with council gold**, not Opus alone — Opus carries
its own biases that the 3-model council triangulates against.

## Heavy jobs on Modal

Three jobs benefit from cloud RAM/CPU/time: Wikipedia GloVe training (multi-hour, 32 GB RAM) and full Hansard XML parsing (GB-scale). Consolidated into `src/modal_jobs.py`, which imports logic from the local `src.*` modules — the cloud runner is the only thing Modal-specific.

```bash
# Train word2vec on a Wikipedia dump
modal run src/modal_jobs.py --job wiki \
    --dump-url https://dumps.wikimedia.org/enwiki/20250101/enwiki-20250101-pages-articles.xml.bz2 \
    --label 2024

# Parse Historic Hansard XML (1803-1918)
modal run src/modal_jobs.py --job hansard-archive

# Parse ParlParse XML (1919-2025)
modal run src/modal_jobs.py --job parlparse

# Classify every freedom/liberty sentence from hansard-speeches.csv
modal run src/modal_jobs.py --job hansard-sentences
```

Upload source data to the `freedom-jobs` Modal volume first with `modal volume put`.

## Web Interface

Interactive results explorer. Runs on Vercel; `web/data/` contains pre-generated sentence JSON files loaded client-side.

```bash
cd web
npm install
npx vercel dev
```

`R2_*` env vars enable label persistence via Cloudflare R2. The site works read-only without them.

## Future work

- **Fine-tune a small classifier** (ModernBERT-base / DeBERTa-v3-base) on the
  council gold set. Once the gold has settled and the prompt arena hits a
  ceiling, training a small classifier gives near-zero inference cost on the
  full corpus and removes API dependency. Estimated training cost: ~$5-15.
  Will live under `src/finetune/`.
- **LLM-driven prompt proposer** for the arena: feed the worst dev errors plus
  council rationales to Opus and have it propose refined prompt variants.
  Automated hill-climbing on top of the manual harness.

## License

MIT
