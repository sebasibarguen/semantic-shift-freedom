# semantic-shift-freedom

Computational analysis of how the word "freedom" shifted meaning across 500 years of English text.

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
| EEBO (Early English Books Online) | [Text Creation Partnership](https://textcreationpartnership.org/) | `tier2_analysis.py`, `negative_positive_eebo.py` |
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

# Google Trends (2004-present or COVID-era)
uv run python -m src.trends --range full
uv run python -m src.trends --range 2020s

# Hansard / EEBO analysis scripts
uv run python -m src.hansard_analysis
uv run python -m src.negative_positive_eebo
```

Each script's `ABOUTME` comment at the top describes inputs and outputs.

## LLM sentence classification

Classifies Hansard sentences as positive/negative/ambiguous/other liberty using Claude Haiku. One request per sentence via the [Message Batches API](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) (50% cheaper, async) with forced tool-use output (rationale + label). Runs local — no cloud infra.

```bash
# Classify every sentence in a decade file (in place)
uv run python -m src.classify_liberty --input web/data/sentences_1980s.json

# Evaluate against the 100-sentence Opus comparison set
uv run python -m src.classify_liberty --eval
```

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

## License

MIT
