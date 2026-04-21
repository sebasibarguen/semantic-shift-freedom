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

## Running

Scripts are independent — most read from `data/` and write to `outputs/`.

```bash
# Embedding trajectory analysis
uv run python src/freedom_liberty_analysis.py

# LLM sentence classification (requires ANTHROPIC_API_KEY)
uv run python src/deep_semantic_analysis.py

# Extract sentences from Hansard
uv run python src/hansard_archive_extractor.py
uv run python src/hansard_sentence_extractor.py

# Robustness and control checks
uv run python src/robustness.py
uv run python src/control_words.py
```

LLM sentence classification uses the [Message Batches API](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — 50% cheaper and async. One request per sentence with forced tool-use output (rationale + label).

```bash
# Classify every sentence in a decade file (in place)
uv run python src/classify_liberty.py --input web/data/sentences_1980s.json

# Evaluate against the 100-sentence Opus gold sample
uv run python src/classify_liberty.py --eval
```

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
