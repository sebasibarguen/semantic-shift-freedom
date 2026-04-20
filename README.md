# Semantic Shift of "Freedom"

A computational study tracking how the word *freedom* shifted between **negative liberty** ("freedom from") and **positive liberty** ("freedom to") across 500 years of English text.

## Key Finding

Freedom hasn't evolved linearly — it has swung back and forth:

| Period | "Freedom FROM" | "Freedom TO" |
|--------|----------------|--------------|
| 1500–1700 | 21% | **79%** |
| 1870–1920 | **76%** | 24% |
| 2010s | 35% | **65%** |

The abolition era (1830–1920) drove a near-complete inversion. Today's "positive freedom" language (financial freedom, medical freedom) increasingly implies trade-offs with collective welfare — which explains why freedom has become such a contested word.

Full write-up: [`paper/simple.md`](paper/simple.md)

## Methods

Three interlocking approaches:

1. **Word embeddings** — HistWords temporal embeddings (COHA + Google Books, 1800–2000s) track semantic neighborhood changes across decades. Extended to 2020s with Wikipedia-trained GloVe vectors.
2. **Corpus analysis** — Full-text analysis of EEBO (1500–1700), Hansard parliamentary debates (1803–2000s), and modern web corpora. Extracts and classifies ~500K sentences containing "freedom" or "liberty."
3. **LLM classification** — Claude (Opus/Sonnet/Haiku) and Gemini classify sentences as negative/positive liberty with reasoning. Robustness checks compare models and validate against human judgments.

## Repository Structure

```
src/                        # Analysis scripts (Python 3.12)
  embeddings.py             # HistWords temporal embedding loader
  metrics.py                # Cosine similarity and vector utilities
  normalizer.py             # Early modern text normalization
  freedom_liberty_analysis.py  # Core trajectory analysis
  semantic_axis.py          # SemAxis projection (negative/positive liberty axis)
  negative_positive_*.py    # Liberty classification (embeddings, ngrams, EEBO)
  deep_semantic_analysis.py # LLM-based sentence classification
  hansard_*.py              # UK parliamentary debate extraction and analysis
  parlparse_extractor.py    # Hansard XML parser (1803–1918)
  wiki_*.py                 # Wikipedia GloVe training (2020s extension)
  control_words.py          # Sanity checks with neutral terms
  corpus_validation.py      # Corpus integrity checks
  robustness.py             # Cross-model robustness analysis
  neighborhood_dynamics.py  # Semantic neighborhood shift over time
  economic_freedom_analysis.py  # Domain analysis: economic freedom
  financial_freedom_deep_dive.py
  trends_full_history.py    # Google Trends 2004–2025
  tier2_analysis.py         # EEBO full-text period analysis
  modal_*.py                # Modal cloud functions (optional, for large jobs)

outputs/                    # Results from analysis runs (JSON + PNG)
paper/                      # Write-up and research proposal
  simple.md                 # Main findings (non-technical)
  proposal.md               # Full research proposal
web/                        # Interactive results explorer (Vercel)
  index.html                # Main dashboard
  data/                     # Sentence data by decade (loaded client-side)
  api/labels.js             # Serverless label persistence (Cloudflare R2)
```

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/sebasibarguen/semantic-shift-freedom
cd semantic-shift-freedom
uv sync
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

## Data Sources

The `data/` directory is excluded from the repo (too large). Download from:

| Dataset | Source | Used by |
|---------|--------|---------|
| HistWords (COHA) | [Stanford NLP](https://nlp.stanford.edu/projects/histwords/) | `embeddings.py`, `freedom_liberty_analysis.py` |
| HistWords (Google Books) | [Stanford NLP](https://nlp.stanford.edu/projects/histwords/) | `modern_embeddings.py` |
| EEBO (Early English Books Online) | [Text Creation Partnership](https://textcreationpartnership.org/) | `tier2_analysis.py`, `negative_positive_eebo.py` |
| Hansard Parliamentary Debates | [Historic Hansard](https://api.parliament.uk/historic-hansard/), [parlparse](https://github.com/mysociety/parlparse) | `hansard_*.py`, `parlparse_extractor.py` |
| Wikipedia dump | [Wikimedia](https://dumps.wikimedia.org/) | `wiki_*.py` (2020s extension) |

Place data in `data/` following the structure expected by each script (see `ABOUTME` comments at top of each file).

## Running the Analysis

Scripts are designed to be run independently. Most read from `data/` and write to `outputs/`.

```bash
# Core embedding analysis
uv run python src/freedom_liberty_analysis.py

# LLM classification (requires ANTHROPIC_API_KEY)
uv run python src/deep_semantic_analysis.py

# Hansard extraction
uv run python src/hansard_archive_extractor.py
uv run python src/hansard_sentence_extractor.py

# Robustness checks
uv run python src/robustness.py
uv run python src/control_words.py
```

For large jobs (Wikipedia training, full Hansard processing), the `modal_*.py` scripts run on [Modal](https://modal.com/) cloud infrastructure.

## Web Interface

The interactive results explorer is deployed to Vercel. To run locally:

```bash
cd web
npm install
npx vercel dev
```

Set `R2_*` environment variables for label persistence (optional — the site works read-only without them).

## License

MIT
