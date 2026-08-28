# Archive

Analyses that are no longer part of the claim in
[`docs/methodology.md`](../docs/methodology.md). They are kept because they
were run and reported on at some point, and dropping them silently would
misrepresent what the project actually explored — including the paths that
went nowhere.

Nothing here is referenced by the methodology brief, the labeling pipeline,
or the tests. Nothing in `src/` imports from `archive/`; the dependency only
runs the other way.

They still run:

```bash
uv run python -m archive.trends --range full
```

| Module | What it did | Why it is here |
|---|---|---|
| `trends.py` | Google Trends interest for freedom terms, 2004–present | Search-interest data cannot speak to sense change, which is the claim under test |
| `economic_freedom_analysis.py` | Freedom's drift toward market/financial vocabulary, 1900–2010 | Subsumed by the positive/negative liberty framing |
| `financial_freedom_deep_dive.py` | Ngram deep dive on the phrase "financial freedom" | Single-phrase side quest |
| `tier2_analysis.py` | EEBO-TCP collocates and domain tagging, 1500–1700 | The EEBO tier is not part of the decade view |
| `tier2_fulltext_analysis.py` | Collocates from the EEBO full-text corpus | Same |
| `normalizer.py` | Early Modern English spelling normalization | Only ever used by the two EEBO scripts |
| `semantic_axis.py` | SemAxis projection onto a constraint→agency axis | Superseded by sentence-level labeling |
| `modern_embeddings.py` | Extends the axis to the 2000s with COHA | Depends on `semantic_axis` |
| `wiki_embeddings.py` | Procrustes-aligns GloVe 2014/2024 onto COHA | Depends on `semantic_axis` |
| `neighborhood_dynamics.py` | Neighborhood restructuring around freedom | Descriptive; no hypothesis attached |

`normalizer.normalize_final_e` is a no-op (`return text`) and has no callers.
It was already dead before the move; noted here rather than silently fixed.

Two things to know if you revive any of this:

- `semantic_axis.linear_trend_r2` reports a **per-year** slope with R-squared.
  It is not `src.stats.linear_trend`, which reports a mean-centred
  **per-century** slope with a standard error. The two were once both called
  `linear_trend`, which is exactly how they got confused.
- These modules import from `src.*` and are not covered by the test suite.
