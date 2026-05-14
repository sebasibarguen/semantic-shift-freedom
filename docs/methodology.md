# Methodology — measuring the semantic shift of "freedom" by decade

**Status:** internal brief for team review. Numbers in this document come from
the current outputs in `outputs/` (regenerable with `uv run python -m
src.liberty_trends`, `src.freedom_liberty_analysis`, `src.robustness`,
`src.negative_positive_embeddings`).

---

## 1. Claim under test

Across the 19th and 20th centuries, the share of **positive-liberty** uses of
*freedom* and *liberty* (capacity / enabled agency) rises relative to
**negative-liberty** uses (non-interference / protection against constraint).
This is a *proportion-trend* claim, not a flip claim — we never assert that
positive-liberty becomes the majority class, only that its share moves up over
time.

Two complementary measurement families are used:

| Family | Source | Signal |
|---|---|---|
| Sentence-level labels | Hansard 1803–2021, LLM-classified | Per-decade proportion `positive / (positive + negative)` |
| Word-embedding geometry | HistWords COHA 1800–1990 | Per-decade distance of *freedom* to legal/status vs personal/capacity concept clusters |

A third measurement (negative-vs-positive embedding cluster gap) was tried and
returned a **null** result. That null is reported in §5 — we are not hiding it.

---

## 2. Corpus and unit of analysis

The unit is the **decade**: a 10-year bucket of speeches (1800s = 1800–1809,
etc.).

**Hansard sentence corpus** (LLM method):

| Decade | Sentences | of which `positive_liberty` | `negative_liberty` | `ambiguous` | `other` |
|---:|---:|---:|---:|---:|---:|
| 1800s | 1,032 | 116 | 583 | 10 | 323 |
| 1810s | 1,843 | 146 | 1,346 | 40 | 311 |
| 1820s | 3,688 | 361 | 2,548 | 91 | 688 |
| 1830s | 6,565 | 746 | 4,001 | 189 | 1,629 |
| 1840s | 4,861 | 487 | 3,148 | 83 | 1,142 |
| 1850s | 6,713 | 819 | 4,178 | 139 | 1,577 |
| 1860s | 6,260 | 816 | 4,233 | 164 | 1,047 |
| 1870s | 4,166 | 478 | 2,997 | 86 | 605 |
| 1880s | 3,511 | 480 | 2,333 | 110 | 588 |
| 1890s | 3,469 | 639 | 2,208 | 96 | 526 |
| 1900s | 3,298 | 613 | 2,173 | 103 | 409 |
| 1910s | **829** | 202 | 521 | 47 | 59 |
| 1920s | 4,909 | 919 | 3,323 | 235 | 431 |
| 1930s | 7,024 | 1,250 | 4,886 | 409 | 479 |
| 1940s | 8,750 | 1,776 | 5,882 | 621 | 471 |
| 1950s | 8,960 | 2,160 | 5,735 | 628 | 437 |
| 1960s | 8,926 | 1,966 | 5,857 | 636 | 467 |
| 1970s | 11,905 | 2,864 | 7,650 | 945 | 446 |
| 1980s | 12,332 | 2,759 | 8,207 | 930 | 436 |
| 1990s | 10,214 | 2,704 | 6,506 | 522 | 482 |
| 2000s | 11,382 | 2,743 | 6,987 | 707 | 945 |
| 2010s | 12,652 | 2,645 | 8,391 | 654 | 962 |
| 2020s | **2,078** | 376 | 1,454 | 117 | 131 |
| **Total** | **145,367** | 28,065 | 95,147 | 7,562 | 14,591 |

Caveats already known:

- **1813 and 1816 have no recorded sentences**, and **1909–1918** is a 10-year
  gap caused by the handoff between two source corpora (Historic Hansard
  archive ends at 1908; ParlParse begins in 1919). The 1910s row above is
  effectively only ~1.5 years of records — treat that decade as
  underrepresented in every chart.
- **The 2020s** covers only 2020–April 2021 (~1.5 years) in the current pull.
  Same caution applies: a partial decade, weight accordingly.
- **All speech is UK House of Commons / House of Lords**. This is one
  prominent English-speaking polity, not English-language usage in general.
  Anything we claim about *English* needs that hedge; the strong claim is
  about *Anglophone parliamentary discourse*.

**HistWords embeddings** (embedding methods): COHA-trained vectors, 1800–1990
in 10-year steps. The embedding methods are independent of the Hansard corpus
— they read the COHA US-English book corpus instead — which is the point: we
want methodological triangulation across data sources, not just across
methods. COHA does not extend past 2000 in a way that aligns cleanly with the
older vectors, so the embedding methods stop at the 1990s.

---

## 3. Method 1 — Sentence-level positive/negative liberty proportions

**Pipeline.** Every Hansard sentence containing *freedom* or *liberty* is
classified by an LLM (Claude Haiku 4.5) into one of four labels: positive
liberty, negative liberty, ambiguous, other. Classification uses a single
structured tool call per sentence via the Anthropic Message Batches API; each
verdict carries a one-line rationale.

**Trend metric.** For each decade, the primary trend metric is

> *positive share* = positive_liberty / (positive_liberty + negative_liberty)

A sensitivity check uses `positive / (positive + negative + ambiguous)` —
direction the same, magnitude slightly smaller (§6).

| Decade | Positive share | 95% Wilson CI |
|---:|:---:|:---:|
| 1800s | 0.166 | [0.140, 0.195] |
| 1810s | 0.098 | [0.084, 0.114] |
| 1820s | 0.124 | [0.113, 0.137] |
| 1830s | 0.157 | [0.147, 0.168] |
| 1840s | 0.134 | [0.123, 0.145] |
| 1850s | 0.164 | [0.154, 0.174] |
| 1860s | 0.162 | [0.152, 0.172] |
| 1870s | 0.138 | [0.127, 0.149] |
| 1880s | 0.171 | [0.157, 0.185] |
| 1890s | 0.224 | [0.209, 0.240] |
| 1900s | 0.220 | [0.205, 0.236] |
| 1910s | 0.279 | [0.248, 0.313] |
| 1920s | 0.217 | [0.205, 0.229] |
| 1930s | 0.204 | [0.194, 0.214] |
| 1940s | 0.232 | [0.223, 0.241] |
| 1950s | 0.274 | [0.264, 0.284] |
| 1960s | 0.251 | [0.242, 0.261] |
| 1970s | 0.272 | [0.264, 0.281] |
| 1980s | 0.252 | [0.244, 0.260] |
| 1990s | 0.294 | [0.284, 0.303] |
| 2000s | 0.282 | [0.273, 0.291] |
| 2010s | 0.240 | [0.232, 0.248] |
| 2020s | 0.205 | [0.188, 0.225] |

**Trend test.** OLS on the per-decade proportions:

- slope = **+0.074 / century**, std-err = 0.009, z = 8.09 → p ≈ 0
- first-decade share (1800s) = 0.166, last-decade share (2020s) = 0.205
- endpoint-only change = +0.040 (z = 2.33, p ≈ 0.02)

The slope is the trustworthy number; the endpoint comparison shouldn't be
overweighted because the first and last decades both have small or partial
samples. The trend is highly significant under either reading. The
mid-19th-century plateau (1810s–1880s share oscillating around 0.13–0.17) and
the post-1890s break are both visible by inspection — we are not committed to
that being a smooth line.

**Validation against a 3-model LLM council.** Production labels are Haiku
v2. The classifier was calibrated against Claude Opus 4.7 alone earlier in
the project (agreement = 71%, n=100). Since then we have built a council
sample-and-vote pipeline that labels a strategic 5K-sentence subset using
Opus 4.7 + GPT-5.5 + Gemini 3.1 Preview in parallel; gold = 3/3 unanimous,
silver = 2/3, disputed = no majority (~2%). On the 239-sentence pilot the
gold rate was 74.5%, and Haiku v2 agreed with the *council* gold at **92.6%**
on the held-out dev split (n=27 — too small to publish, but the harness is
solid; the full 5K run will give a ~750-sentence dev split). Replacement of
"Opus-as-ground-truth" with "council-as-ground-truth" is the right call:
Opus has its own systematic biases that the 3-model triangulation surfaces,
and three of those bias directions are unlikely to align by accident.

**What the LLM labels are doing semantically.** Rationales are stored per
sentence (`methods.llm.rationale`). Sampling errors and reviewing rationales
is currently the cheapest QA loop; we should also run a human-annotated
adjudication pass on the disputed-cases bucket, per
[`docs/annotation_protocol.md`](annotation_protocol.md).

---

## 4. Method 2 — Freedom ↔ Liberty embedding divergence

**Pipeline.** For each decade in HistWords (1800–1990), compute the cosine
similarity of the *freedom* and *liberty* vectors. If *freedom* and *liberty*
have been moving toward different referents, similarity should drop.

| Decade | cos(freedom, liberty) | 95% bootstrap CI |
|---:|:---:|:---:|
| 1800 | 0.648 | [0.622, 0.653] |
| 1830 | 0.615 | [0.587, 0.619] |
| 1850 | 0.681 | [0.645, 0.682] |
| 1870 | 0.607 | [0.577, 0.613] |
| 1880 | 0.617 | [0.592, 0.623] |
| 1900 | 0.663 | [0.634, 0.667] |
| 1920 | 0.544 | [0.523, 0.552] |
| 1950 | 0.561 | [0.542, 0.568] |
| 1970 | 0.571 | [0.551, 0.578] |
| 1990 | 0.506 | [0.487, 0.515] |

The 1800 and 1990 confidence intervals do not overlap → the
divergence between the two words is statistically distinguishable from
sampling noise (`divergence_significant = True` in
`outputs/robustness.json`). Note the non-monotonic mid-century shape: there
is no clean ratchet, only a long-run downward drift.

---

## 5. Method 3 — Legal/status vs personal/capacity cluster gap

**Pipeline.** Build two small concept clusters:

- **Legal/status cluster** (8 words): *slavery, bondage, emancipation,
  rights, law, citizen, slave, servitude*
- **Personal/capacity cluster** (8 words): *choice, autonomy, independence,
  self, ability, power, individual, personal*

For each decade compute the mean cosine distance from *freedom* to each
cluster, and report the **gap** `personal_distance − legal_distance`. A
**positive** gap means legal-cluster is closer (freedom sits in
legal/status semantic space); a **negative** gap means personal-cluster is
closer (freedom has moved toward personal/capacity space).

| Decade | gap | direction |
|---:|---:|:---|
| 1800 | +0.056 | legal-status closer |
| 1810 | +0.058 | legal-status closer |
| 1820 | +0.061 | legal-status closer |
| 1830 | +0.061 | legal-status closer |
| 1840 | +0.065 | legal-status closer |
| 1850 | +0.071 | legal-status closer |
| 1860 | +0.027 | legal-status closer |
| 1870 | +0.027 | legal-status closer |
| 1880 | −0.002 | crossover decade |
| 1890 | −0.021 | personal-capacity closer |
| 1900 | −0.015 | personal-capacity closer |
| 1910 | −0.019 | personal-capacity closer |
| 1920 | −0.048 | personal-capacity closer |
| 1930 | −0.012 | personal-capacity closer |
| 1940 | −0.025 | personal-capacity closer |
| 1950 | −0.016 | personal-capacity closer |
| 1960 | −0.025 | personal-capacity closer |
| 1970 | −0.022 | personal-capacity closer |
| 1980 | +0.017 | (mild reversal) |
| 1990 | −0.007 | personal-capacity closer |

- Early-decade mean gap (1800s–1820s): **+0.058**
- Late-decade mean gap (1970s–1990s): **−0.004**
- Slope: **−0.050 / century**, z = −5.32
- Permutation test (1000 shuffles of gap values across decades): **p = 0**

This signal aligns with Method 1's direction but lives in an entirely
different data source and statistical apparatus. That cross-modal agreement
is the strongest single piece of evidence we have.

---

## 6. Method 4 (null result, kept honest) — negative vs. positive concept cluster gap

**Pipeline.** Same machinery as §5 but with different anchor clusters:

- **Negative-liberty cluster**: removal/absence words (*lack, escape,
  liberation, …*) plus constraint nouns (*slavery, tyranny, oppression, …*)
- **Positive-liberty cluster**: rights/entitlement words (*right,
  entitlement, claim, …*) plus capacity words (*ability, capacity, power, …*)
  plus action verbs (*choose, act, pursue, …*)

The metric is *positive_tilt = mean_distance_to_negative −
mean_distance_to_positive*. Positive value = *freedom* closer to positive
cluster.

| Window | mean positive_tilt |
|---|---:|
| Early (1800–1820) | +0.0196 |
| Late (1970–1990) | +0.0098 |

Trend slope = **+0.0003 / century**, z = 0.054, p ≈ **0.957** — **no trend
detected**.

This is on purpose included. Methods 1 and 3 agree that *freedom* moves
toward personal/capacity space; Method 4 says that when we replace the
personal/capacity cluster with one constructed directly from Berlinian
negative/positive vocabulary (*right, entitlement, ability, choose, …*) the
trend disappears. There are several possible interpretations:

1. The Berlinian vocabulary terms (*right, ability, choose*) are tightly
   bound to *freedom* in **every** decade, so there is no headroom to move.
2. The negative cluster contains *slavery, tyranny, oppression* — words
   which are oppositional to *freedom* throughout the period. The vector
   geometry encodes "what freedom is opposed to", and that doesn't change.
3. The cluster construction is too small or noisy.

We do not currently know which of (1)–(3) is right; this is a research
question worth chasing. The honest position is that *one of our three
embedding probes returns a null*, and the published claim must be the more
specific one ("freedom moves from legal/status toward personal/capacity")
not the more general one ("freedom becomes more positive in embedding
space").

---

## 7. Triangulation: where the methods agree and where they diverge

| Method | Data | Direction over period | Statistic |
|---|---|---|---|
| §3 LLM positive share | UK Hansard, 145K sentences | **rises** (0.166 → 0.205, slope +0.074/century) | z = 8.1, p ≈ 0 |
| §4 Freedom-liberty cos | COHA HistWords | **falls** (0.648 → 0.506) | endpoint CIs disjoint |
| §5 Legal–personal gap | COHA HistWords | **falls** (+0.058 → −0.004) | perm-p = 0 |
| §6 Pos/neg cluster | COHA HistWords | flat (+0.020 → +0.010) | p ≈ 0.96, **null** |

**Two signals point one way (sentence-level labels + legal→personal embedding
drift), one signal is consistent (freedom and liberty pulling apart), one
signal is null (negative/positive embedding clusters).** The legal→personal
finding is the embedding-side result we can defend; the negative/positive
embedding result is *not* what we publish. The publishable claim is
proportion-trend + legal→personal shift, with the freedom/liberty divergence
as a coherent third axis.

---

## 8. What we want feedback on (open methodological choices)

These are decisions where the team's read matters. Each is summarized in one
sentence + a "what we picked, and why we're not sure" note.

1. **Denominator for the proportion trend.** Primary metric is
   `positive / (positive + negative)`; sensitivity uses
   `positive / (positive + negative + ambiguous)`. Both are significant; the
   primary has tighter CIs. **Risk:** if the *ambiguous* class is doing real
   semantic work, excluding it shifts the metric's meaning. Should ambiguous
   be in the denominator?

2. **Decade vs. year vs. change-point.** We use 10-year buckets for
   readability and to match the HistWords decade-step. Switching to yearly
   bins would 10× the noise. Change-point detection (Bayesian or BIC) might
   identify a structural break — preliminary inspection suggests one around
   the 1890s. Worth a follow-up analysis?

3. **The two short decades (1910s, 2020s).** 1910s has ~1.5 years of
   sentences (Hansard handoff gap + WW1), 2020s has ~1.5 years (current
   pull through April 2021). Drop both from the trend test? Keep but
   weight by sample size in the OLS? Currently we keep both unweighted.

4. **Hansard ≠ "English".** Methods 1 ranges over UK Parliament only.
   Methods 2–4 range over COHA (US-English books). The triangulation works
   because the two sources disagree about almost everything except the
   one signal we care about. Is one sentence about that enough hedging?

5. **Lexicons in Methods 3 and 4** are 8-word lists picked by us. Wordlist
   sensitivity analysis (leave-one-out, lexicon swaps) is in
   `src/robustness.py` but only for Method 3. Should we run the same
   sensitivity sweep on Method 4's clusters before describing the result
   as a "null"?

6. **Validation of LLM labels at scale.** The 5K LLM-council pass is what
   we lean on. The pilot's 92.6% Haiku-vs-council agreement is on 27
   sentences — too small. The full 5K-sample council run will produce a
   real number, but the pipeline is built and ready. Should that block
   sharing the brief externally?

7. **What about *the modern era* (2000s–2020s)?** Method 1 covers it.
   Methods 2–4 don't (HistWords stops at 1990). We have a Wikipedia
   GloVe model trained for 2024 (`src/wiki_train.py`) that we could
   Procrustes-align to the 1990 COHA frame, extending Method 2–3
   trajectories. Worth doing for this paper?

---

## 9. Reproducibility

All numbers in this document come from these scripts and outputs:

```bash
# Sentence-level proportion trend (Method 1)
uv run python -m src.liberty_trends
#   → outputs/liberty_trends.json

# Freedom-liberty divergence (Method 2)
uv run python -m src.freedom_liberty_analysis
uv run python -m src.robustness  # bootstrap CIs for §4 and §5
#   → outputs/freedom_liberty_analysis.json
#   → outputs/robustness.json

# Negative/positive cluster (Method 4 — null result)
uv run python -m src.negative_positive_embeddings
#   → outputs/negative_positive_embeddings.json

# Corpus audit (the table in §2 is derived from this)
uv run python -m src.corpus_manifest
#   → outputs/corpus_manifest.json
```

Sentence-level labels and per-sentence rationales live in
`web/data/sentences_*.json` (one file per decade). The LLM council
infrastructure is in `src/council/`; the prompt-iteration harness for the
small classifier is in `src/prompt_arena/`. Tests cover the trend math,
splits, and metrics (`tests/test_liberty_trends.py`,
`tests/test_prompt_arena.py` — 12 unit tests, all green).
