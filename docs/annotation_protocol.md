# Human Annotation Protocol

This project now treats positive/negative liberty as a sentence-level label whose
proportions can change over time. The hypothesis test is about trends in label
shares, not an absolute historical switch from one meaning to another.

## Labels

Use the same four labels as the LLM classifier:

- `negative_liberty`: liberty as non-interference, protection against coercion, restraint, arbitrary power, detention, censorship, or infringement.
- `positive_liberty`: liberty as enabled capacity, self-government, opportunity, institutional empowerment, or practical ability to act.
- `ambiguous`: both readings are plausible, or the sentence lacks enough context to decide.
- `other`: procedural idioms, proper names, titles, metadata, or incidental uses that are not substantive liberty claims.

## Sampling

```bash
uv run python -m src.sample_annotation_set
```

This writes two files:

- `web/data/validation_set.json` — annotator-facing. Model labels (Haiku, SBERT,
  council) are **stripped**; only the sentence and the context the classifier
  itself sees (year/speaker/party) remain.
- `outputs/validation_answer_key.json` — scorer only. Holds the Haiku and council
  labels per id. **Do not share with annotators.**

The sample mixes a representative **random** draw (the only subset used to
recompute the trend) with the council's **silver** and **disputed** cases — the
hard sentences excluded from every accuracy number the project currently
reports. Sizes are configurable (`--n-random`, `--n-silver`, `--n-disputed`,
`--n-gold`); the draw is deterministic by seed.

## Annotation Workflow

1. Each annotator opens the **blind** browser tool and labels independently:
   `web/compare.html?blind=1&set=validation`. Blind mode hides all model
   outputs and other annotators' labels, so judgments stay independent. The
   `blind=1` link locks the toggle on.
2. When done, each annotator uses **Export JSON** to save their labels.
3. Score the exports:
   ```bash
   uv run python -m src.score_annotations \
       --answer-key outputs/validation_answer_key.json \
       --labels alice.json bob.json --names alice bob
   ```
4. Adjudicate genuine disagreements by discussion if a single human gold label
   is needed; the scorer otherwise drops strict ties from the consensus.

## Reliability Targets

- Report **Cohen's / Fleiss' kappa**, not raw agreement — one class dominates,
  so raw % overstates reliability. The scorer computes both.
- Inter-annotator kappa is the ceiling on any model score: if humans only agree
  at kappa X, no classifier can be trusted past it.
- Read `council_vs_human` by tier: it shows whether the LLM "gold" actually
  tracks people, and how far the silver/disputed cases fall.
- Treat low agreement as a measurement problem, not a model problem.
- Keep `ambiguous` available; forcing hard labels will inflate apparent trend strength.

## Analysis Rule

Primary trend metric:

`positive_liberty / (positive_liberty + negative_liberty)`

Sensitivity metric:

`positive_liberty / (positive_liberty + negative_liberty + ambiguous)`

The main claim is robust only if the direction of change is similar across the
primary and sensitivity metrics and stable after adjudication.
