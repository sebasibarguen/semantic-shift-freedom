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

Generate a stratified sample from the sentence corpus:

```bash
uv run python -m src.sample_annotation_set --per-bucket 6
```

The sampler selects up to `N` records per decade and current LLM-label bucket.
This intentionally oversamples rare classes for validation rather than matching
the corpus distribution.

## Annotation Workflow

1. Two annotators independently fill `annotator_1` and `annotator_2`.
2. Disagreements are adjudicated into `adjudicated_label`.
3. Use `notes` for edge cases or guideline refinements.
4. Report agreement before adjudication and model performance against adjudicated labels.

## Reliability Targets

- Report raw agreement and per-label confusion matrices.
- Report Krippendorff's alpha or Cohen's kappa before adjudication.
- Treat low agreement as a measurement problem, not a model problem.
- Keep `ambiguous` available; forcing hard labels will inflate apparent trend strength.

## Analysis Rule

Primary trend metric:

`positive_liberty / (positive_liberty + negative_liberty)`

Sensitivity metric:

`positive_liberty / (positive_liberty + negative_liberty + ambiguous)`

The main claim is robust only if the direction of change is similar across the
primary and sensitivity metrics and stable after adjudication.
