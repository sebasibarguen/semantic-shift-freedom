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

## What Blind Mode Hides

The annotator link is
`https://freedom-semantic-shift.vercel.app/compare.html?blind=1&set=validation`
(`blind=1` locks the toggle on). It hides:

- model outputs (Haiku, SBERT, council) and the LLM/agreement filters,
- other annotators' labels,
- the **date, speaker, and party** on each card, plus the decade and party
  filters that would otherwise reveal the same thing.

Era and party are the strongest priors on the label — an annotator who knows a
sentence is from 2015 is primed to read positive liberty into it. Hiding them
is what makes the human labels an independent check rather than a rehearsal of
the hypothesis. The sample is shuffled, so card order does not leak era either.

## Two Stages Per Sentence

Every sentence is labeled twice, in a fixed order:

1. **From the sentence alone.** Context is not reachable until an answer is
   given. This is what the classifier sees, so it is the only human label
   comparable to it, and it is never overwritten (`label`).
2. **After the surrounding speech is revealed.** It opens automatically once
   stage 1 is recorded. Picking a different label records a revision
   (`label_with_context`, `revised`); moving on keeps the first answer.

Because the reveal happens strictly *after* a label is committed, seeing
context cannot bias the comparable label. That ordering is what lets the tool
show context to everyone without giving up the like-for-like number.

It also replaces an earlier design where context was opt-in. Opt-in made the
amount of context a matter of annotator temperament — cautious annotators
opened it constantly, confident ones never — and that variance landed straight
in inter-annotator kappa, which ceilings every other number here.

The gap between the two passes is itself a result. `ambiguous` is the shortest
class in the set (median 22 words, against 30 for both substantive labels),
which is the signature of a window artifact rather than genuine indeterminacy.
`context_effect` in the scorer reports how many `ambiguous` calls survive once
the speech around them is visible.

## Surrounding Context

The sentence being judged stays in place, highlighted, with the speech around
it muted — so the annotator reads one passage rather than matching two copies
of the same sentence. **＋ More context** widens the passage further
(3 → 9 → 25 sentences per side) and disappears once everything recovered for
that sentence is on screen. Context comes from the original sources via
`src.extract_context`:

```bash
uv run python -m src.extract_context --window 25 \
    --speech-csv path/to/hansard-speeches-v310.csv
```

`--window` sets how many sentences per side are stored, which caps how far
**＋ More context** can reach. The UI never claims to be showing a whole
speech, since a long speech can exceed the stored window.

This writes `web/data/validation_context.json` (`{id: {before, after}}`). It
carries no speaker or date, so it is safe to show while blind.

`context_reviewed` is set only when surrounding sentences were actually put on
screen — not while the file is still loading, and not for the 14 sentences that
are whole speeches with nothing around them. It is the denominator for the
revision rate, so counting those would drag the rate toward zero.

Do **not** make context conditional on which bucket a sentence came from. Tier
is derived from council agreement, so "this card came with a paragraph" would
be a visible tell for "the models disagreed here" — model information leaking
into a blind task, priming `ambiguous` on exactly the disputed items. Stage 1
is identical for every sentence for the same reason.

## Annotation Workflow

1. Each annotator opens the blind link above and labels independently.
2. Labels save to the cloud automatically as they click; **nothing needs to be
   exported**. Pull each annotator's file when you are ready to score:
   ```bash
   curl -s "https://freedom-semantic-shift.vercel.app/api/labels?user=alice@example.com" > alice.json
   ```
3. Score them:
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
- Quote `haiku_vs_human` (stage 1) as the classifier's accuracy.
  `haiku_vs_human_context_revised` scores it against a standard it could not
  have met, since it never saw the context — report it as the cost of the
  one-sentence window, not as accuracy.
- Expect the revised pass to have a lower consensus `n`: revisions give
  annotators a second chance to diverge, and strict ties are dropped.

## Budget

Reading volume per sentence drives how many a volunteer can do:

| Shown | Words | 500 sentences |
|---|---|---|
| Sentence alone | 32 | 1.1 h |
| ± 3 sentences | 146 | 4.9 h |
| ± 9 sentences | 324 | 10.8 h |
| ± 25 sentences | 656 | 21.9 h |

Both stages together cost roughly the ±3 row, since stage 2 opens at three
sentences per side. Plan **~250 sentences** for a half-day sitting, not the
400–500 that a sentence-only pass would allow.

## Analysis Rule

Primary trend metric:

`positive_liberty / (positive_liberty + negative_liberty)`

Sensitivity metric:

`positive_liberty / (positive_liberty + negative_liberty + ambiguous)`

The main claim is robust only if the direction of change is similar across the
primary and sensitivity metrics and stable after adjudication.
