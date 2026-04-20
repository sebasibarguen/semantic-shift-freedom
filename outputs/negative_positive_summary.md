# Negative vs Positive Freedom Analysis: Summary of Findings

## Key Finding: Non-Linear Trajectory

The shift from negative to positive freedom is **NOT a simple linear progression**. Instead:

| Period | % Negative ("freedom FROM") | % Positive ("freedom TO") | Source |
|--------|----------------------------|---------------------------|--------|
| 1500-1700 | **21%** | 79% | EEBO-TCP |
| 1800s | 57% | 43% | Google Ngrams |
| 1870-1920 | **76%** (peak) | 24% | Google Ngrams |
| 1960s | 50% | 50% | Google Ngrams |
| 2010s | **35%** | 65% | Google Ngrams |

## The Trajectory

```
        % Negative Framing ("freedom FROM")
   80% |           ****
       |         **    **
   60% |       **        **
       |     **            **
   40% |   **                ***
       | **                     ****
   20% |*
       +--------------------------------
        1500  1700  1800  1900  2000  2020

        Early    Peak      Return to
        Modern   Negative  Positive
```

## Interpretation

### Phase 1: Early Modern (1500-1700) — Predominantly Positive
- "Liberty to speak", "freedom to worship", "liberty to do"
- Freedom = PERMISSION to act
- Religious context: "liberty of conscience" (freedom to worship as you choose)

### Phase 2: 19th Century — Peak Negative
- "Freedom from slavery", "freedom from oppression", "freedom from tyranny"
- Freedom = ABSENCE of constraints
- Historical context: Abolition debates, anti-slavery movement
- **1820-1830**: Biggest semantic shift (correlates with abolition debates)

### Phase 3: Early 20th Century — Sustained Negative
- "Freedom from want", "freedom from fear" (FDR's Four Freedoms, 1941)
- Still defined by what you're FREE FROM

### Phase 4: Late 20th Century — Return to Positive
- **1965**: Crossover year (freedom TO > freedom FROM)
- "Freedom to choose", "freedom to act", "freedom to live"
- Coincides with: Civil Rights Movement, individual rights discourse

### Phase 5: 21st Century — Positive Dominance
- "Financial freedom" (freedom TO retire early, TO not work)
- "Medical freedom" (freedom TO refuse treatment)
- Freedom = ENTITLEMENT or CAPACITY

## What This Means for the Zero-Sum Hypothesis

The shift toward positive freedom framing has implications:

| Type | Framing | Zero-Sum Potential |
|------|---------|-------------------|
| "Freedom from slavery" | Negative | Minimal (removing constraint doesn't cost others) |
| "Freedom to choose" | Positive | Low (abstract capacity) |
| "Financial freedom" | Positive | **High** (passive income comes from others' labor) |
| "Medical freedom" | Positive | **High** (herd immunity depends on compliance) |

Modern positive freedoms increasingly describe **entitlements that may impose costs on others**.

## Data Sources

1. **Google Ngrams (1800-2019)**: `outputs/negative_positive_ngrams.json`
2. **HistWords Embeddings (1800-1990)**: `outputs/negative_positive_embeddings.json`
3. **EEBO-TCP (1500-1700)**: `outputs/negative_positive_eebo.json`

## Key Statistics

- **1965**: Year "freedom to" surpassed "freedom from" in published books
- **3.45x**: Growth of "freedom to" (1800→2019)
- **1.52x**: Growth of "freedom from" (1800→2019)
- **76%**: Peak negative framing (1900-1920)
- **-38%**: Autonomy-freedom distance change (1800→1990) — dramatic convergence

## Hypotheses Tested

| Hypothesis | Result |
|------------|--------|
| H7: Simple negative→positive shift | **PARTIAL** (trajectory is non-linear) |
| H8: Modern freedoms are positive | **SUPPORTED** (financial, medical freedom) |
| H9: Embedding distance confirms shift | **PARTIAL** (autonomy convergence confirmed) |

## Conclusion

"Freedom" has not simply shifted from negative to positive. Instead:

1. **Early Modern**: Predominantly positive (permission to act)
2. **19th Century**: Shifted to negative (absence of slavery/oppression)
3. **Late 20th Century**: Returned to positive (individual rights/entitlements)

The current surge in positive freedom language ("financial freedom", "medical freedom") represents a return to historical patterns, not a novel development. However, the **content** of modern positive freedoms is new: they describe economic entitlements and personal autonomy claims that may impose costs on others.
