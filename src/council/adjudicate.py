# ABOUTME: Adjudicates per-sentence council labels into gold / silver / disputed buckets.
# ABOUTME: Per the chosen rules, ambiguous and disputed cases are NOT auto-labeled.

from collections import Counter
from dataclasses import asdict

from .providers.base import CouncilLabel
from .schema import LABEL_VALUES

# Adjudication outcomes
TIER_GOLD = "gold"
TIER_SILVER = "silver"
TIER_DISPUTED = "disputed"          # 0-majority — leave for human review


def adjudicate_one(
    sentence_id: str,
    sentence: dict,
    verdicts: list[CouncilLabel],
    confidence_floor: float = 0.6,
) -> dict:
    """Apply majority + confidence rules to a single sentence's verdicts.

    Returns a record with:
        sentence_id, tier, gold_label (or None), majority_size, all_verdicts,
        rationales, low_confidence flag.
    """
    by_label: Counter = Counter()
    for v in verdicts:
        if v.label in LABEL_VALUES:  # exclude "error" / "missing"
            by_label[v.label] += 1
    total_valid = sum(by_label.values())

    # No valid verdicts at all
    if total_valid == 0:
        return _record(
            sentence_id, sentence, verdicts,
            tier=TIER_DISPUTED,
            gold_label=None,
            majority_size=0,
            note="all providers errored",
        )

    top_label, top_count = by_label.most_common(1)[0]

    if top_count == total_valid and total_valid >= 2:
        # Unanimous (among non-erroring providers)
        confs = [v.confidence for v in verdicts if v.label == top_label and v.confidence >= 0]
        low = bool(confs) and max(confs) < confidence_floor
        return _record(
            sentence_id, sentence, verdicts,
            tier=TIER_GOLD,
            gold_label=top_label,
            majority_size=top_count,
            low_confidence=low,
        )

    # Majority but not unanimous (e.g. 2/3)
    if top_count > total_valid - top_count:
        confs = [v.confidence for v in verdicts if v.label == top_label and v.confidence >= 0]
        low = bool(confs) and max(confs) < confidence_floor
        return _record(
            sentence_id, sentence, verdicts,
            tier=TIER_SILVER,
            gold_label=top_label,
            majority_size=top_count,
            low_confidence=low,
        )

    # No majority — disputed
    return _record(
        sentence_id, sentence, verdicts,
        tier=TIER_DISPUTED,
        gold_label=None,
        majority_size=top_count,
        note="no majority among valid verdicts",
    )


def _record(sentence_id, sentence, verdicts, *, tier, gold_label, majority_size,
            low_confidence=False, note=None) -> dict:
    return {
        "id": sentence_id,
        "year": sentence.get("year"),
        "speaker": sentence.get("speaker"),
        "party": sentence.get("party"),
        "sentence": sentence.get("sentence"),
        "haiku_label": sentence.get("haiku_label"),
        "_sample_reason": sentence.get("_sample_reason"),
        "tier": tier,
        "gold_label": gold_label,
        "majority_size": majority_size,
        "n_valid_verdicts": sum(1 for v in verdicts if v.label in LABEL_VALUES),
        "low_confidence": low_confidence,
        "note": note,
        "verdicts": [asdict(v) for v in verdicts],
        "rationales": [v.rationale for v in verdicts if v.rationale],
    }


def adjudicate_all(
    sample: list[dict],
    provider_outputs: dict[str, list[CouncilLabel]],
    confidence_floor: float = 0.6,
) -> dict:
    """Run adjudication across all sentences. provider_outputs maps provider_name → list aligned with sample.

    Returns {gold: [...], silver: [...], disputed: [...], summary: {...}}.
    """
    n = len(sample)
    for name, outs in provider_outputs.items():
        if len(outs) != n:
            raise ValueError(f"provider {name} returned {len(outs)} labels for {n} sentences")

    gold, silver, disputed = [], [], []
    for i, rec in enumerate(sample):
        verdicts = [provider_outputs[name][i] for name in provider_outputs]
        adj = adjudicate_one(rec["id"], rec, verdicts, confidence_floor=confidence_floor)
        bucket = {TIER_GOLD: gold, TIER_SILVER: silver, TIER_DISPUTED: disputed}[adj["tier"]]
        bucket.append(adj)

    summary = {
        "n_sample": n,
        "n_providers": len(provider_outputs),
        "providers": list(provider_outputs.keys()),
        "n_gold": len(gold),
        "n_silver": len(silver),
        "n_disputed": len(disputed),
        "gold_rate": round(len(gold) / n, 4) if n else 0,
        "silver_rate": round(len(silver) / n, 4) if n else 0,
        "disputed_rate": round(len(disputed) / n, 4) if n else 0,
        "low_confidence_in_gold": sum(1 for g in gold if g["low_confidence"]),
        "label_distribution_gold": _dist(gold),
        "label_distribution_silver": _dist(silver),
    }
    return {"gold": gold, "silver": silver, "disputed": disputed, "summary": summary}


def _dist(records: list[dict]) -> dict:
    c: Counter = Counter()
    for r in records:
        if r["gold_label"]:
            c[r["gold_label"]] += 1
    return dict(sorted(c.items()))
