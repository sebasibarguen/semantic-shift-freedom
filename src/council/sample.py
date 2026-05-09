# ABOUTME: Strategic stratified sampler for council labeling.
# ABOUTME: Stratifies by decade × Haiku label, oversamples hard cases, anchors with eval set.

import json
import random
import re
from collections import defaultdict
from pathlib import Path


# Patterns that historically tripped the v1 classifier — oversample them
# so the council gets a solid signal on the boundary cases.
FREEDOM_OF_X_RE = re.compile(
    r"\bfreedom of (speech|press|religion|conscience|debate|expression|"
    r"contract|trade|navigation|movement|association|information|choice)\b",
    re.IGNORECASE,
)


def sbert_direction(record: dict) -> str | None:
    """Coerce SBERT label to {pos, neg, none}."""
    sb = record.get("methods", {}).get("sbert") or {}
    label = sb.get("label")
    if label == "agency":
        return "pos"
    if label == "constraint":
        return "neg"
    return None


def llm_direction(record: dict) -> str | None:
    """Coerce LLM label to {pos, neg, none}."""
    label = record.get("methods", {}).get("llm", {}).get("label")
    if label == "positive_liberty":
        return "pos"
    if label == "negative_liberty":
        return "neg"
    return None


def is_method_disagreement(record: dict) -> bool:
    a = sbert_direction(record)
    b = llm_direction(record)
    return a is not None and b is not None and a != b


def is_freedom_of_x(record: dict) -> bool:
    return bool(FREEDOM_OF_X_RE.search(record.get("sentence", "")))


def llm_label(record: dict) -> str:
    label = record.get("methods", {}).get("llm", {}).get("label")
    return label if label in {"positive_liberty", "negative_liberty", "ambiguous", "other"} else "missing"


def decade_of(record: dict) -> int:
    return (int(record["year"]) // 10) * 10


def load_corpus(data_dir: Path) -> list[dict]:
    records: list[dict] = []
    for p in sorted(data_dir.glob("sentences_*s.json")):
        records.extend(json.loads(p.read_text()))
    return records


def load_anchor_set(anchor_path: Path) -> list[str]:
    """opus_vs_haiku.json or similar — return the sentence ids to force-include."""
    if not anchor_path.exists():
        return []
    return [r["id"] for r in json.loads(anchor_path.read_text()) if "id" in r]


def stratified_pick(records: list[dict], per_bucket: int, rng: random.Random) -> list[dict]:
    """Decade × LLM-label buckets; up to per_bucket per cell."""
    buckets: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in records:
        buckets[(decade_of(r), llm_label(r))].append(r)
    chosen: list[dict] = []
    for key in sorted(buckets):
        pool = buckets[key]
        rng.shuffle(pool)
        chosen.extend(pool[:per_bucket])
    return chosen


def take_n(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n <= 0 or not pool:
        return []
    rng.shuffle(pool)
    return pool[:n]


def build_sample(
    data_dir: Path,
    anchor_path: Path | None = None,
    per_decade_label: int = 32,    # 23 × 4 ≈ 92 cells × 32 ≈ 2,944
    n_disagreement: int = 600,
    n_freedom_of_x: int = 400,
    n_random: int = 500,
    n_short_tail: int = 500,        # underrepresented decades (1810s-1900s)
    short_tail_decades: tuple[int, ...] = (1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900),
    seed: int = 42,
) -> dict:
    """Return the deterministic 5K sample with provenance for each pick."""
    rng = random.Random(seed)
    all_records = load_corpus(data_dir)
    by_id = {r["id"]: r for r in all_records}

    anchor_ids = load_anchor_set(anchor_path) if anchor_path else []
    chosen_ids: dict[str, str] = {}  # id → reason

    # 1. Anchor set (regression test)
    for sid in anchor_ids:
        if sid in by_id:
            chosen_ids[sid] = "anchor"

    # 2. Stratified base grid
    for r in stratified_pick(all_records, per_decade_label, rng):
        chosen_ids.setdefault(r["id"], "stratified")

    # 3. Method disagreement
    pool = [r for r in all_records if r["id"] not in chosen_ids and is_method_disagreement(r)]
    for r in take_n(pool, n_disagreement, rng):
        chosen_ids.setdefault(r["id"], "disagreement")

    # 4. "freedom of X" patterns
    pool = [r for r in all_records if r["id"] not in chosen_ids and is_freedom_of_x(r)]
    for r in take_n(pool, n_freedom_of_x, rng):
        chosen_ids.setdefault(r["id"], "freedom_of_x")

    # 5. Pure random
    pool = [r for r in all_records if r["id"] not in chosen_ids]
    for r in take_n(pool, n_random, rng):
        chosen_ids.setdefault(r["id"], "random")

    # 6. Short-tail decade boost
    pool = [
        r for r in all_records
        if r["id"] not in chosen_ids and decade_of(r) in short_tail_decades
    ]
    for r in take_n(pool, n_short_tail, rng):
        chosen_ids.setdefault(r["id"], "short_tail")

    sample = []
    for sid, reason in chosen_ids.items():
        rec = by_id[sid]
        sample.append({
            "id": rec["id"],
            "year": rec["year"],
            "speaker": rec.get("speaker"),
            "party": rec.get("party"),
            "sentence": rec["sentence"],
            "haiku_label": rec.get("methods", {}).get("llm", {}).get("label"),
            "haiku_rationale": rec.get("methods", {}).get("llm", {}).get("rationale"),
            "_sample_reason": reason,
        })

    # Stable order for reproducibility
    sample.sort(key=lambda r: r["id"])

    return {
        "n": len(sample),
        "seed": seed,
        "config": {
            "per_decade_label": per_decade_label,
            "n_disagreement": n_disagreement,
            "n_freedom_of_x": n_freedom_of_x,
            "n_random": n_random,
            "n_short_tail": n_short_tail,
        },
        "reason_counts": _count(sample, "_sample_reason"),
        "decade_counts": _count(sample, lambda r: decade_of(r)),
        "haiku_label_counts": _count(sample, "haiku_label"),
        "records": sample,
    }


def _count(records: list[dict], key) -> dict:
    counts: dict = defaultdict(int)
    for r in records:
        v = key(r) if callable(key) else r.get(key)
        counts[str(v)] += 1
    return dict(sorted(counts.items()))
