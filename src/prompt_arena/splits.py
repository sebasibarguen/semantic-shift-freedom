# ABOUTME: Deterministic train/dev/test splits of the council gold set.
# ABOUTME: Hash-based bucketing so the same sentence id always lands in the same split.

import hashlib
import json
from pathlib import Path
from typing import Literal

SplitName = Literal["train", "dev", "test"]

# 14/3/3 = 70/15/15 over 20 buckets
_BUCKETS = 20
_TRAIN_BUCKETS = 14
_DEV_BUCKETS = 3
_TEST_BUCKETS = 3
assert _TRAIN_BUCKETS + _DEV_BUCKETS + _TEST_BUCKETS == _BUCKETS


def assign_split(sentence_id: str) -> SplitName:
    """Map a sentence id to one of {train, dev, test} deterministically."""
    h = int(hashlib.sha256(sentence_id.encode("utf-8")).hexdigest()[:8], 16) % _BUCKETS
    if h < _TRAIN_BUCKETS:
        return "train"
    if h < _TRAIN_BUCKETS + _DEV_BUCKETS:
        return "dev"
    return "test"


def load_gold(gold_path: Path, include_silver: bool = False, silver_path: Path | None = None) -> list[dict]:
    """Load council gold labels (and optionally silver) into a flat list."""
    records = json.loads(gold_path.read_text())
    if include_silver and silver_path and silver_path.exists():
        silver = json.loads(silver_path.read_text())
        # tag origin so we can weight or filter later
        for r in records:
            r["_origin"] = "gold"
        for r in silver:
            r["_origin"] = "silver"
        records = records + silver
    return records


def split_gold(records: list[dict], split: SplitName) -> list[dict]:
    """Filter records to the named split via deterministic hash."""
    return [r for r in records if assign_split(r["id"]) == split]


def split_summary(records: list[dict]) -> dict:
    counts = {"train": 0, "dev": 0, "test": 0}
    for r in records:
        counts[assign_split(r["id"])] += 1
    return counts
