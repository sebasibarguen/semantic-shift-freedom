# ABOUTME: Builds a blind validation set for human labeling plus a held-out answer key.
# ABOUTME: Mixes a representative random draw with the council's hard (silver/disputed) cases.

"""
Sample sentences for independent human validation of the liberty labels.

The output is split in two so annotators never see a model's answer:

    web/data/validation_set.json      annotator-facing; model labels stripped
    outputs/validation_answer_key.json   scorer-only; Haiku + council labels by id

The sample deliberately mixes tiers so the scorer can measure different things:

    random           pure random draw from the full corpus — the only subset that
                     is representative, so the only one used to recompute the trend
    council_gold     sentences all three council models agreed on (easy cases)
    council_silver   2/3 council agreement
    council_disputed no council majority — the hardest cases, excluded from every
                     accuracy number the project currently reports

Load it in the browser tool blind:

    web/compare.html?blind=1&set=validation

Then score the exported labels:

    uv run python -m src.score_annotations \\
        --answer-key outputs/validation_answer_key.json \\
        --labels alice.json bob.json --names alice bob
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


# Fields shown to the annotator — the same context the classifier itself receives
# (year/speaker/party), and nothing that reveals a model's judgment.
ANNOTATOR_FIELDS = ("id", "year", "date", "word", "speaker", "party", "sentence")

COUNCIL_TIERS = ("gold", "silver", "disputed")


def load_corpus(data_dir: Path) -> dict[str, dict]:
    """Map sentence id → full record across all decade files."""
    by_id: dict[str, dict] = {}
    for path in sorted(data_dir.glob("sentences_*s.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")
        for rec in data:
            by_id[rec["id"]] = rec
    return by_id


def load_council(council_dir: Path) -> list[dict]:
    """Flatten gold/silver/disputed council records into one list (tier preserved)."""
    records: list[dict] = []
    for tier in COUNCIL_TIERS:
        path = council_dir / f"{tier}.json"
        if path.exists():
            records.extend(json.loads(path.read_text()))
    return records


def haiku_label_of(record: dict) -> str | None:
    return (record.get("methods", {}).get("llm") or {}).get("label")


def annotator_record(corpus_rec: dict, reason: str) -> dict:
    """Strip a corpus record down to the blind annotator view."""
    out = {field: corpus_rec.get(field) for field in ANNOTATOR_FIELDS}
    out["methods"] = {}  # the browser tool reads s.methods; keep it empty, not absent
    out["_sample_reason"] = reason
    return out


def build_validation_sample(
    corpus_by_id: dict[str, dict],
    council_records: list[dict],
    *,
    n_random: int = 300,
    n_gold: int = 120,
    n_silver: int = 200,
    n_disputed: int | None = None,   # None → take all disputed
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Return (annotator_records, answer_key). Deterministic given the same inputs/seed.

    Selection order is fixed (disputed → silver → gold → random) so a given id always
    lands in the same bucket; ids already chosen are not re-drawn.
    """
    rng = random.Random(seed)

    council_by_id: dict[str, dict] = {}
    by_tier: dict[str, list[str]] = defaultdict(list)
    for rec in council_records:
        sid = rec["id"]
        council_by_id[sid] = {"tier": rec.get("tier"), "gold_label": rec.get("gold_label")}
        by_tier[rec.get("tier")].append(sid)

    chosen: dict[str, str] = {}  # id → reason

    def take(candidate_ids: list[str], n: int | None, reason: str) -> None:
        pool = [i for i in candidate_ids if i in corpus_by_id and i not in chosen]
        rng.shuffle(pool)
        picked = pool if n is None else pool[:n]
        for sid in picked:
            chosen[sid] = reason

    take(by_tier.get("disputed", []), n_disputed, "council_disputed")
    take(by_tier.get("silver", []), n_silver, "council_silver")
    take(by_tier.get("gold", []), n_gold, "council_gold")
    take(list(corpus_by_id.keys()), n_random, "random")

    # Shuffle the final order so tiers/labels are not clustered for the annotator.
    ordered_ids = list(chosen.keys())
    rng.shuffle(ordered_ids)

    records: list[dict] = []
    answer_key: dict[str, dict] = {}
    for sid in ordered_ids:
        corpus_rec = corpus_by_id[sid]
        reason = chosen[sid]
        records.append(annotator_record(corpus_rec, reason))
        council = council_by_id.get(sid, {})
        answer_key[sid] = {
            "year": corpus_rec.get("year"),
            "sample_reason": reason,
            "haiku_label": haiku_label_of(corpus_rec),
            "council_tier": council.get("tier"),
            "council_gold": council.get("gold_label"),
        }

    reason_counts: dict[str, int] = defaultdict(int)
    for reason in chosen.values():
        reason_counts[reason] += 1

    meta = {
        "n": len(records),
        "seed": seed,
        "config": {
            "n_random": n_random,
            "n_gold": n_gold,
            "n_silver": n_silver,
            "n_disputed": n_disputed,
        },
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    return records, {"meta": meta, "keys": answer_key}


def main() -> None:
    project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=project_root / "web" / "data")
    parser.add_argument("--council-dir", type=Path,
                        default=project_root / "outputs" / "council" / "full")
    parser.add_argument("--set-output", type=Path,
                        default=project_root / "web" / "data" / "validation_set.json")
    parser.add_argument("--key-output", type=Path,
                        default=project_root / "outputs" / "validation_answer_key.json")
    parser.add_argument("--n-random", type=int, default=300)
    parser.add_argument("--n-gold", type=int, default=120)
    parser.add_argument("--n-silver", type=int, default=200)
    parser.add_argument("--n-disputed", type=int, default=None,
                        help="default: all disputed council cases")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus_by_id = load_corpus(args.data_dir)
    council_records = load_council(args.council_dir)

    records, answer_key = build_validation_sample(
        corpus_by_id,
        council_records,
        n_random=args.n_random,
        n_gold=args.n_gold,
        n_silver=args.n_silver,
        n_disputed=args.n_disputed,
        seed=args.seed,
    )

    args.set_output.parent.mkdir(parents=True, exist_ok=True)
    args.set_output.write_text(json.dumps({"meta": answer_key["meta"], "records": records}, indent=2))
    args.key_output.parent.mkdir(parents=True, exist_ok=True)
    args.key_output.write_text(json.dumps(answer_key, indent=2))

    print("=" * 70)
    print("VALIDATION SAMPLE")
    print("=" * 70)
    print(f"Corpus records:  {len(corpus_by_id):,}")
    print(f"Council records: {len(council_records):,}")
    print(f"Sampled:         {len(records):,}")
    for reason, count in answer_key["meta"]["reason_counts"].items():
        print(f"  {reason:<18} {count:>5}")
    print(f"\nAnnotator set → {args.set_output}  (model labels stripped)")
    print(f"Answer key    → {args.key_output}  (scorer only — do not share)")
    print("\nLabel it blind:  web/compare.html?blind=1&set=validation")


if __name__ == "__main__":
    main()
