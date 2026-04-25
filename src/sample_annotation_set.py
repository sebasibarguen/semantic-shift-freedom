# ABOUTME: Samples sentence records for human annotation of liberty labels.
# ABOUTME: Produces a deterministic CSV template for inter-annotator validation.

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .liberty_trends import get_llm_label


ANNOTATION_FIELDS = [
    "id",
    "decade",
    "year",
    "date",
    "word",
    "speaker",
    "party",
    "sentence",
    "llm_label",
    "annotator_1",
    "annotator_2",
    "adjudicated_label",
    "notes",
]


def load_records(data_dir: Path) -> list[dict]:
    records = []
    for path in sorted(data_dir.glob("sentences_*s.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")
        records.extend(data)
    return records


def bucket_key(record: dict) -> tuple[int, str]:
    year = int(record["year"])
    decade = (year // 10) * 10
    return decade, get_llm_label(record)


def stratified_sample(records: list[dict], per_bucket: int, seed: int) -> list[dict]:
    """Sample up to per_bucket records per decade/LLM-label bucket."""
    rng = random.Random(seed)
    buckets: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for record in records:
        if record.get("year") is None:
            continue
        buckets[bucket_key(record)].append(record)

    sampled = []
    for key in sorted(buckets):
        bucket = buckets[key]
        rng.shuffle(bucket)
        sampled.extend(bucket[:per_bucket])

    sampled.sort(key=lambda r: (int(r["year"]), get_llm_label(r), r.get("id", "")))
    return sampled


def to_annotation_row(record: dict) -> dict:
    year = int(record["year"])
    return {
        "id": record.get("id", ""),
        "decade": (year // 10) * 10,
        "year": year,
        "date": record.get("date", ""),
        "word": record.get("word", ""),
        "speaker": record.get("speaker", ""),
        "party": record.get("party", ""),
        "sentence": record.get("sentence", ""),
        "llm_label": get_llm_label(record),
        "annotator_1": "",
        "annotator_2": "",
        "adjudicated_label": "",
        "notes": "",
    }


def write_csv(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(to_annotation_row(record))


def main() -> None:
    project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "web" / "data")
    parser.add_argument("--output", type=Path, default=project_root / "outputs" / "annotation_sample.csv")
    parser.add_argument("--per-bucket", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_records(args.data_dir)
    sampled = stratified_sample(records, per_bucket=args.per_bucket, seed=args.seed)
    write_csv(sampled, args.output)

    print("=" * 70)
    print("ANNOTATION SAMPLE")
    print("=" * 70)
    print(f"Input records: {len(records):,}")
    print(f"Sampled records: {len(sampled):,}")
    print(f"Rows per decade/label bucket: up to {args.per_bucket}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
