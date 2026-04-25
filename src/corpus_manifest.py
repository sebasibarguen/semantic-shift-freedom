# ABOUTME: Builds a lightweight audit manifest for generated sentence corpora.
# ABOUTME: Records coverage gaps, method keys, and label availability.

import argparse
import json
from collections import Counter
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text())


def contiguous_missing_years(years: list[int]) -> list[dict]:
    """Return missing-year ranges inside the observed min/max year span."""
    if not years:
        return []
    year_set = set(years)
    missing = [y for y in range(min(years), max(years) + 1) if y not in year_set]
    if not missing:
        return []

    ranges = []
    start = prev = missing[0]
    for year in missing[1:]:
        if year == prev + 1:
            prev = year
            continue
        ranges.append({"start": start, "end": prev, "n_years": prev - start + 1})
        start = prev = year
    ranges.append({"start": start, "end": prev, "n_years": prev - start + 1})
    return ranges


def scan_sentence_files(data_dir: Path) -> dict:
    """Scan sentence JSON files for method and label coverage."""
    method_counts = Counter()
    llm_labels = Counter()
    sentence_files = {}
    years = set()
    parties = set()
    total_records = 0

    for path in sorted(data_dir.glob("sentences_*s.json")):
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")

        file_methods = Counter()
        file_labels = Counter()
        for record in data:
            total_records += 1
            if record.get("year") is not None:
                years.add(int(record["year"]))
            if record.get("party"):
                parties.add(record["party"])

            methods = record.get("methods", {})
            for method in methods:
                method_counts[method] += 1
                file_methods[method] += 1

            label = methods.get("llm", {}).get("label", "missing")
            llm_labels[label] += 1
            file_labels[label] += 1

        sentence_files[path.name] = {
            "records": len(data),
            "size_kb": round(path.stat().st_size / 1024, 1),
            "method_counts": dict(sorted(file_methods.items())),
            "llm_label_counts": dict(sorted(file_labels.items())),
        }

    sorted_years = sorted(years)
    return {
        "total_records": total_records,
        "years": sorted_years,
        "year_range": [sorted_years[0], sorted_years[-1]] if sorted_years else None,
        "missing_year_ranges": contiguous_missing_years(sorted_years),
        "parties": sorted(parties),
        "method_counts": dict(sorted(method_counts.items())),
        "llm_label_counts": dict(sorted(llm_labels.items())),
        "sentence_files": sentence_files,
        "stale_method_keys": [key for key in method_counts if key == "from_to"],
    }


def build_manifest(data_dir: Path) -> dict:
    """Build an audit manifest from web/data and index metadata."""
    index_path = data_dir / "index.json"
    index = load_json(index_path) if index_path.exists() else {}
    scanned = scan_sentence_files(data_dir)

    return {
        "metadata": {
            "data_dir": str(data_dir),
            "purpose": "Audit generated sentence-corpus coverage and stale method fields.",
            "note": (
                "This manifest is generated from local JSON outputs. It does not "
                "replace source-corpus provenance, but it makes coverage gaps "
                "and stale methods explicit."
            ),
        },
        "index_summary": {
            "total_sentences": index.get("total_sentences"),
            "methods": index.get("methods", []),
            "decades": index.get("facets", {}).get("decades", []),
            "year_count": len(index.get("facets", {}).get("years", [])),
        },
        "scan": scanned,
        "checks": {
            "index_total_matches_scan": index.get("total_sentences") == scanned["total_records"]
            if index else None,
            "from_to_removed": "from_to" not in scanned["method_counts"],
            "llm_labels_present": sum(
                count for label, count in scanned["llm_label_counts"].items()
                if label not in {"missing", "error"}
            ),
        },
    }


def main() -> None:
    project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "web" / "data")
    parser.add_argument("--output", type=Path, default=project_root / "outputs" / "corpus_manifest.json")
    args = parser.parse_args()

    manifest = build_manifest(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))

    scan = manifest["scan"]
    print("=" * 70)
    print("CORPUS MANIFEST")
    print("=" * 70)
    print(f"Records: {scan['total_records']:,}")
    print(f"Year range: {scan['year_range']}")
    print(f"Missing year ranges: {scan['missing_year_ranges']}")
    print(f"Method keys: {sorted(scan['method_counts'])}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
