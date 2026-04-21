# ABOUTME: Extracts and classifies every freedom/liberty sentence from Hansard.
# ABOUTME: Produces decade-chunked JSON files for the sentence comparison web tool.

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SENTENCE_RE = re.compile(r"[^.!?;]+[.!?;]?", re.DOTALL)
FROM_RE = re.compile(r"\bfreedom\s+from\s+(\w+)", re.IGNORECASE)
TO_RE = re.compile(r"\bfreedom\s+to\s+(\w+)", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z]+")

# Berlin-style pole words for a simple heuristic score
CONSTRAINT_POLES = {
    "slavery", "bondage", "oppression", "coercion", "compulsion", "tyranny",
    "despotism", "servitude", "domination", "subjection", "chains", "restraint",
    "interference", "discrimination", "persecution", "arrest", "detention",
    "fear", "want", "hunger", "pain", "torture", "censorship", "ban",
}
AGENCY_POLES = {
    "autonomy", "choice", "choose", "decide", "capacity", "ability", "power",
    "opportunity", "right", "express", "innovate", "participate", "act",
    "speak", "worship", "travel", "movement", "enterprise", "self",
    "dignity", "fulfillment", "development", "education", "prosperity",
}


def extract_sentences(df, frequency_data):
    """Extract every sentence containing freedom/liberty with classifications."""
    from .domain_tagger import DomainTagger
    tagger = DomainTagger()

    sentences_by_decade = defaultdict(list)
    total = 0
    skipped = 0

    relevant = df[df["has_freedom"] | df["has_liberty"]].copy()
    print(f"Processing {len(relevant):,} speeches with freedom/liberty...")

    for _, row in relevant.iterrows():
        text = str(row.get("speech") or row.get("text") or "")
        year = int(row["year"])
        date = str(row.get("date") or row.get("speech_date") or "")
        speaker = str(row.get("display_as") or row.get("speaker") or "Unknown")
        party = str(row.get("party") or "Unknown")
        decade = (year // 10) * 10

        # Split into sentences
        for i, match in enumerate(SENTENCE_RE.finditer(text)):
            sent = match.group(0).strip()
            sent_lower = sent.lower()

            if len(sent) < 15 or len(sent) > 500:
                continue

            has_freedom = "freedom" in sent_lower
            has_liberty = "liberty" in sent_lower
            if not has_freedom and not has_liberty:
                continue

            word = "freedom" if has_freedom else "liberty"

            # Generate stable ID
            id_source = f"{date}-{speaker}-{i}"
            sid = f"{date[:10]}-{hashlib.md5(id_source.encode()).hexdigest()[:6]}-{i:03d}"

            # Method 1: FROM/TO framing
            from_to = classify_from_to(sent)

            # Method 2: Domain tagging
            words_in_sent = WORD_RE.findall(sent_lower)
            domain_dist = tagger.get_domain_distribution(words_in_sent)
            # Keep only non-zero domains, drop 'untagged'
            domains = {k: v for k, v in domain_dist.items() if v > 0 and k != "untagged"}

            # Method 3: Pole word heuristic
            pole_score = compute_pole_score(words_in_sent)

            # Method 4: Frequency context
            freq_ctx = get_frequency_context(year, frequency_data)

            record = {
                "id": sid,
                "sentence": sent,
                "word": word,
                "year": year,
                "date": date[:10],
                "speaker": speaker,
                "party": party,
                "methods": {
                    "from_to": from_to,
                    "domains": domains,
                    "pole_score": pole_score,
                    "freq": freq_ctx,
                },
            }

            sentences_by_decade[decade].append(record)
            total += 1

        if total % 10000 == 0 and total > 0:
            print(f"  Extracted {total:,} sentences...")

    print(f"  Total: {total:,} sentences across {len(sentences_by_decade)} decades")
    return sentences_by_decade


def classify_from_to(sentence):
    """Classify a sentence's FROM/TO framing."""
    from_match = FROM_RE.search(sentence)
    to_match = TO_RE.search(sentence)

    if from_match:
        return {"type": "from", "object": from_match.group(1).lower()}
    elif to_match:
        return {"type": "to", "object": to_match.group(1).lower()}
    return {"type": "neither", "object": None}


def compute_pole_score(words):
    """Compute a simple constraint-vs-agency score from pole word counts."""
    c_count = sum(1 for w in words if w in CONSTRAINT_POLES)
    a_count = sum(1 for w in words if w in AGENCY_POLES)
    total = c_count + a_count
    if total == 0:
        return {"constraint": 0, "agency": 0, "score": 0.0}
    score = round((a_count - c_count) / total, 2)
    return {"constraint": c_count, "agency": a_count, "score": score}


def get_frequency_context(year, frequency_data):
    """Attach frequency context for the sentence's year."""
    year_str = str(year)
    if year_str in frequency_data:
        d = frequency_data[year_str]
        return {
            "rate": d.get("freedom_rate", 0),
            "trend": "rising" if d.get("freedom_rate", 0) > 12 else "stable",
        }
    return {"rate": 0, "trend": "unknown"}


def write_output(sentences_by_decade, output_dir):
    """Write decade-chunked JSON files and an index manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "total_sentences": 0,
        "files": {},
        "facets": {
            "decades": [],
            "parties": set(),
            "years": set(),
        },
    }

    for decade in sorted(sentences_by_decade.keys()):
        data = sentences_by_decade[decade]
        filename = f"sentences_{decade}s.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(data, f, separators=(",", ":"))

        size_kb = filepath.stat().st_size / 1024
        manifest["total_sentences"] += len(data)
        manifest["files"][str(decade)] = {
            "file": filename,
            "count": len(data),
            "size_kb": round(size_kb, 1),
        }
        manifest["facets"]["decades"].append(decade)
        for rec in data:
            manifest["facets"]["parties"].add(rec["party"])
            manifest["facets"]["years"].add(rec["year"])

        print(f"  {filename}: {len(data):,} sentences ({size_kb:.0f} KB)")

    # Convert sets to sorted lists for JSON
    manifest["facets"]["parties"] = sorted(manifest["facets"]["parties"])
    manifest["facets"]["years"] = sorted(manifest["facets"]["years"])

    with open(output_dir / "index.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Total: {manifest['total_sentences']:,} sentences")
    print(f"  Manifest: {output_dir / 'index.json'}")


def run_extraction(csv_path="/data/hansard-speeches-v310.csv", output_dir="/tmp/hansard_data"):
    """Run the full extraction pipeline."""
    from .hansard_analysis import load_hansard

    # Load dataset
    df = load_hansard(csv_path)

    # Need has_freedom / has_liberty columns
    df["has_freedom"] = df["text_lower"].str.contains("freedom", na=False)
    df["has_liberty"] = df["text_lower"].str.contains("liberty", na=False)

    # Load frequency data for context
    freq_path = Path(__file__).parent.parent / "outputs" / "hansard_analysis.json"
    frequency_data = {}
    if freq_path.exists():
        with open(freq_path) as f:
            frequency_data = json.load(f).get("frequency", {})
    else:
        print("  Warning: hansard_analysis.json not found, frequency context will be empty")

    # Extract
    sentences_by_decade = extract_sentences(df, frequency_data)

    # Write output
    write_output(sentences_by_decade, output_dir)

    return sentences_by_decade


if __name__ == "__main__":
    run_extraction(output_dir="web/data")
