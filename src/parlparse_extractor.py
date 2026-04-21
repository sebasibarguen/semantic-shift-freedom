# ABOUTME: Extracts freedom/liberty sentences from ParlParse XML (1919-2025).
# ABOUTME: Produces decade-chunked JSON matching the existing sentence browser format.

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from xml.etree.ElementTree import iterparse, ParseError


SENTENCE_RE = re.compile(r"[^.!?;]+[.!?;]?", re.DOTALL)
WORD_RE = re.compile(r"[a-z]+")
TARGET_WORDS = {"freedom", "liberty"}


def extract_text(elem):
    """Get all text from an element and its children."""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        if child.text:
            parts.append(child.text)
        if child.tail:
            parts.append(child.tail)
    return " ".join(parts).strip()


def parse_date_from_filename(filename):
    """Extract date from filename like debates1919-02-04a.xml."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return m.group(1) if m else None


def extract_from_parlparse(debates_dir, output_dir, domain_tagger=None):
    """
    Extract every freedom/liberty sentence from ParlParse XML files.
    Produces decade-chunked JSON compatible with the sentence browser.
    """
    debates_dir = Path(debates_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(debates_dir.glob("debates*.xml"))
    print(f"Found {len(xml_files)} debate files in {debates_dir}")

    sentences_by_decade = defaultdict(list)
    total_speeches = 0
    total_sentences = 0
    files_processed = 0

    for fpath in xml_files:
        date = parse_date_from_filename(fpath.name)
        if not date:
            continue

        year = int(date[:4])
        decade = (year // 10) * 10

        try:
            for event, elem in iterparse(str(fpath), events=("end",)):
                if elem.tag != "speech":
                    continue

                person_id = elem.get("person_id", "")
                speaker = elem.get("speakername", "Unknown")
                # Clean up speaker name (remove parenthetical descriptions)
                speaker_clean = re.sub(r"\s*\(.*?\)\s*$", "", speaker).strip()
                if not speaker_clean:
                    speaker_clean = "Unknown"

                # Extract full speech text from <p> elements
                paragraphs = []
                for p in elem.findall("p"):
                    text = extract_text(p)
                    if text:
                        paragraphs.append(text)

                speech_text = " ".join(paragraphs)
                speech_lower = speech_text.lower()
                total_speeches += 1

                if "freedom" not in speech_lower and "liberty" not in speech_lower:
                    elem.clear()
                    continue

                # Extract individual sentences
                for i, match in enumerate(SENTENCE_RE.finditer(speech_text)):
                    sent = match.group(0).strip()
                    sent_lower = sent.lower()

                    if len(sent) < 15 or len(sent) > 500:
                        continue

                    has_freedom = "freedom" in sent_lower
                    has_liberty = "liberty" in sent_lower
                    if not has_freedom and not has_liberty:
                        continue

                    word = "freedom" if has_freedom else "liberty"

                    # Stable ID
                    id_source = f"{date}-{speaker_clean}-{i}"
                    sid = f"{date}-{hashlib.md5(id_source.encode()).hexdigest()[:6]}-{i:03d}"

                    # Domain tagging
                    domains = {}
                    if domain_tagger:
                        words_in_sent = WORD_RE.findall(sent_lower)
                        dist = domain_tagger.get_domain_distribution(words_in_sent)
                        domains = {k: v for k, v in dist.items() if v > 0 and k != "untagged"}

                    record = {
                        "id": sid,
                        "sentence": sent,
                        "word": word,
                        "year": year,
                        "date": date,
                        "speaker": speaker_clean,
                        "party": "",  # ParlParse doesn't embed party in speech XML
                        "person_id": person_id,
                        "methods": {
                            "domains": domains,
                        },
                    }

                    sentences_by_decade[decade].append(record)
                    total_sentences += 1

                elem.clear()

        except (ParseError, Exception) as e:
            print(f"  Error parsing {fpath.name}: {e}")
            continue

        files_processed += 1
        if files_processed % 1000 == 0:
            print(f"  Processed {files_processed} files, {total_sentences:,} sentences...")

    print(f"\nDone: {files_processed} files, {total_speeches:,} speeches, {total_sentences:,} sentences")

    # Write output
    manifest = {
        "source": "ParlParse (TheyWorkForYou)",
        "total_sentences": total_sentences,
        "total_speeches": total_speeches,
        "total_files": files_processed,
        "files": {},
        "facets": {"decades": [], "years": set()},
    }

    for decade in sorted(sentences_by_decade.keys()):
        data = sentences_by_decade[decade]
        filename = f"sentences_{decade}s.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(data, f, separators=(",", ":"))

        size_kb = filepath.stat().st_size / 1024
        manifest["files"][str(decade)] = {
            "file": filename,
            "count": len(data),
            "size_kb": round(size_kb, 1),
        }
        manifest["facets"]["decades"].append(decade)
        for rec in data:
            manifest["facets"]["years"].add(rec["year"])

        print(f"  {filename}: {len(data):,} sentences ({size_kb:.0f} KB)")

    manifest["facets"]["years"] = sorted(manifest["facets"]["years"])

    with open(output_dir / "index.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Total: {manifest['total_sentences']:,} sentences")
    return manifest


if __name__ == "__main__":
    from .domain_tagger import DomainTagger

    tagger = DomainTagger()
    extract_from_parlparse(
        debates_dir="data/parlparse/debates",
        output_dir="web/data_full",
        domain_tagger=tagger,
    )
