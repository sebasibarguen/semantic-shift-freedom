# ABOUTME: Extracts freedom/liberty sentences from Hansard Archive XML (1803-1918).
# ABOUTME: Parses the Parliament archive format with <member>/<membercontribution> tags.

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from xml.etree.ElementTree import iterparse, ParseError


SENTENCE_RE = re.compile(r"[^.!?;]+[.!?;]?", re.DOTALL)
WORD_RE = re.compile(r"[a-z]+")

# Volume naming encodes date ranges: S1V0001 = Series 1 Volume 1 (1803)
# We extract dates from <p> text or infer from volume numbering
SERIES_DATES = {
    "S1": (1803, 1820),  # 1st series
    "S2": (1820, 1830),  # 2nd series
    "S3": (1830, 1891),  # 3rd series
    "S4": (1892, 1908),  # 4th series
    "S5": (1909, 1981),  # 5th series (overlaps with ParlParse at 1919)
}


def extract_text_recursive(elem):
    """Get all text from an element and descendants."""
    texts = []
    if elem.text:
        texts.append(elem.text)
    for child in elem:
        texts.extend(extract_text_recursive(child))
        if child.tail:
            texts.append(child.tail)
    return texts


def get_element_text(elem):
    """Get full text content of an element."""
    return " ".join(extract_text_recursive(elem)).strip()


def infer_year_from_filename(filename):
    """Infer approximate year from Hansard archive filename like S1V0001P0.xml."""
    m = re.match(r"S(\d)V(\d{4})", filename)
    if not m:
        return None
    series = int(m.group(1))
    volume = int(m.group(2))

    series_key = f"S{series}"
    if series_key not in SERIES_DATES:
        return None

    start, end = SERIES_DATES[series_key]
    # Rough linear interpolation within series
    # Series volumes vary but this gives a reasonable estimate
    series_volumes = {1: 41, 2: 25, 3: 356, 4: 199, 5: 1000}
    max_vol = series_volumes.get(series, 100)
    fraction = min(volume / max_vol, 1.0)
    return start + int(fraction * (end - start))


def extract_from_archive(xml_path, domain_tagger=None):
    """Extract freedom/liberty sentences from a single Hansard archive XML."""
    filename = Path(xml_path).stem
    approx_year = infer_year_from_filename(filename)
    if not approx_year:
        return []

    sentences = []
    try:
        for event, elem in iterparse(str(xml_path), events=("end",)):
            if elem.tag != "p":
                continue

            # Look for <member> + <membercontribution> pattern
            member_elem = elem.find("member")
            contrib_elem = elem.find("membercontribution")

            if contrib_elem is not None:
                speaker = get_element_text(member_elem) if member_elem is not None else "Unknown"
                text = get_element_text(contrib_elem)
            else:
                # Some entries are just <p> with text
                text = get_element_text(elem)
                speaker = "Unknown"

            if not text or len(text) < 20:
                elem.clear()
                continue

            text_lower = text.lower()
            if "freedom" not in text_lower and "liberty" not in text_lower:
                elem.clear()
                continue

            # Clean speaker name
            speaker = re.sub(r"\s*[,;.]+\s*$", "", speaker).strip()
            if not speaker:
                speaker = "Unknown"

            # Extract sentences
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

                id_source = f"{approx_year}-{filename}-{speaker}-{i}"
                sid = f"{approx_year}-{hashlib.md5(id_source.encode()).hexdigest()[:6]}-{i:03d}"

                domains = {}
                if domain_tagger:
                    words = WORD_RE.findall(sent_lower)
                    dist = domain_tagger.get_domain_distribution(words)
                    domains = {k: v for k, v in dist.items() if v > 0 and k != "untagged"}

                sentences.append({
                    "id": sid,
                    "sentence": sent,
                    "word": word,
                    "year": approx_year,
                    "date": f"{approx_year}-01-01",
                    "speaker": speaker,
                    "party": "",
                    "person_id": "",
                    "source_file": filename,
                    "methods": {"domains": domains},
                })

            elem.clear()

    except (ParseError, Exception) as e:
        # Some archive files have malformed XML — skip them
        pass

    return sentences


def run_archive_extraction(archive_dir, output_dir, domain_tagger=None):
    """Process all archive XML files and write decade-chunked JSON."""
    archive_dir = Path(archive_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(archive_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} archive XML files")

    sentences_by_decade = defaultdict(list)
    total = 0
    files_processed = 0

    for fpath in xml_files:
        result = extract_from_archive(fpath, domain_tagger)
        for s in result:
            decade = (s["year"] // 10) * 10
            sentences_by_decade[decade].append(s)
            total += 1
        files_processed += 1
        if files_processed % 100 == 0:
            print(f"  Processed {files_processed}/{len(xml_files)} files, {total:,} sentences...")

    print(f"\nDone: {files_processed} files, {total:,} sentences")

    # Write output
    manifest = {"source": "Hansard Archive (Parliament)", "total_sentences": total, "files": {}, "facets": {"decades": [], "years": set()}}

    for decade in sorted(sentences_by_decade.keys()):
        # Only keep pre-1919 (ParlParse covers 1919+)
        if decade >= 1920:
            continue
        data = sentences_by_decade[decade]
        filename = f"sentences_{decade}s.json"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        size_kb = filepath.stat().st_size / 1024
        manifest["files"][str(decade)] = {"file": filename, "count": len(data), "size_kb": round(size_kb, 1)}
        manifest["facets"]["decades"].append(decade)
        for r in data:
            manifest["facets"]["years"].add(r["year"])
        print(f"  {filename}: {len(data):,} sentences ({size_kb:.0f} KB)")

    manifest["facets"]["years"] = sorted(manifest["facets"]["years"])
    with open(output_dir / "index.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Total (pre-1920): {manifest['total_sentences']:,} sentences")
    return manifest
