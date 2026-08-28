# ABOUTME: Recovers surrounding-sentence context for annotation sentences from their source documents.
# ABOUTME: Re-downloads Hansard archive volumes (pre-1919) and ParlParse day files (1919+) and matches by sentence text.

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
from xml.etree.ElementTree import iterparse
from zipfile import ZipFile

from src.hansard_archive_extractor import SENTENCE_RE, get_element_text
from src.parlparse_extractor import extract_text

ARCHIVE_URL_LIST = "https://raw.githubusercontent.com/econandrew/uk-hansard-archive-urls/master/urls.txt"
PARLPARSE_BASE = "https://www.theyworkforyou.com/pwdata/scrapedxml/debates"
PARLPARSE_SUFFIXES = ("a", "b", "c", "d")


def normalize(text):
    """Collapse whitespace so stored sentences match freshly parsed source text."""
    return re.sub(r"\s+", " ", text).strip()


def load_corpus_index(data_dir):
    """Map sentence id -> record for every decade file."""
    index = {}
    for path in sorted(data_dir.glob("sentences_*.json")):
        for record in json.loads(path.read_text()):
            index[record["id"]] = record
    return index


def group_by_source(records, corpus_index):
    """Split annotation records into archive-volume and parlparse-date buckets.

    Pre-1919 sentences carry a `source_file` naming a Hansard archive volume.
    ParlParse-derived sentences have no `source_file`; their day file is keyed
    by date.
    """
    archive = defaultdict(list)
    parlparse = defaultdict(list)
    unresolved = []
    for record in records:
        source = corpus_index.get(record["id"])
        if source is None:
            unresolved.append(record["id"])
            continue
        volume = source.get("source_file")
        if volume:
            archive[volume].append(record)
        elif source.get("date"):
            parlparse[source["date"]].append(record)
        else:
            unresolved.append(record["id"])
    return archive, parlparse, unresolved


def fetch(url, dest):
    """Download to dest unless already cached. Returns False when unavailable."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urlopen(url, timeout=120) as resp:
            tmp.write_bytes(resp.read())
    except Exception:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    return True


def archive_urls(cache_dir):
    """Volume stem -> zip URL, from the published archive link list."""
    listing = cache_dir / "hansard_archive_urls.txt"
    if not listing.exists() and not fetch(ARCHIVE_URL_LIST, listing):
            raise RuntimeError(f"Could not download archive URL list: {ARCHIVE_URL_LIST}")
    urls = {}
    for line in listing.read_text().splitlines():
        line = line.strip()
        if line.endswith(".zip"):
            urls[line.rsplit("/", 1)[-1][: -len(".zip")]] = line.replace("http://", "https://")
    return urls


def archive_documents(xml_path):
    """Yield (speaker, contribution_text) for each contribution in a volume."""
    for _event, elem in iterparse(str(xml_path), events=("end",)):
        if elem.tag != "p":
            continue
        contribution = elem.find("membercontribution")
        if contribution is not None:
            member = elem.find("member")
            speaker = get_element_text(member) if member is not None else "Unknown"
            text = get_element_text(contribution)
        else:
            speaker, text = "Unknown", get_element_text(elem)
        if text:
            yield speaker, text
        elem.clear()


def parlparse_documents(xml_path):
    """Yield (speaker, speech_text) for each speech in a ParlParse day file."""
    for _event, elem in iterparse(str(xml_path), events=("end",)):
        if elem.tag != "speech":
            continue
        speaker = re.sub(r"\s*\(.*?\)\s*$", "", elem.get("speakername", "Unknown")).strip()
        paragraphs = [t for t in (extract_text(p) for p in elem.findall("p")) if t]
        text = " ".join(paragraphs)
        if text:
            yield speaker or "Unknown", text
        elem.clear()


def contexts_from_document(documents, wanted, window):
    """Find each wanted sentence in a document stream and slice its neighbours.

    `wanted` maps normalized sentence text -> sentence id. Matching on text
    rather than the original id hash keeps this independent of how speaker
    names were cleaned when the corpus was first built.
    """
    found = {}
    for _speaker, text in documents:
        normalized_text = normalize(text)
        candidates = [s for s in wanted if s in normalized_text]
        if not candidates:
            continue
        sentences = [normalize(m.group(0)) for m in SENTENCE_RE.finditer(text)]
        for sentence in candidates:
            if sentence in found:
                continue
            try:
                i = sentences.index(sentence)
            except ValueError:
                continue
            found[sentence] = {
                "before": sentences[max(0, i - window):i],
                "after": sentences[i + 1:i + 1 + window],
            }
        if len(found) == len(wanted):
            break
    return found


def collect(records, corpus_index, cache_dir, window, workers):
    archive, parlparse, unresolved = group_by_source(records, corpus_index)
    print(f"{len(archive)} archive volumes, {len(parlparse)} parlparse dates, "
          f"{len(unresolved)} unresolved", file=sys.stderr)

    urls = archive_urls(cache_dir)
    missing_urls = [v for v in archive if v not in urls]
    if missing_urls:
        print(f"No archive URL for {len(missing_urls)} volumes: {missing_urls[:5]}", file=sys.stderr)

    archive_dir = cache_dir / "archive"
    parlparse_dir = cache_dir / "parlparse"

    jobs = [(urls[v], archive_dir / f"{v}.zip") for v in archive if v in urls]
    jobs += [(f"{PARLPARSE_BASE}/debates{d}{s}.xml", parlparse_dir / f"debates{d}{s}.xml")
             for d in parlparse for s in PARLPARSE_SUFFIXES]

    print(f"Downloading up to {len(jobs)} source files...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda job: fetch(*job), jobs))

    contexts = {}

    for volume, group in sorted(archive.items()):
        zip_path = archive_dir / f"{volume}.zip"
        if not zip_path.exists():
            continue
        wanted = {normalize(r["sentence"]): r["id"] for r in group}
        try:
            with ZipFile(zip_path) as zf:
                names = [n for n in zf.namelist() if n.endswith(".xml")]
                if not names:
                    continue
                extracted = zf.extract(names[0], archive_dir)
        except Exception as e:
            print(f"  {volume}: unreadable ({e})", file=sys.stderr)
            continue
        found = contexts_from_document(archive_documents(extracted), wanted, window)
        Path(extracted).unlink(missing_ok=True)
        for sentence, ctx in found.items():
            contexts[wanted[sentence]] = ctx

    for date, group in sorted(parlparse.items()):
        wanted = {normalize(r["sentence"]): r["id"] for r in group}
        remaining = dict(wanted)
        for suffix in PARLPARSE_SUFFIXES:
            if not remaining:
                break
            xml_path = parlparse_dir / f"debates{date}{suffix}.xml"
            if not xml_path.exists():
                continue
            try:
                found = contexts_from_document(parlparse_documents(xml_path), remaining, window)
            except Exception as e:
                print(f"  {date}{suffix}: parse failed ({e})", file=sys.stderr)
                continue
            for sentence, ctx in found.items():
                contexts[remaining.pop(sentence)] = ctx

    return contexts, unresolved


def fill_from_speech_csv(records, corpus_index, contexts, csv_path, window):
    """Second pass for sentences the XML sources did not yield.

    Modern sentences came from the Hansard speeches CSV, whose sitting-day
    coverage and speech segmentation differ from ParlParse — so the misses
    cluster in recent decades. Left alone that gap leaks era to the annotator,
    which is exactly what blind mode is meant to prevent.
    """
    import polars as pl

    def empty(ctx):
        return not ctx or (not ctx.get("before") and not ctx.get("after"))

    missing = [r for r in records if empty(contexts.get(r["id"]))]
    if not missing:
        return 0
    dates = sorted({corpus_index[r["id"]]["date"] for r in missing
                    if corpus_index.get(r["id"], {}).get("date")})
    print(f"Gap-filling {len(missing)} sentences across {len(dates)} dates from {csv_path.name}",
          file=sys.stderr)

    frame = pl.scan_csv(csv_path, infer_schema_length=0)
    columns = frame.collect_schema().names()
    text_col = "speech" if "speech" in columns else "text"
    date_col = "speech_date" if "speech_date" in columns else "date"
    speeches = (frame
                .select([pl.col(date_col).str.slice(0, 10).alias("date"), pl.col(text_col)])
                .filter(pl.col("date").is_in(dates))
                .collect())
    print(f"  {speeches.height} speeches on those dates", file=sys.stderr)

    wanted = {normalize(r["sentence"]): r["id"] for r in missing}
    documents = (("", text) for text in speeches[text_col] if text)
    found = contexts_from_document(documents, wanted, window)
    filled = 0
    for sentence, ctx in found.items():
        if not empty(ctx):
            contexts[wanted[sentence]] = ctx
            filled += 1
    return filled


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "web" / "data")
    parser.add_argument("--validation-set", type=Path,
                        default=project_root / "web" / "data" / "validation_set.json")
    parser.add_argument("--out", type=Path,
                        default=project_root / "web" / "data" / "validation_context.json")
    parser.add_argument("--cache-dir", type=Path, default=project_root / "data" / "context_sources")
    parser.add_argument("--window", type=int, default=3,
                        help="Sentences to keep on each side")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--speech-csv", type=Path,
                        help="Hansard speeches CSV used to fill gaps the XML sources missed")
    args = parser.parse_args()

    records = json.loads(args.validation_set.read_text())["records"]
    corpus_index = load_corpus_index(args.data_dir)
    contexts, unresolved = collect(records, corpus_index, args.cache_dir, args.window, args.workers)

    if args.speech_csv:
        filled = fill_from_speech_csv(records, corpus_index, contexts,
                                      args.speech_csv, args.window)
        print(f"Gap-fill recovered {filled} more", file=sys.stderr)

    args.out.write_text(json.dumps({
        "meta": {
            "n_records": len(records),
            "n_with_context": len(contexts),
            "window": args.window,
            "unresolved": unresolved,
        },
        "contexts": contexts,
    }))
    print(f"Context for {len(contexts)}/{len(records)} sentences -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
