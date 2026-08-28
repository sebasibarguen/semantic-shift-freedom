# ABOUTME: One-time migration: re-mints every sentence id in web/data with src.ids and re-keys
# ABOUTME: every artifact that joins on those ids. Idempotent; --dry-run reports without writing.

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .ids import IdMinter, split_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "web" / "data"
OUTPUTS = PROJECT_ROOT / "outputs"
COUNCIL = OUTPUTS / "council"

# Artifacts that carry both an id and the sentence text, so each row can be
# re-keyed exactly even where the old id was shared.
RECORD_LISTS = [
    OUTPUTS / "opus_vs_haiku.json",
    OUTPUTS / "haiku_v2_eval.json",
    OUTPUTS / "gemini_vs_haiku_460.json",
    OUTPUTS / "opus_gold_500.json",
    OUTPUTS / "opus_gold_100_reasoned.json",
    *(COUNCIL / sub / f"{tier}.json" for sub in ("full", "pilot") for tier in ("gold", "silver", "disputed")),
]
RECORD_DICTS = [  # {"records": [...]} wrappers
    COUNCIL / "sample.json",
    COUNCIL / "pilot_sample.json",
    DATA_DIR / "validation_set.json",
]


def new_id_for(record: dict, minter: IdMinter) -> str:
    prefix, index = split_id(record["id"])
    return minter.mint(prefix, record.get("speaker", ""), index, record["sentence"])


def load(path: Path):
    return json.loads(path.read_text())


def dump(path: Path, data, dry_run: bool) -> None:
    if not dry_run:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


class Remap:
    """Resolves an old id to its new id, using the sentence text to break ties."""

    def __init__(self):
        self.by_pair: dict[tuple[str, str], str] = {}
        self.by_old: dict[str, set[str]] = defaultdict(set)
        self.unresolved: list[tuple[str, str]] = []

    def learn(self, old: str, sentence: str, new: str) -> None:
        self.by_pair[(old, sentence.strip())] = new
        self.by_old[old].add(new)

    def resolve(self, old: str, sentence: str | None, where: str) -> str:
        if sentence is not None and (old, sentence.strip()) in self.by_pair:
            return self.by_pair[(old, sentence.strip())]
        if len(self.by_old.get(old, ())) == 1:
            return next(iter(self.by_old[old]))
        self.unresolved.append((where, old))
        return old


def remint_corpus(remap: Remap, dry_run: bool) -> tuple[int, int]:
    total = changed = 0
    minter = IdMinter()
    for path in sorted(DATA_DIR.glob("sentences_*s.json")):
        records = load(path)
        for r in records:
            new = new_id_for(r, minter)
            remap.learn(r["id"], r["sentence"], new)
            changed += new != r["id"]
            r["id"] = new
            total += 1
        dump(path, records, dry_run)
    dupes = {k: n for k, n in Counter(new for s in remap.by_old.values() for new in s).items() if n > 1}
    if dupes:
        raise SystemExit(f"re-mint still yields {len(dupes)} duplicate ids; refusing to write: {list(dupes)[:5]}")
    return total, changed


def rekey_records(records: list[dict], remap: Remap, where: str, id_key: str = "id") -> int:
    n = 0
    for r in records:
        new = remap.resolve(r[id_key], r.get("sentence"), where)
        n += new != r[id_key]
        r[id_key] = new
    return n


def rekey_dict(mapping: dict, old_to_new: dict[str, str], where: str, remap: Remap) -> dict:
    out = {}
    for old, v in mapping.items():
        if old not in old_to_new:
            remap.unresolved.append((where, old))
        out[old_to_new.get(old, old)] = v
    return out


def unique_map(records: list[dict], remap: Remap, where: str) -> dict[str, str]:
    """old→new for a record list whose old ids are unique within it (asserted)."""
    olds = Counter(r["_old_id"] for r in records)
    if any(n > 1 for n in olds.values()):
        raise SystemExit(f"{where}: old ids repeat inside the file; cannot re-key its id-only dependents")
    return {r["_old_id"]: r["id"] for r in records}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run
    remap = Remap()

    total, changed = remint_corpus(remap, dry)
    print(f"corpus: {total} records, {changed} ids changed, {len(remap.by_old)} old ids → {sum(len(s) for s in remap.by_old.values())} new ids")

    report = []
    dependents: dict[Path, dict[str, str]] = {}
    for path in RECORD_LISTS + RECORD_DICTS:
        if not path.exists():
            report.append(f"  skip (absent): {path.relative_to(PROJECT_ROOT)}")
            continue
        data = load(path)
        records = data["records"] if isinstance(data, dict) else data
        for r in records:
            r["_old_id"] = r["id"]
        n = rekey_records(records, remap, str(path))
        if path in RECORD_DICTS:
            dependents[path] = unique_map(records, remap, str(path))
        for r in records:
            del r["_old_id"]
        dump(path, data, dry)
        report.append(f"  {path.relative_to(PROJECT_ROOT)}: {n}/{len(records)} re-keyed")

    # Id-only dependents: council label dumps follow their sample; validation
    # context and answer key follow the validation set.
    for sub, sample in (("full", COUNCIL / "sample.json"), ("pilot", COUNCIL / "pilot_sample.json")):
        m = dependents.get(sample, {})
        for path in sorted((COUNCIL / sub).glob("labels_*.json")):
            labels = load(path)
            n = 0
            for row in labels:
                old = row["sentence_id"]
                if old not in m:
                    remap.unresolved.append((str(path), old))
                row["sentence_id"] = m.get(old, old)
                n += row["sentence_id"] != old
            dump(path, labels, dry)
            report.append(f"  {path.relative_to(PROJECT_ROOT)}: {n}/{len(labels)} re-keyed")
    vmap = dependents.get(DATA_DIR / "validation_set.json", {})
    for path, key in ((DATA_DIR / "validation_context.json", "contexts"), (OUTPUTS / "validation_answer_key.json", "keys")):
        if path.exists():
            data = load(path)
            data[key] = rekey_dict(data[key], vmap, str(path), remap)
            dump(path, data, dry)
            report.append(f"  {path.relative_to(PROJECT_ROOT)}: {len(data[key])} keys re-keyed")

    # Persist the map so labels saved under old ids (R2 exports) can follow.
    remap_out = {
        "validation": vmap,
        "corpus": {old: (sorted(s)[0] if len(s) == 1 else sorted(s)) for old, s in remap.by_old.items()},
    }
    dump(OUTPUTS / "id_remap.json", remap_out, dry)

    print("\n".join(report))
    if remap.unresolved:
        by_file = Counter(w for w, _ in remap.unresolved)
        print(f"\nUNRESOLVED ids ({len(remap.unresolved)}):")
        for w, n in by_file.items():
            print(f"  {n:5d}  {Path(w).relative_to(PROJECT_ROOT)}")
        raise SystemExit(1)
    print("\nall ids resolved" + (" (dry run — nothing written)" if dry else ""))


if __name__ == "__main__":
    main()
