# ABOUTME: Rewrites an exported annotator-labels JSON ({old_id: label}) onto re-minted sentence ids.
# ABOUTME: Uses outputs/id_remap.json from src.remint_ids; ids it cannot place unambiguously are reported, not guessed.

import argparse
import json
from pathlib import Path


def remap_labels(labels: dict, remap: dict) -> tuple[dict, list[str]]:
    validation, corpus = remap["validation"], remap["corpus"]
    out, ambiguous = {}, []
    for old, label in labels.items():
        new = validation.get(old)
        if new is None:
            cand = corpus.get(old, old)
            if isinstance(cand, list):
                ambiguous.append(old)
                continue
            new = cand
        out[new] = label
    return out, ambiguous


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels", type=Path, nargs="+", help="exported labels JSON, one per annotator")
    ap.add_argument("--remap", type=Path, default=Path("outputs/id_remap.json"))
    args = ap.parse_args()
    remap = json.loads(args.remap.read_text())
    for path in args.labels:
        new, ambiguous = remap_labels(json.loads(path.read_text()), remap)
        out = path.with_name(path.stem + "_remapped.json")
        out.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
        print(f"{path} → {out}: {len(new)} labels kept, {len(ambiguous)} dropped as ambiguous")
        for old in ambiguous:
            print(f"    ambiguous (browse-mode label on a shared old id): {old}")


if __name__ == "__main__":
    main()
