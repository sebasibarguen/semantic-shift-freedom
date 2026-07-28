# ABOUTME: Scores human validation labels against each other, Haiku, and the council.
# ABOUTME: Reports chance-corrected agreement (kappa), not just raw %, plus the human trend.

"""
Score human annotations of the liberty validation set.

Inputs are per-annotator label files exported from the browser tool
(`Export JSON` → {id: {label, ...}}), plus the answer key written by
`src.sample_annotation_set`.

    uv run python -m src.score_annotations \\
        --answer-key outputs/validation_answer_key.json \\
        --labels alice.json bob.json --names alice bob

What it reports, and why each matters:

    inter-annotator kappa   how reliable the human construct is at all — the ceiling
                            on any model score. Raw % agreement hides class imbalance.
    Haiku vs human          the honest classifier accuracy, on hard cases included
    council vs human        whether the LLM "gold" tracks people, by tier
    human positive-share    the trend recomputed on human labels (random subset only,
                            since that is the only representative draw)
"""

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from .council.schema import LABEL_VALUES
from .liberty_trends import weighted_linear_trend, wilson_interval


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa for two raters over paired labels. None if undefined."""
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({lab for pair in pairs for lab in pair})
    po = sum(1 for a, b in pairs if a == b) / n
    count_a = Counter(a for a, _ in pairs)
    count_b = Counter(b for _, b in pairs)
    pe = sum((count_a[lab] / n) * (count_b[lab] / n) for lab in labels)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(item_label_counts: list[dict[str, int]]) -> float | None:
    """Fleiss' kappa. Each item is a {label: count} dict; every item must have the
    same total number of ratings. None if undefined (e.g. <2 raters)."""
    items = [c for c in item_label_counts if sum(c.values()) > 0]
    if not items:
        return None
    n_raters = sum(items[0].values())
    if n_raters < 2 or any(sum(c.values()) != n_raters for c in items):
        return None
    labels = sorted({lab for c in items for lab in c})
    N = len(items)

    p_j = {lab: sum(c.get(lab, 0) for c in items) / (N * n_raters) for lab in labels}
    P_bar = sum(
        (sum(c.get(lab, 0) ** 2 for lab in labels) - n_raters) / (n_raters * (n_raters - 1))
        for c in items
    ) / N
    P_e = sum(v ** 2 for v in p_j.values())
    if P_e >= 1.0:
        return 1.0 if P_bar >= 1.0 else 0.0
    return (P_bar - P_e) / (1 - P_e)


def confusion_and_prf(predicted: dict[str, str], gold: dict[str, str]) -> dict:
    """Agreement, kappa, per-class P/R/F1, confusion matrix over shared ids."""
    ids = [i for i in predicted if i in gold]
    pairs = [(predicted[i], gold[i]) for i in ids]
    n = len(pairs)
    agreement = sum(1 for p, g in pairs if p == g) / n if n else None

    cm: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p, g in pairs:
        cm[p][g] += 1

    per_class = {}
    for label in LABEL_VALUES:
        tp = cm[label].get(label, 0)
        fp = sum(cm[label].get(o, 0) for o in LABEL_VALUES if o != label)
        fn = sum(cm[o].get(label, 0) for o in LABEL_VALUES if o != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    return {
        "n": n,
        "agreement": round(agreement, 4) if agreement is not None else None,
        "cohen_kappa": round(cohen_kappa(pairs), 4) if n else None,
        "per_class": per_class,
        "confusion_matrix": {row: dict(cols) for row, cols in cm.items()},
    }


def load_annotator_file(path: Path) -> dict[str, str]:
    """Load {id: {label,...}} export → {id: label}, keeping only the four valid labels."""
    raw = json.loads(Path(path).read_text())
    out: dict[str, str] = {}
    for sid, entry in raw.items():
        label = entry.get("label") if isinstance(entry, dict) else entry
        if label in LABEL_VALUES:
            out[sid] = label
    return out


def load_context_flags(path: Path) -> set[str]:
    """Ids this annotator opened the surrounding sentences for before labeling."""
    raw = json.loads(Path(path).read_text())
    return {sid for sid, entry in raw.items()
            if isinstance(entry, dict) and entry.get("used_context")}


def human_consensus(annotators: dict[str, dict[str, str]]) -> tuple[dict[str, str], dict[str, int]]:
    """Majority label per id across annotators. Strict ties are dropped (no consensus)."""
    by_id: dict[str, list[str]] = defaultdict(list)
    for labels in annotators.values():
        for sid, lab in labels.items():
            by_id[sid].append(lab)

    consensus: dict[str, str] = {}
    stats = {"agreed": 0, "tie": 0, "single": 0}
    for sid, labs in by_id.items():
        counts = Counter(labs)
        top, top_n = counts.most_common(1)[0]
        if len(labs) == 1:
            consensus[sid] = top
            stats["single"] += 1
        elif sum(1 for _, c in counts.items() if c == top_n) > 1:
            stats["tie"] += 1  # no strict majority → excluded
        else:
            consensus[sid] = top
            stats["agreed"] += 1
    return consensus, stats


def positive_share_trend(labels: dict[str, str], key_by_id: dict[str, dict]) -> dict | None:
    """Per-decade positive/(positive+negative) with Wilson CIs + weighted trend."""
    counts: dict[int, Counter] = defaultdict(Counter)
    for sid, lab in labels.items():
        year = key_by_id.get(sid, {}).get("year")
        if year is None:
            continue
        counts[(int(year) // 10) * 10][lab] += 1

    by_decade = {}
    points = []
    for decade in sorted(counts):
        pos = counts[decade]["positive_liberty"]
        neg = counts[decade]["negative_liberty"]
        denom = pos + neg
        ci = wilson_interval(pos, denom)
        by_decade[str(decade)] = {"positive": pos, "negative": neg, "denom": denom, "share": ci}
        if denom > 0:
            points.append((decade, ci["point"], denom))

    return {
        "by_decade": by_decade,
        "trend": weighted_linear_trend(points),
        "n_labeled": len(labels),
    }


def per_annotator_vs_gold(keys: dict, annotators: dict[str, dict[str, str]]) -> dict:
    """Each annotator's accuracy on the council gold-tier items (attention check).

    Gold items are 3/3 unanimous across the council, so they function as
    known-answer checks: an annotator who scores far below the others on gold
    is likely rushing or misunderstanding the rubric. Answers never reach the
    browser — this is computed only here, at scoring time.
    """
    gold = {sid: info["council_gold"] for sid, info in keys.items()
            if info.get("council_tier") == "gold"
            and info.get("council_gold") in LABEL_VALUES}
    out = {"n_gold_items": len(gold), "by_annotator": {}}
    for name, labels in annotators.items():
        shared = [sid for sid in gold if sid in labels]
        pairs = [(labels[sid], gold[sid]) for sid in shared]
        agree = sum(1 for p, g in pairs if p == g) / len(pairs) if pairs else None
        out["by_annotator"][name] = {
            "n_gold_labeled": len(pairs),
            "agreement": round(agree, 4) if agree is not None else None,
            "cohen_kappa": round(cohen_kappa(pairs), 4) if pairs else None,
        }
    return out


def haiku_by_context_use(haiku: dict[str, str], consensus: dict[str, str],
                         used_context: set[str]) -> dict:
    """Split Haiku-vs-human by whether the human read the surrounding sentences.

    Haiku only ever sees the single sentence. Where the human needed context to
    decide, the two are not answering the same question, so that subset is a
    floor on Haiku rather than a fair test of it.
    """
    out = {}
    for name, ids in (("no_context", set(consensus) - used_context),
                      ("used_context", set(consensus) & used_context)):
        subset = {i: haiku[i] for i in ids if i in haiku}
        out[name] = confusion_and_prf(subset, {i: consensus[i] for i in subset})
    return out


def score(answer_key: dict, annotators: dict[str, dict[str, str]],
          used_context: set[str] | None = None) -> dict:
    keys = answer_key["keys"]
    valid_ids = set(keys)

    # Restrict every annotator to the validation set.
    annotators = {
        name: {sid: lab for sid, lab in labels.items() if sid in valid_ids}
        for name, labels in annotators.items()
    }

    coverage = {
        name: {"labeled": len(labels), "of": len(valid_ids)}
        for name, labels in annotators.items()
    }

    # Inter-annotator reliability
    pairwise = {}
    for a, b in combinations(sorted(annotators), 2):
        shared = [i for i in annotators[a] if i in annotators[b]]
        pairs = [(annotators[a][i], annotators[b][i]) for i in shared]
        pairwise[f"{a}__{b}"] = {
            "n_shared": len(pairs),
            "agreement": round(sum(1 for x, y in pairs if x == y) / len(pairs), 4) if pairs else None,
            "cohen_kappa": round(cohen_kappa(pairs), 4) if pairs else None,
        }

    fleiss = None
    if len(annotators) >= 2:
        all_ids = set.intersection(*(set(labs) for labs in annotators.values()))
        item_counts = [
            Counter(annotators[name][sid] for name in annotators)
            for sid in all_ids
        ]
        fk = fleiss_kappa([dict(c) for c in item_counts])
        fleiss = {"n_items_all_rated": len(all_ids), "fleiss_kappa": round(fk, 4) if fk is not None else None}

    consensus, consensus_stats = human_consensus(annotators)

    haiku = {sid: keys[sid]["haiku_label"] for sid in consensus
             if keys[sid].get("haiku_label") in LABEL_VALUES}
    council = {sid: keys[sid]["council_gold"] for sid in consensus
               if keys[sid].get("council_gold") in LABEL_VALUES}

    # Council vs human, broken out by the tier the council assigned.
    by_tier = {}
    for tier in ("gold", "silver"):
        tier_ids = {sid for sid in council if keys[sid].get("council_tier") == tier}
        if tier_ids:
            by_tier[tier] = confusion_and_prf(
                {i: council[i] for i in tier_ids},
                {i: consensus[i] for i in tier_ids},
            )

    # Trend: only the representative random draw is corpus-like.
    random_consensus = {sid: lab for sid, lab in consensus.items()
                        if keys[sid].get("sample_reason") == "random"}
    random_haiku = {sid: keys[sid]["haiku_label"] for sid in random_consensus
                    if keys[sid].get("haiku_label") in LABEL_VALUES}

    return {
        "n_validation_set": len(valid_ids),
        "annotators": sorted(annotators),
        "coverage": coverage,
        "inter_annotator": {"pairwise": pairwise, "fleiss": fleiss},
        "human_consensus": {"n": len(consensus), **consensus_stats},
        "per_annotator_vs_gold": per_annotator_vs_gold(keys, annotators),
        "haiku_vs_human": confusion_and_prf(haiku, consensus),
        "haiku_vs_human_by_context": haiku_by_context_use(haiku, consensus, used_context or set()),
        "council_vs_human": confusion_and_prf(council, consensus),
        "council_vs_human_by_tier": by_tier,
        "trend_random_subset": {
            "human": positive_share_trend(random_consensus, keys),
            "haiku": positive_share_trend(random_haiku, keys),
        },
    }


def print_summary(result: dict) -> None:
    print("=" * 70)
    print("HUMAN VALIDATION")
    print("=" * 70)
    print(f"Validation set: {result['n_validation_set']}   annotators: {', '.join(result['annotators'])}")
    for name, cov in result["coverage"].items():
        print(f"  {name:<14} labeled {cov['labeled']}/{cov['of']}")

    print("\nInter-annotator reliability (kappa, not raw %):")
    for pair, m in result["inter_annotator"]["pairwise"].items():
        print(f"  {pair:<24} n={m['n_shared']:<4} agree={m['agreement']}  kappa={m['cohen_kappa']}")
    fl = result["inter_annotator"]["fleiss"]
    if fl:
        print(f"  Fleiss (all raters)      n={fl['n_items_all_rated']:<4} kappa={fl['fleiss_kappa']}")

    cs = result["human_consensus"]
    print(f"\nHuman consensus: {cs['n']} ids  (agreed {cs['agreed']}, single {cs['single']}, tie/dropped {cs['tie']})")

    pag = result["per_annotator_vs_gold"]
    print(f"\nPer-annotator accuracy on {pag['n_gold_items']} council-gold items (attention check):")
    for name, m in sorted(pag["by_annotator"].items()):
        print(f"  {name:<14} labeled {m['n_gold_labeled']:<4} agree={m['agreement']} kappa={m['cohen_kappa']}")

    def show(title, m):
        print(f"\n{title}: n={m['n']} agree={m['agreement']} kappa={m['cohen_kappa']}")
        pos = m["per_class"].get("positive_liberty", {})
        print(f"  positive_liberty  P={pos.get('precision')} R={pos.get('recall')} "
              f"F1={pos.get('f1')} (support {pos.get('support')})")

    show("Haiku vs human", result["haiku_vs_human"])
    ctx = result["haiku_vs_human_by_context"]
    print("  split by whether the human opened the surrounding sentences "
          "(Haiku only sees the sentence):")
    for subset in ("no_context", "used_context"):
        m = ctx[subset]
        print(f"    {subset:<13} n={m['n']} agree={m['agreement']} kappa={m['cohen_kappa']}")
    show("Council-gold vs human", result["council_vs_human"])
    for tier, m in result["council_vs_human_by_tier"].items():
        print(f"    {tier:<8} n={m['n']} agree={m['agreement']} kappa={m['cohen_kappa']}")

    print("\nPositive-share trend on the RANDOM subset (representative):")
    for who in ("human", "haiku"):
        t = result["trend_random_subset"][who]["trend"]
        n = result["trend_random_subset"][who]["n_labeled"]
        if t:
            print(f"  {who:<6} n={n:<4} slope/century={t['slope_per_century']:+.4f} "
                  f"(first={t['first_proportion']:.3f} last={t['last_proportion']:.3f})")
        else:
            print(f"  {who:<6} n={n:<4} insufficient data for a trend")


def main() -> None:
    project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--answer-key", type=Path,
                        default=project_root / "outputs" / "validation_answer_key.json")
    parser.add_argument("--labels", type=Path, nargs="+", required=True,
                        help="Per-annotator exported JSON label files")
    parser.add_argument("--names", type=str, nargs="*",
                        help="Names for each --labels file (defaults to file stems)")
    parser.add_argument("--output", type=Path,
                        default=project_root / "outputs" / "human_validation.json")
    args = parser.parse_args()

    names = args.names or [p.stem for p in args.labels]
    if len(names) != len(args.labels):
        parser.error("--names must have one entry per --labels file")

    answer_key = json.loads(args.answer_key.read_text())
    annotators = {name: load_annotator_file(path) for name, path in zip(names, args.labels)}
    used_context = set().union(*(load_context_flags(p) for p in args.labels))

    result = score(answer_key, annotators, used_context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print_summary(result)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
