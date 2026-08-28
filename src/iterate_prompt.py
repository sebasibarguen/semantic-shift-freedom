# ABOUTME: CLI for the prompt-iteration arena.
# ABOUTME: Hill-climb a small model toward council-gold agreement on dev; lock test for final eval.

"""
Iterate on prompts for the small classifier.

Usage:
    # See what splits look like and confirm the gold dataset is there
    uv run python -m src.iterate_prompt status

    # Evaluate a candidate prompt on dev (hill-climbing target)
    uv run python -m src.iterate_prompt eval \\
        --prompt prompts/haiku_v3.txt --split dev

    # Lock-in evaluation on the test split (do this once, at the end)
    uv run python -m src.iterate_prompt eval \\
        --prompt prompts/haiku_v3.txt --split test --no-cache

    # Show history sorted by dev score
    uv run python -m src.iterate_prompt history

The arena targets agreement with council gold (not Opus alone). The test
split is locked — only touch it for the final score, never during iteration.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.prompt_arena.evaluator import evaluate_prompt
from src.prompt_arena.history import history_table
from src.prompt_arena.splits import (
    assign_split,
    load_gold,
    split_gold,
    split_summary,
)

load_dotenv()


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_GOLD_DIR = PROJECT_ROOT / "outputs" / "council" / "full"
PILOT_GOLD_DIR = PROJECT_ROOT / "outputs" / "council" / "pilot"
HISTORY_PATH = PROJECT_ROOT / "outputs" / "prompts" / "history.jsonl"


def _resolve_gold_dir(args) -> Path:
    """Pick the gold directory: explicit --gold-dir, --pilot-gold, or default to full."""
    if getattr(args, "gold_dir", None):
        return Path(args.gold_dir)
    if getattr(args, "pilot_gold", False):
        return PILOT_GOLD_DIR
    return DEFAULT_GOLD_DIR


def _load_records(args):
    gold_dir = _resolve_gold_dir(args)
    gold_path = gold_dir / "gold.json"
    silver_path = gold_dir / "silver.json"
    if not gold_path.exists():
        sys.exit(
            f"No gold dataset at {gold_path}. Run the council first:\n"
            "  uv run python -m src.classify_council --build-sample\n"
            "  uv run python -m src.classify_council --run"
        )
    print(f"Loading gold from: {gold_dir}")
    return load_gold(gold_path, include_silver=getattr(args, "include_silver", False), silver_path=silver_path)


def cmd_status(args):
    records = _load_records(args)
    summary = split_summary(records)
    print(f"Total records: {len(records)}")
    print(f"Split sizes:   {summary}")
    print(f"History:       {HISTORY_PATH}")
    print()
    print("Label distribution by split:")
    from collections import Counter
    for split in ("train", "dev", "test"):
        sub = [r for r in records if assign_split(r["id"]) == split]
        c = Counter(r["gold_label"] for r in sub)
        print(f"  {split:<6} ({len(sub):>4}): {dict(c)}")


def cmd_eval(args):
    records = _load_records(args)
    if args.split == "all":
        sys.exit("--split must be one of train, dev, test")
    subset = split_gold(records, args.split)
    if not subset:
        sys.exit(f"No records in split={args.split}")

    if args.split == "test" and args.use_cache:
        # Force the user to think before re-running test
        print("WARNING: evaluating on test split. Use --no-cache only for the final eval.")

    print(f"Evaluating prompt: {args.prompt}")
    print(f"  split={args.split}  n={len(subset)}  model={args.model}")
    print(f"  cache={'on' if args.use_cache else 'off'}")
    print()

    result = evaluate_prompt(
        prompt_path=Path(args.prompt),
        records=subset,
        split=args.split,
        model=args.model,
        history_path=HISTORY_PATH,
        use_cache=args.use_cache,
        sample_errors=args.sample_errors,
    )

    if result.cached:
        print("(cached result from history.jsonl)")
    print(f"Accuracy:        {result.accuracy*100:.2f}%   (skipped: {result.n_skipped})")
    print()
    print("Per-class F1:")
    for label, f1 in sorted(result.per_class_f1.items(), key=lambda kv: -kv[1]):
        p = result.per_class_precision.get(label, 0)
        r = result.per_class_recall.get(label, 0)
        print(f"  {label:<20} P={p:.2f}  R={r:.2f}  F1={f1:.2f}")
    print()
    print("Confusion matrix (rows=predicted, cols=gold):")
    labels = sorted(result.per_class_f1.keys())
    header = "pred\\gold".ljust(22) + "  ".join(lbl[:8].ljust(8) for lbl in labels)
    print(header)
    for row in labels:
        cells = [str(result.confusion_matrix.get(row, {}).get(col, 0)).rjust(8) for col in labels]
        print(row[:20].ljust(22) + "  ".join(cells))


def cmd_history(args):
    rows = history_table(HISTORY_PATH)
    if not rows:
        print(f"No evaluations logged at {HISTORY_PATH}")
        return
    rows = [r for r in rows if r.get("split") == args.split]
    rows.sort(key=lambda r: r.get("accuracy", 0), reverse=True)

    print(f"split={args.split}  ({len(rows)} evals)")
    print(f"{'Acc':>6}  {'F1+':>5}  {'F1-':>5}  {'F1amb':>5}  {'F1oth':>5}  prompt")
    for r in rows[:args.limit]:
        f1 = r.get("per_class_f1", {})
        print(f"{r.get('accuracy', 0)*100:>5.1f}%  "
              f"{f1.get('positive_liberty', 0):>5.2f}  "
              f"{f1.get('negative_liberty', 0):>5.2f}  "
              f"{f1.get('ambiguous', 0):>5.2f}  "
              f"{f1.get('other', 0):>5.2f}  "
              f"{r.get('prompt_path', r.get('prompt_hash'))}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_status = sub.add_parser("status", help="Show split sizes and gold availability")
    sp_status.add_argument("--include-silver", action="store_true")
    sp_status.add_argument("--pilot-gold", action="store_true", help="Use the pilot gold dir instead of full")
    sp_status.add_argument("--gold-dir", help="Explicit path to council gold directory")
    sp_status.set_defaults(func=cmd_status)

    sp_eval = sub.add_parser("eval", help="Evaluate a prompt on a split")
    sp_eval.add_argument("--prompt", required=True, help="Path to prompt text file")
    sp_eval.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    sp_eval.add_argument("--model", default="claude-haiku-4-5")
    sp_eval.add_argument("--include-silver", action="store_true",
                         help="Include silver labels (only for train; not recommended for dev/test)")
    sp_eval.add_argument("--pilot-gold", action="store_true", help="Use the pilot gold dir instead of full")
    sp_eval.add_argument("--gold-dir", help="Explicit path to council gold directory")
    sp_eval.add_argument("--no-cache", dest="use_cache", action="store_false", default=True)
    sp_eval.add_argument("--sample-errors", type=int, default=30)
    sp_eval.set_defaults(func=cmd_eval)

    sp_hist = sub.add_parser("history", help="Show past evaluations sorted by accuracy")
    sp_hist.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    sp_hist.add_argument("--limit", type=int, default=20)
    sp_hist.set_defaults(func=cmd_history)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
