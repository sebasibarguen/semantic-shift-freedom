# ABOUTME: CLI for the LLM council pipeline.
# ABOUTME: Build sample → submit to enabled providers → adjudicate → persist.

"""
Run the LLM council to build a gold-standard reference dataset.

Usage:
    # 50-sentence pilot (Claude only, ~$0.15)
    uv run python -m src.classify_council --pilot

    # Full 5K sample with all enabled providers (~$35 with all 3)
    uv run python -m src.classify_council --build-sample
    uv run python -m src.classify_council --run

Pipeline:
    1. Build sample (deterministic seed=42, written to outputs/council/sample.json)
    2. Submit one batch per enabled provider
    3. Wait for all, collect, persist raw outputs per provider
    4. Adjudicate: 3/3 → gold, 2/3 → silver, no-majority → disputed (kept aside)
    5. Write outputs/council/{gold,silver,disputed,summary}.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.council.council import run_council
from src.council.sample import build_sample
from src.council.providers.claude import ClaudeProvider
from src.council.providers.gpt import GPTProvider
from src.council.providers.gemini import GeminiProvider
from src.council.providers.base import BaseCouncilProvider

load_dotenv()


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "web" / "data"
ANCHOR_PATH = PROJECT_ROOT / "outputs" / "opus_vs_haiku.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "council"


def enabled_providers(only: list[str] | None = None) -> list[BaseCouncilProvider]:
    """Instantiate the providers we have keys for, optionally filtered."""
    providers: list[BaseCouncilProvider] = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append(ClaudeProvider())
    if os.environ.get("OPENAI_API_KEY"):
        providers.append(GPTProvider())
    if os.environ.get("GOOGLE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        providers.append(GeminiProvider())
    if only:
        providers = [p for p in providers if p.name in only]
    if not providers:
        sys.exit("No providers enabled. Set ANTHROPIC_API_KEY (and OPENAI_API_KEY, GOOGLE_AI_API_KEY) in .env.")
    return providers


def cmd_build_sample(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        result = build_sample(
            DATA_DIR,
            anchor_path=ANCHOR_PATH if ANCHOR_PATH.exists() else None,
            per_decade_label=1,
            n_disagreement=10,
            n_freedom_of_x=10,
            n_random=15,
            n_short_tail=10,
            seed=args.seed,
        )
    else:
        result = build_sample(
            DATA_DIR,
            anchor_path=ANCHOR_PATH if ANCHOR_PATH.exists() else None,
            seed=args.seed,
        )
    out_path = OUTPUT_DIR / ("pilot_sample.json" if args.pilot else "sample.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}: n={result['n']}")
    print(f"  Reasons:        {result['reason_counts']}")
    print(f"  Decades (top):  {dict(list(result['decade_counts'].items())[:6])}")
    print(f"  Haiku labels:   {result['haiku_label_counts']}")


def cmd_run(args):
    sample_path = OUTPUT_DIR / ("pilot_sample.json" if args.pilot else "sample.json")
    if not sample_path.exists():
        sys.exit(f"Sample not found at {sample_path}. Run --build-sample first.")
    sample_doc = json.loads(sample_path.read_text())
    sample = sample_doc["records"]

    providers = enabled_providers(only=args.only.split(",") if args.only else None)
    print(f"Enabled providers: {[p.name for p in providers]}")

    sub_out_dir = OUTPUT_DIR / ("pilot" if args.pilot else "full")
    run_council(sample, providers, sub_out_dir)


def cmd_collect(args):
    """Collect from existing batch state files without re-submitting.

    Use after a crash mid-pipeline, or when re-running adjudication.
    """
    import threading
    from dataclasses import asdict
    from src.council.adjudicate import adjudicate_all

    sample_path = OUTPUT_DIR / ("pilot_sample.json" if args.pilot else "sample.json")
    if not sample_path.exists():
        sys.exit(f"Sample not found at {sample_path}.")
    sample_doc = json.loads(sample_path.read_text())
    sample = sample_doc["records"]

    sub_out_dir = OUTPUT_DIR / ("pilot" if args.pilot else "full")
    providers = enabled_providers(only=args.only.split(",") if args.only else None)

    # Load saved state for each provider
    handles: dict[str, str] = {}
    surviving_providers = []
    for p in providers:
        state_path = sub_out_dir / f".batch_state_{p.name}.json"
        if not state_path.exists():
            print(f"[{p.name}] no batch state at {state_path}, skipping")
            continue
        state = json.loads(state_path.read_text())
        # Restore submitted_ids on the provider so collect() can map back
        p._submitted_ids = state.get("submitted_ids", [r["id"] for r in sample])
        handles[p.name] = state.get("batch_id") or state.get("batch_name")
        surviving_providers.append(p)
        print(f"[{p.name}] resuming handle={handles[p.name]}")

    if not surviving_providers:
        sys.exit("No saved batch state files found; nothing to collect.")

    # Collect each provider's results in parallel
    results: dict[str, list] = {}
    errors: dict[str, Exception] = {}

    def _collect(provider, handle):
        try:
            results[provider.name] = provider.collect(handle)
        except Exception as e:
            errors[provider.name] = e
            print(f"[{provider.name}] ERROR during collect: {e}")

    threads = [threading.Thread(target=_collect, args=(p, handles[p.name]), daemon=False)
               for p in surviving_providers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        print(f"Some providers failed to collect: {list(errors.keys())}")

    # Persist whatever we got
    for name, outs in results.items():
        path = sub_out_dir / f"labels_{name}.json"
        path.write_text(json.dumps([asdict(o) for o in outs], indent=2))
        print(f"Wrote {path} ({len(outs)} labels)")

    # Adjudicate across whatever providers succeeded
    if not results:
        sys.exit("Nothing to adjudicate — all providers failed.")
    adjudicated = adjudicate_all(sample, results)
    summary = adjudicated["summary"]
    print()
    print("=" * 70)
    print("COUNCIL ADJUDICATION (collected)")
    print("=" * 70)
    print(f"  Providers : {summary['providers']}")
    print(f"  Gold      : {summary['n_gold']:>5} ({summary['gold_rate']*100:.1f}%)")
    print(f"  Silver    : {summary['n_silver']:>5} ({summary['silver_rate']*100:.1f}%)")
    print(f"  Disputed  : {summary['n_disputed']:>5} ({summary['disputed_rate']*100:.1f}%)")
    print(f"  Gold label distribution: {summary['label_distribution_gold']}")
    if summary["n_silver"]:
        print(f"  Silver label distribution: {summary['label_distribution_silver']}")

    (sub_out_dir / "gold.json").write_text(json.dumps(adjudicated["gold"], indent=2))
    (sub_out_dir / "silver.json").write_text(json.dumps(adjudicated["silver"], indent=2))
    (sub_out_dir / "disputed.json").write_text(json.dumps(adjudicated["disputed"], indent=2))
    (sub_out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_mutually_exclusive_group(required=True)
    sub.add_argument("--build-sample", action="store_true", help="Construct the sentence sample")
    sub.add_argument("--run", action="store_true", help="Run council on the prepared sample")
    sub.add_argument("--pilot", action="store_true", help="Shortcut: build pilot sample AND run, in one shot")
    sub.add_argument("--collect", action="store_true",
                     help="Collect from existing batch state files (no re-submit). Use --pilot to target the pilot dir.")
    p.add_argument("--pilot-dir", action="store_true", dest="pilot",
                   help="When used with --collect or --run, target the pilot subdir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", type=str, default="", help="Comma-separated provider names to include")
    args = p.parse_args()

    if args.collect:
        cmd_collect(args)
    elif args.pilot:
        # In pilot mode, do both steps in sequence.
        cmd_build_sample(args)
        cmd_run(args)
    elif args.build_sample:
        args.pilot = False
        cmd_build_sample(args)
    elif args.run:
        args.pilot = False
        cmd_run(args)


if __name__ == "__main__":
    main()
