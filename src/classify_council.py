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
    # GPT and Gemini providers will be added in PR2.
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


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_mutually_exclusive_group(required=True)
    sub.add_argument("--build-sample", action="store_true", help="Construct the sentence sample")
    sub.add_argument("--run", action="store_true", help="Run council on the prepared sample")
    sub.add_argument("--pilot", action="store_true", help="Shortcut: build pilot sample AND run, in one shot")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", type=str, default="", help="Comma-separated provider names to include")
    args = p.parse_args()

    if args.pilot:
        # In pilot mode, do both steps in sequence.
        args.pilot = True
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
