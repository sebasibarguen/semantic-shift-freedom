# ABOUTME: Council orchestrator — submits one batch per provider, collects in parallel.
# ABOUTME: Works with 1, 2, or 3 providers; pilot uses just Claude.

import json
import threading
from dataclasses import asdict
from pathlib import Path

from .adjudicate import adjudicate_all
from .providers.base import BaseCouncilProvider, CouncilLabel


def run_council(
    sample: list[dict],
    providers: list[BaseCouncilProvider],
    output_dir: Path,
    confidence_floor: float = 0.6,
) -> dict:
    """End-to-end: submit each provider's batch, wait for all, adjudicate, persist."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Council: {len(sample)} sentences × {len(providers)} providers")
    total_estimate = sum(p.estimate_cost(len(sample)) for p in providers)
    print(f"Estimated total cost: ${total_estimate:.2f}")

    # Submit all batches first (parallel by virtue of being separate API calls)
    handles: dict[str, str] = {}
    for p in providers:
        state_path = output_dir / f".batch_state_{p.name}.json"
        handles[p.name] = p.submit(sample, state_path)

    # Collect each provider's results. Run each collect in its own thread so we
    # don't block on a slow vendor while others have already finished.
    results: dict[str, list[CouncilLabel]] = {}
    errors: dict[str, Exception] = {}

    def _collect(provider: BaseCouncilProvider, handle: str):
        try:
            results[provider.name] = provider.collect(handle)
        except Exception as e:
            errors[provider.name] = e
            print(f"[{provider.name}] ERROR during collect: {e}")

    threads = [
        threading.Thread(target=_collect, args=(p, handles[p.name]), daemon=False)
        for p in providers
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise RuntimeError(f"Council failed: {errors}")

    # Persist raw outputs per provider
    for name, outs in results.items():
        path = output_dir / f"labels_{name}.json"
        path.write_text(json.dumps([asdict(o) for o in outs], indent=2))
        print(f"Wrote {path} ({len(outs)} labels)")

    # Adjudicate
    adjudicated = adjudicate_all(sample, results, confidence_floor=confidence_floor)
    summary = adjudicated["summary"]
    print()
    print("=" * 70)
    print("COUNCIL ADJUDICATION SUMMARY")
    print("=" * 70)
    print(f"  Gold      : {summary['n_gold']:>5} ({summary['gold_rate']*100:.1f}%)")
    print(f"  Silver    : {summary['n_silver']:>5} ({summary['silver_rate']*100:.1f}%)")
    print(f"  Disputed  : {summary['n_disputed']:>5} ({summary['disputed_rate']*100:.1f}%)")
    print(f"  Low-conf in gold: {summary['low_confidence_in_gold']}")
    print(f"  Gold label distribution: {summary['label_distribution_gold']}")
    if summary["n_silver"]:
        print(f"  Silver label distribution: {summary['label_distribution_silver']}")

    # Persist adjudicated buckets
    (output_dir / "gold.json").write_text(json.dumps(adjudicated["gold"], indent=2))
    (output_dir / "silver.json").write_text(json.dumps(adjudicated["silver"], indent=2))
    (output_dir / "disputed.json").write_text(json.dumps(adjudicated["disputed"], indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return adjudicated
