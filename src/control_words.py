# ABOUTME: Compares freedom/liberty divergence against control word pairs.
# ABOUTME: Establishes whether observed semantic drift is unusual or typical for abstract nouns.

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from embeddings import TemporalEmbeddings
from metrics import cosine_similarity, semantic_change_score, neighbor_overlap


# Control pairs: (word1, word2, category)
# Near-synonyms expected to have moderate drift
SYNONYM_PAIRS = [
    ("justice", "fairness", "near-synonym"),
    ("truth", "honesty", "near-synonym"),
    ("honor", "dignity", "near-synonym"),
    ("power", "authority", "near-synonym"),
    ("virtue", "morality", "near-synonym"),
]

# Stable pairs expected to have low drift
STABLE_PAIRS = [
    ("king", "queen", "stable"),
    ("mother", "father", "stable"),
    ("war", "peace", "stable-antonym"),
]

# The pair under study
TARGET_PAIR = ("freedom", "liberty", "target")

ALL_PAIRS = [TARGET_PAIR] + SYNONYM_PAIRS + STABLE_PAIRS


def pair_similarity_trajectory(emb, word1, word2, decades):
    """Compute cosine similarity between two words across decades."""
    trajectory = {}
    for decade in decades:
        v1 = emb.get_vector(word1, decade)
        v2 = emb.get_vector(word2, decade)
        if v1 is not None and v2 is not None:
            trajectory[decade] = cosine_similarity(v1, v2)
    return trajectory


def pair_total_divergence(trajectory):
    """Compute total divergence: similarity change from first to last available decade."""
    decades = sorted(trajectory.keys())
    if len(decades) < 2:
        return None
    return trajectory[decades[-1]] - trajectory[decades[0]]


def word_semantic_drift(emb, word, decades):
    """Compute total semantic change of a single word (cosine distance first→last)."""
    available = [d for d in decades if emb.word_exists(word, d)]
    if len(available) < 2:
        return None
    return semantic_change_score(emb, word, available[0], available[-1])


def run_analysis():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "sgns"

    print("=" * 70)
    print("CONTROL WORD ANALYSIS")
    print("Is freedom/liberty divergence unusual?")
    print("=" * 70)
    print()

    print("Loading HistWords embeddings...")
    emb = TemporalEmbeddings(str(data_dir))
    emb.load_decades(start=1800, end=1990, step=10)
    decades = emb.decades
    print()

    results = {
        "pair_trajectories": {},
        "divergence_summary": {},
        "individual_drift": {},
        "neighbor_overlap": {},
    }

    # =========================================================================
    # PAIR SIMILARITY TRAJECTORIES
    # =========================================================================
    print("PAIR SIMILARITY OVER TIME")
    print("-" * 70)
    print(f"{'Pair':<25} {'1800':>8} {'1850':>8} {'1900':>8} {'1950':>8} {'1990':>8} {'Δ':>8}")
    print("-" * 70)

    for word1, word2, category in ALL_PAIRS:
        traj = pair_similarity_trajectory(emb, word1, word2, decades)
        divergence = pair_total_divergence(traj)

        results["pair_trajectories"][f"{word1}/{word2}"] = {
            "category": category,
            "trajectory": {str(d): round(v, 4) for d, v in traj.items()},
            "divergence": round(divergence, 4) if divergence is not None else None,
        }

        key_decades = [1800, 1850, 1900, 1950, 1990]
        vals = [f"{traj.get(d, float('nan')):.3f}" if d in traj else "  N/A " for d in key_decades]
        div_str = f"{divergence:+.3f}" if divergence is not None else "  N/A"
        label = f"{word1}/{word2}"
        marker = " ***" if category == "target" else ""
        print(f"{label:<25} {'  '.join(vals)}  {div_str}{marker}")

        results["divergence_summary"][f"{word1}/{word2}"] = {
            "category": category,
            "divergence": round(divergence, 4) if divergence is not None else None,
        }

    print()

    # =========================================================================
    # RANK THE DIVERGENCE
    # =========================================================================
    print("DIVERGENCE RANKING (most diverging first)")
    print("-" * 70)

    ranked = sorted(
        results["divergence_summary"].items(),
        key=lambda x: x[1]["divergence"] if x[1]["divergence"] is not None else 0,
    )

    for i, (pair, info) in enumerate(ranked):
        div = info["divergence"]
        cat = info["category"]
        marker = " <-- TARGET" if cat == "target" else ""
        if div is not None:
            print(f"  {i+1}. {pair:<25} Δ = {div:+.4f}  ({cat}){marker}")

    target_div = results["divergence_summary"]["freedom/liberty"]["divergence"]
    synonym_divs = [
        v["divergence"] for k, v in results["divergence_summary"].items()
        if v["category"] == "near-synonym" and v["divergence"] is not None
    ]

    if synonym_divs:
        avg_synonym_div = sum(synonym_divs) / len(synonym_divs)
        print()
        print(f"  Freedom/liberty divergence: {target_div:+.4f}")
        print(f"  Average near-synonym divergence: {avg_synonym_div:+.4f}")
        ratio = target_div / avg_synonym_div if avg_synonym_div != 0 else float('inf')
        print(f"  Ratio: {ratio:.1f}x")

        more_divergent = sum(1 for d in synonym_divs if d <= target_div)
        print(f"  Freedom/liberty is more divergent than {more_divergent}/{len(synonym_divs)} synonym pairs")

    print()

    # =========================================================================
    # INDIVIDUAL WORD DRIFT (is "freedom" changing more than other words?)
    # =========================================================================
    print("INDIVIDUAL WORD SEMANTIC DRIFT (1800→1990)")
    print("-" * 70)

    all_words = set()
    for w1, w2, _ in ALL_PAIRS:
        all_words.add(w1)
        all_words.add(w2)

    for word in sorted(all_words):
        drift = word_semantic_drift(emb, word, decades)
        if drift is not None:
            results["individual_drift"][word] = round(drift, 4)
            marker = " <--" if word == "freedom" else ""
            print(f"  {word:<20} cosine distance = {drift:.4f}{marker}")

    print()

    ranked_drift = sorted(results["individual_drift"].items(), key=lambda x: x[1], reverse=True)
    freedom_rank = next(i for i, (w, _) in enumerate(ranked_drift) if w == "freedom") + 1
    print(f"  'freedom' ranks #{freedom_rank}/{len(ranked_drift)} in semantic drift")
    results["freedom_drift_rank"] = f"{freedom_rank}/{len(ranked_drift)}"

    print()

    # =========================================================================
    # NEIGHBOR OVERLAP (stability of semantic neighborhood)
    # =========================================================================
    print("NEIGHBOR OVERLAP (1800 vs 1990, k=50)")
    print("(Higher = more stable neighborhood)")
    print("-" * 70)

    for word in sorted(all_words):
        overlap = neighbor_overlap(emb, word, 1800, 1990, k=50)
        if overlap is not None:
            results["neighbor_overlap"][word] = round(overlap, 4)
            marker = " <--" if word == "freedom" else ""
            print(f"  {word:<20} Jaccard overlap = {overlap:.4f}{marker}")

    print()

    # =========================================================================
    # SAVE
    # =========================================================================
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "control_words.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    if target_div is not None and synonym_divs:
        if target_div < min(synonym_divs):
            print("Freedom/liberty diverged MORE than all control synonym pairs.")
            print("The divergence is UNUSUAL — not just typical semantic drift.")
        elif target_div < avg_synonym_div:
            print("Freedom/liberty diverged more than average for synonym pairs.")
            print("The divergence is NOTABLE but not extreme.")
        else:
            print("Freedom/liberty divergence is WITHIN NORMAL RANGE for synonym pairs.")
            print("The finding still holds but the magnitude is not exceptional.")

    return results


if __name__ == "__main__":
    run_analysis()
