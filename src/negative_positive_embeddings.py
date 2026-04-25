# ABOUTME: Analyzes freedom's semantic distance to negative vs positive concept clusters.
# ABOUTME: Uses HistWords embeddings to test Berlin's negative/positive liberty distinction.

"""
Negative vs Positive Freedom Analysis - Word Embeddings

Uses HistWords temporal embeddings to measure whether "freedom" has trended
toward positive-liberty concepts (rights, capacities, opportunities) or
negative-liberty concepts (absence, removal of constraints).

Based on Isaiah Berlin's "Two Concepts of Liberty" (1958).
"""

import json
from pathlib import Path
from datetime import datetime
from math import erf, sqrt

import numpy as np

from .embeddings import TemporalEmbeddings
from .metrics import cosine_distance, cosine_similarity


# Concept clusters for negative vs positive freedom
NEGATIVE_CONCEPTS = {
    # Absence/removal words
    'absence': ['absence', 'lack', 'without', 'remove', 'removal', 'escape', 'release'],
    # Constraint words (what you're free FROM)
    'constraints': ['slavery', 'bondage', 'chains', 'tyranny', 'oppression', 'constraint',
                    'coercion', 'domination', 'subjection', 'servitude'],
    # Liberation words
    'liberation': ['liberation', 'emancipation', 'deliverance', 'release', 'rescue'],
}

POSITIVE_CONCEPTS = {
    # Rights/entitlement words
    'rights': ['right', 'rights', 'entitlement', 'entitlements', 'claim', 'privilege'],
    # Capacity/ability words
    'capacity': ['ability', 'capacity', 'power', 'capability', 'potential', 'opportunity'],
    # Action words (what you're free TO do)
    'action': ['choose', 'act', 'pursue', 'achieve', 'accomplish', 'realize', 'fulfill'],
    # Self-determination words
    'autonomy': ['autonomy', 'self-determination', 'independence', 'sovereignty', 'agency'],
}


def normal_two_sided_p(z: float) -> float:
    """Approximate two-sided p-value from a normal z-score."""
    return float(2 * (1 - (0.5 * (1 + erf(abs(z) / sqrt(2))))))


def linear_trend(decades: list[int], values: list[float]) -> dict | None:
    """OLS trend over decades. Slope is reported per century."""
    if len(decades) < 3 or len(decades) != len(values):
        return None

    x = (np.array(decades, dtype=float) - np.mean(decades)) / 100.0
    y = np.array(values, dtype=float)
    ss_xx = float(np.sum(x**2))
    if ss_xx == 0:
        return None

    slope = float(np.sum(x * (y - np.mean(y))) / ss_xx)
    intercept = float(np.mean(y))
    y_hat = intercept + slope * x
    residuals = y - y_hat
    rss = float(np.sum(residuals**2))
    df = len(y) - 2
    se = sqrt((rss / df) / ss_xx) if df > 0 and ss_xx > 0 else 0.0
    z = slope / se if se > 0 else 0.0

    return {
        "slope_per_century": round(slope, 6),
        "intercept_at_mean_decade": round(intercept, 6),
        "std_error": round(se, 6),
        "z": round(z, 3),
        "p_value_approx": round(normal_two_sided_p(z), 6) if se > 0 else None,
    }


def compute_cluster_distance(embeddings: TemporalEmbeddings, word: str, decade: int,
                            concept_cluster: dict) -> dict:
    """
    Compute average distance from a word to a concept cluster.
    Returns distances to each sub-cluster and overall average.
    """
    word_vec = embeddings.get_vector(word, decade)
    if word_vec is None:
        return {'error': f"Word '{word}' not found in {decade}"}

    results = {}
    all_distances = []

    for cluster_name, concepts in concept_cluster.items():
        distances = []
        for concept in concepts:
            concept_vec = embeddings.get_vector(concept, decade)
            if concept_vec is not None:
                dist = cosine_distance(word_vec, concept_vec)
                distances.append(dist)
                all_distances.append(dist)

        if distances:
            results[cluster_name] = {
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'n_concepts': len(distances),
                'concepts_found': len(distances),
                'concepts_total': len(concepts),
            }

    if all_distances:
        results['overall'] = {
            'mean_distance': np.mean(all_distances),
            'std_distance': np.std(all_distances),
            'n_concepts': len(all_distances),
        }

    return results


def compute_cluster_similarity(embeddings: TemporalEmbeddings, word: str, decade: int,
                               concept_cluster: dict) -> dict:
    """
    Compute average similarity from a word to a concept cluster.
    Higher similarity = closer semantic relationship.
    """
    word_vec = embeddings.get_vector(word, decade)
    if word_vec is None:
        return {'error': f"Word '{word}' not found in {decade}"}

    results = {}
    all_similarities = []

    for cluster_name, concepts in concept_cluster.items():
        similarities = []
        for concept in concepts:
            concept_vec = embeddings.get_vector(concept, decade)
            if concept_vec is not None:
                sim = cosine_similarity(word_vec, concept_vec)
                similarities.append(sim)
                all_similarities.append(sim)

        if similarities:
            results[cluster_name] = {
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'n_concepts': len(similarities),
            }

    if all_similarities:
        results['overall'] = {
            'mean_similarity': np.mean(all_similarities),
            'std_similarity': np.std(all_similarities),
            'n_concepts': len(all_similarities),
        }

    return results


def analyze_freedom_trajectory(embeddings: TemporalEmbeddings) -> dict:
    """
    Analyze how 'freedom' moves relative to negative/positive concept clusters
    across all loaded decades.
    """
    decades = embeddings.decades
    results = {
        'decades': decades,
        'negative_cluster': {},
        'positive_cluster': {},
        'ratio_trajectory': {},
        'individual_concepts': {},
    }

    for decade in decades:
        # Distance to negative concepts
        neg_dist = compute_cluster_distance(embeddings, 'freedom', decade, NEGATIVE_CONCEPTS)
        if 'error' not in neg_dist:
            results['negative_cluster'][decade] = neg_dist

        # Distance to positive concepts
        pos_dist = compute_cluster_distance(embeddings, 'freedom', decade, POSITIVE_CONCEPTS)
        if 'error' not in pos_dist:
            results['positive_cluster'][decade] = pos_dist

        # Compute signed distance gap. Positive means closer to positive concepts
        # because positive distance is smaller than negative distance.
        if 'error' not in neg_dist and 'error' not in pos_dist:
            neg_mean = neg_dist.get('overall', {}).get('mean_distance', 1)
            pos_mean = pos_dist.get('overall', {}).get('mean_distance', 1)
            positive_tilt = neg_mean - pos_mean
            results['ratio_trajectory'][decade] = {
                'neg_distance': neg_mean,
                'pos_distance': pos_mean,
                'distance_ratio_neg_over_pos': neg_mean / pos_mean if pos_mean > 0 else None,
                'positive_tilt': positive_tilt,
                'closer_to': 'negative' if neg_mean < pos_mean else 'positive',
            }

    # Track individual key concepts over time
    key_concepts = ['slavery', 'bondage', 'right', 'rights', 'autonomy', 'entitlement',
                    'oppression', 'tyranny', 'ability', 'power', 'choose']

    for concept in key_concepts:
        results['individual_concepts'][concept] = {}
        for decade in decades:
            freedom_vec = embeddings.get_vector('freedom', decade)
            concept_vec = embeddings.get_vector(concept, decade)
            if freedom_vec is not None and concept_vec is not None:
                dist = cosine_distance(freedom_vec, concept_vec)
                results['individual_concepts'][concept][decade] = round(dist, 4)

    return results


def summarize_tilt_trend(ratio_trajectory: dict[int, dict]) -> dict:
    """Summarize whether the positive/negative semantic tilt changed over time."""
    decades = sorted(ratio_trajectory.keys())
    tilts = [ratio_trajectory[d]['positive_tilt'] for d in decades]
    trend = linear_trend(decades, tilts)

    early_decades = decades[:3]
    late_decades = decades[-3:]
    early_mean = float(np.mean([ratio_trajectory[d]['positive_tilt'] for d in early_decades]))
    late_mean = float(np.mean([ratio_trajectory[d]['positive_tilt'] for d in late_decades]))
    endpoint_change = late_mean - early_mean

    return {
        "metric": "positive_tilt = negative_cluster_distance - positive_cluster_distance",
        "interpretation": "positive values are closer to positive-liberty concepts; the hypothesis test concerns change over time, not an absolute crossover",
        "first_decade": decades[0],
        "last_decade": decades[-1],
        "early_decades": early_decades,
        "late_decades": late_decades,
        "early_mean_positive_tilt": round(early_mean, 6),
        "late_mean_positive_tilt": round(late_mean, 6),
        "early_to_late_change": round(endpoint_change, 6),
        "trend": trend,
    }


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'sgns'
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("NEGATIVE vs POSITIVE FREEDOM ANALYSIS (Word Embeddings)")
    print("=" * 70)
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Load embeddings
    print("Loading HistWords embeddings...")
    embeddings = TemporalEmbeddings(data_dir)
    embeddings.load_decades(1800, 1990, 10)
    print(f"Loaded {len(embeddings.decades)} decades")
    print()

    # Run analysis
    print("Analyzing 'freedom' trajectory relative to concept clusters...")
    print("-" * 70)

    results = analyze_freedom_trajectory(embeddings)

    # Add metadata
    output = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'source': 'HistWords (Google Books)',
            'coverage': '1800-1990',
            'negative_concepts': NEGATIVE_CONCEPTS,
            'positive_concepts': POSITIVE_CONCEPTS,
        },
        'trajectory': results,
        'trend_test': summarize_tilt_trend(results['ratio_trajectory']),
    }

    # Print summary
    print("\nDistance to Concept Clusters Over Time:")
    print(f"{'Decade':<10} {'Neg Distance':>14} {'Pos Distance':>14} {'Pos Tilt':>12} {'Closer To':>12}")
    print("-" * 68)

    for decade in results['decades']:
        if decade in results['ratio_trajectory']:
            data = results['ratio_trajectory'][decade]
            print(f"{decade:<10} {data['neg_distance']:>14.4f} {data['pos_distance']:>14.4f} "
                  f"{data['positive_tilt']:>12.4f} {data['closer_to']:>12}")

    # Summary statistics
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    trend_test = output['trend_test']
    trend = trend_test['trend'] or {}
    print(f"""
1. Primary metric: positive_tilt = negative distance - positive distance.
   Higher values mean "freedom" is relatively closer to positive-liberty concepts.

2. Early mean ({trend_test['early_decades']}): {trend_test['early_mean_positive_tilt']:+.4f}
   Late mean ({trend_test['late_decades']}):  {trend_test['late_mean_positive_tilt']:+.4f}
   Change: {trend_test['early_to_late_change']:+.4f}

3. Linear trend: {trend.get('slope_per_century', 'N/A')} per century
   Approx p-value: {trend.get('p_value_approx', 'N/A')}
""")

    # Key concept trajectories
    print()
    print("Key Concept Distance Changes (1800 → 1990):")
    print("-" * 50)

    for concept in ['slavery', 'bondage', 'right', 'rights', 'autonomy', 'oppression']:
        if concept in results['individual_concepts']:
            distances = results['individual_concepts'][concept]
            if 1800 in distances and 1990 in distances:
                change = distances[1990] - distances[1800]
                direction = "↑ further" if change > 0 else "↓ closer"
                print(f"  {concept:<15} {distances[1800]:.3f} → {distances[1990]:.3f} ({direction})")

    # Save results
    output_path = output_dir / 'negative_positive_embeddings.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Key findings
    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Determine trend
    tilt_change = trend_test['early_to_late_change']

    if tilt_change > 0:
        print("""
FINDING: 'Freedom' trended toward POSITIVE concepts over the observed period.

This supports the revised hypothesis only as a trend claim: the relative
association changed over time. It does not require an absolute switch from
one liberty category to another.
""")
    else:
        print("""
FINDING: 'Freedom' did not trend toward POSITIVE concepts over the observed period.

This weakens the revised hypothesis in the embedding-cluster analysis. The
result should be triangulated against sentence-level label proportions.
""")


if __name__ == '__main__':
    main()
