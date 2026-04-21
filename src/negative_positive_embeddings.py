# ABOUTME: Analyzes freedom's semantic distance to negative vs positive concept clusters.
# ABOUTME: Uses HistWords embeddings to test Berlin's negative/positive liberty distinction.

"""
Negative vs Positive Freedom Analysis - Word Embeddings

Uses HistWords temporal embeddings to measure whether "freedom" has moved
closer to positive-liberty concepts (rights, entitlements) or negative-liberty
concepts (absence, removal of constraints).

Based on Isaiah Berlin's "Two Concepts of Liberty" (1958).
"""

import json
from pathlib import Path
from datetime import datetime

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

        # Compute ratio (negative / positive) - higher = more negative-leaning
        if 'error' not in neg_dist and 'error' not in pos_dist:
            neg_mean = neg_dist.get('overall', {}).get('mean_distance', 1)
            pos_mean = pos_dist.get('overall', {}).get('mean_distance', 1)
            # Invert: if closer to positive (lower distance), ratio should be lower
            # ratio < 1 means closer to positive, ratio > 1 means closer to negative
            results['ratio_trajectory'][decade] = {
                'neg_distance': neg_mean,
                'pos_distance': pos_mean,
                'ratio': neg_mean / pos_mean if pos_mean > 0 else None,
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
    }

    # Print summary
    print("\nDistance to Concept Clusters Over Time:")
    print(f"{'Decade':<10} {'Neg Distance':>14} {'Pos Distance':>14} {'Ratio':>10} {'Closer To':>12}")
    print("-" * 60)

    for decade in results['decades']:
        if decade in results['ratio_trajectory']:
            data = results['ratio_trajectory'][decade]
            print(f"{decade:<10} {data['neg_distance']:>14.4f} {data['pos_distance']:>14.4f} "
                  f"{data['ratio']:>10.3f} {data['closer_to']:>12}")

    # Summary statistics
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    ratios = results['ratio_trajectory']
    first_decade = min(ratios.keys())
    last_decade = max(ratios.keys())

    first_ratio = ratios[first_decade]['ratio']
    last_ratio = ratios[last_decade]['ratio']

    print(f"""
1. In {first_decade}, 'freedom' was {'closer to NEGATIVE' if first_ratio > 1 else 'closer to POSITIVE'} concepts
   (ratio: {first_ratio:.3f})

2. In {last_decade}, 'freedom' was {'closer to NEGATIVE' if last_ratio > 1 else 'closer to POSITIVE'} concepts
   (ratio: {last_ratio:.3f})

3. Change: {((last_ratio - first_ratio) / first_ratio * 100):+.1f}% shift in ratio
""")

    # Track when "freedom" became closer to positive cluster
    crossover_decade = None
    prev_closer = ratios[first_decade]['closer_to']
    for decade in sorted(ratios.keys()):
        current_closer = ratios[decade]['closer_to']
        if prev_closer == 'negative' and current_closer == 'positive':
            crossover_decade = decade
            break
        prev_closer = current_closer

    if crossover_decade:
        print(f"4. Crossover decade (became closer to positive): {crossover_decade}")
    else:
        print("4. No crossover - 'freedom' remained closer to same cluster throughout")

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
    ratios_list = [ratios[d]['ratio'] for d in sorted(ratios.keys())]
    trend_start = np.mean(ratios_list[:3])  # First 3 decades
    trend_end = np.mean(ratios_list[-3:])   # Last 3 decades

    if trend_end < trend_start:
        print("""
FINDING: 'Freedom' has moved CLOSER to POSITIVE concepts over time.

This supports the hypothesis that freedom shifted from:
- Negative liberty (freedom FROM constraints)
- To positive liberty (freedom TO do/have things)
""")
    else:
        print("""
FINDING: 'Freedom' has remained closer to or moved toward NEGATIVE concepts.

This suggests negative liberty framing (freedom FROM) may still dominate
in semantic space, even if phrase-level analysis shows different patterns.
""")


if __name__ == '__main__':
    main()
