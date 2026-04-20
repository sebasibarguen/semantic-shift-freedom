# ABOUTME: Analyzes economic dimensions of "freedom" across 1900-2010.
# ABOUTME: Tests hypothesis that freedom increasingly associates with financial/market concepts.

"""
Economic Freedom Analysis

Tests the hypothesis that "freedom" has become more associated with
economic/financial concepts in the 20th century, particularly post-1980.

Data sources:
- HistWords embeddings (1900-1990)
- COHA collocates (1900-2010)
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from embeddings import TemporalEmbeddings
import numpy as np


def cosine_distance(v1, v2):
    """Compute cosine distance between two vectors."""
    if v1 is None or v2 is None:
        return None
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return None
    similarity = np.dot(v1, v2) / (norm1 * norm2)
    return 1 - similarity


# Economic terms to track proximity to "freedom"
ECONOMIC_TERMS = [
    'market', 'markets', 'trade', 'commerce', 'business', 'enterprise',
    'capitalism', 'capitalist', 'socialist', 'socialism',
    'economic', 'economy', 'economies',
    'property', 'ownership', 'private',
    'wealth', 'capital', 'investment',
    'labor', 'worker', 'workers', 'wage', 'wages',
    'consumer', 'consumption',
    'regulation', 'deregulation',
    'tax', 'taxes', 'taxation',
    'profit', 'profits', 'income',
    'competition', 'competitive',
    'entrepreneur', 'entrepreneurial',
    'fiscal', 'monetary',
]

# Specific bigrams/phrases to look for (for reference - need Ngram API)
ECONOMIC_PHRASES = [
    'economic freedom',
    'financial freedom',
    'free market',
    'free enterprise',
    'free trade',
    'market freedom',
]


def analyze_histwords_economic(embeddings: TemporalEmbeddings, decades: list[int]) -> dict:
    """Analyze distance between 'freedom' and economic terms over time."""
    results = {}

    for decade in decades:
        decade_results = {}

        # Get freedom's neighbors for this decade
        neighbors = embeddings.get_nearest_neighbors('freedom', decade, k=100)
        neighbor_words = [w for w, _ in neighbors]

        # Check which economic terms appear in top 100 neighbors
        economic_in_neighbors = []
        for term in ECONOMIC_TERMS:
            if term in neighbor_words:
                rank = neighbor_words.index(term) + 1
                score = next((s for w, s in neighbors if w == term), None)
                economic_in_neighbors.append({
                    'term': term,
                    'rank': rank,
                    'similarity': score
                })

        # Calculate distances to key economic terms
        freedom_vec = embeddings.get_vector('freedom', decade)
        distances = {}
        for term in ['market', 'capitalism', 'property', 'trade', 'economic', 'wealth']:
            term_vec = embeddings.get_vector(term, decade)
            dist = cosine_distance(freedom_vec, term_vec)
            if dist is not None:
                distances[term] = round(dist, 4)

        decade_results['economic_in_top100'] = economic_in_neighbors
        decade_results['distances_to_economic_terms'] = distances
        decade_results['count_economic_neighbors'] = len(economic_in_neighbors)

        results[decade] = decade_results

    return results


def analyze_coha_economic(coha_data: dict) -> dict:
    """Extract economic signal from COHA collocate data."""
    results = {}

    freedom_decades = coha_data.get('freedom', {}).get('by_decade', {})

    for decade_str, data in freedom_decades.items():
        decade = int(decade_str)
        if decade < 1900:
            continue

        collocates = data.get('top_collocates', [])
        collocate_words = [w for w, _ in collocates]

        # Find economic terms in collocates
        economic_collocates = []
        for term in ECONOMIC_TERMS:
            if term in collocate_words:
                idx = collocate_words.index(term)
                count = collocates[idx][1]
                economic_collocates.append({
                    'term': term,
                    'rank': idx + 1,
                    'count': count
                })

        # Get domain percentages
        domain_pcts = data.get('domain_percentages', {})

        results[decade] = {
            'economic_domain_pct': domain_pcts.get('economic', 0),
            'economic_collocates': economic_collocates,
            'count': len(economic_collocates),
        }

    return results


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'outputs'

    print("=" * 60)
    print("ECONOMIC FREEDOM ANALYSIS (1900-2010)")
    print("=" * 60)
    print()
    print("Testing hypothesis: Freedom increasingly associated with")
    print("economic/financial concepts in the 20th century.")
    print()

    # Load HistWords
    print("Loading HistWords embeddings...")
    data_dir = project_root / 'data' / 'sgns'
    embeddings = TemporalEmbeddings(data_dir)
    embeddings.load_decades(start=1900, end=1990)

    decades_1900s = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990]

    print("\n" + "=" * 60)
    print("HISTWORDS ANALYSIS: Economic Terms Near 'Freedom'")
    print("=" * 60)

    histwords_results = analyze_histwords_economic(embeddings, decades_1900s)

    # Print summary
    print("\nEconomic terms in freedom's top 100 neighbors:")
    print("-" * 50)
    for decade in decades_1900s:
        data = histwords_results[decade]
        count = data['count_economic_neighbors']
        terms = [e['term'] for e in data['economic_in_top100'][:5]]
        terms_str = ', '.join(terms) if terms else '(none)'
        print(f"  {decade}s: {count} terms - {terms_str}")

    print("\nDistance from 'freedom' to key economic concepts:")
    print("-" * 50)
    print(f"{'Decade':<10}", end="")
    for term in ['market', 'capitalism', 'property', 'trade', 'wealth']:
        print(f"{term:<12}", end="")
    print()

    for decade in decades_1900s:
        distances = histwords_results[decade]['distances_to_economic_terms']
        print(f"{decade}s     ", end="")
        for term in ['market', 'capitalism', 'property', 'trade', 'wealth']:
            dist = distances.get(term, '-')
            if isinstance(dist, float):
                print(f"{dist:<12.3f}", end="")
            else:
                print(f"{dist:<12}", end="")
        print()

    # Load COHA data
    coha_path = output_dir / 'coha_collocates.json'
    if coha_path.exists():
        print("\n" + "=" * 60)
        print("COHA ANALYSIS: Economic Collocates (1900-2010)")
        print("=" * 60)

        with open(coha_path) as f:
            coha_data = json.load(f)

        coha_results = analyze_coha_economic(coha_data)

        print("\nEconomic domain percentage in freedom collocates:")
        print("-" * 50)
        for decade in sorted(coha_results.keys()):
            data = coha_results[decade]
            pct = data['economic_domain_pct']
            count = data['count']
            terms = [e['term'] for e in data['economic_collocates'][:3]]
            terms_str = ', '.join(terms) if terms else '(none)'
            print(f"  {decade}s: {pct:5.1f}% ({count} terms) - {terms_str}")
    else:
        coha_results = {}
        print("\nCOHA data not found, skipping...")

    # Save results
    output = {
        'hypothesis': 'Freedom increasingly associated with economic/financial concepts post-1980',
        'coverage': {
            'histwords': '1900-1990',
            'coha': '1900-2010',
        },
        'economic_terms_tracked': ECONOMIC_TERMS,
        'histwords_analysis': histwords_results,
        'coha_analysis': coha_results,
    }

    output_path = output_dir / 'economic_freedom_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Compute trend
    if histwords_results:
        early_count = histwords_results[1900]['count_economic_neighbors']
        late_count = histwords_results[1990]['count_economic_neighbors']
        print(f"\nHistWords economic neighbors: {early_count} (1900) -> {late_count} (1990)")

        # Distance to 'market' trend
        early_market = histwords_results[1900]['distances_to_economic_terms'].get('market')
        late_market = histwords_results[1990]['distances_to_economic_terms'].get('market')
        if early_market and late_market:
            if late_market < early_market:
                print(f"Freedom-market distance: {early_market:.3f} (1900) -> {late_market:.3f} (1990) [CLOSER]")
            else:
                print(f"Freedom-market distance: {early_market:.3f} (1900) -> {late_market:.3f} (1990) [FURTHER]")

    if coha_results:
        decades = sorted(coha_results.keys())
        early_pct = coha_results[decades[0]]['economic_domain_pct']
        late_pct = coha_results[decades[-1]]['economic_domain_pct']
        print(f"COHA economic domain: {early_pct:.1f}% ({decades[0]}) -> {late_pct:.1f}% ({decades[-1]})")

    print(f"\nResults saved to: {output_path}")
    print()
    print("NOTE: For recent data (2010-present), query Google Ngram Viewer")
    print("for phrases like 'economic freedom', 'financial freedom', 'free market'")


if __name__ == '__main__':
    main()
