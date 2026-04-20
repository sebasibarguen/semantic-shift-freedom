# ABOUTME: Analyzes "freedom/liberty from" vs "freedom/liberty to" patterns in EEBO corpus.
# ABOUTME: Establishes Early Modern baseline for negative vs positive freedom framing.

"""
Negative vs Positive Freedom Analysis - EEBO-TCP (1500-1700)

Searches the Early English Books Online corpus for:
- "freedom/liberty FROM X" patterns (negative freedom)
- "freedom/liberty TO Y" patterns (positive freedom)

Establishes baseline for comparison with modern period.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def normalize_text(text: str) -> str:
    """Normalize Early Modern English text for searching."""
    # Basic normalization
    text = text.lower()
    # Long-s to s
    text = text.replace('ſ', 's')
    # Common variants
    text = text.replace('vv', 'w')
    # u/v normalization (simplified)
    text = re.sub(r'\bvnto\b', 'unto', text)
    text = re.sub(r'\bvpon\b', 'upon', text)
    text = re.sub(r'\bvs\b', 'us', text)
    text = re.sub(r'\bhaue\b', 'have', text)
    text = re.sub(r'\bgiue\b', 'give', text)
    return text


def search_from_to_patterns(contexts: list) -> dict:
    """
    Search for "freedom/liberty FROM X" and "freedom/liberty TO Y" patterns.
    Returns counts and examples.
    """
    results = {
        'from_patterns': [],
        'to_patterns': [],
        'from_count': 0,
        'to_count': 0,
    }

    # Patterns to match
    # Note: lookbehind for freedom/liberty, then FROM/TO, then capture what follows
    freedom_words = r'(?:freedom|freedome|freedomes|liberty|libertie|liberties)'

    # "freedom/liberty from X" - negative freedom
    from_pattern = re.compile(
        rf'({freedom_words})\s+from\s+(\w+(?:\s+\w+)?)',
        re.IGNORECASE
    )

    # "freedom/liberty to X" - positive freedom
    to_pattern = re.compile(
        rf'({freedom_words})\s+to\s+(\w+)',
        re.IGNORECASE
    )

    for ctx in contexts:
        full_context = ctx.get('full_context', '')
        normalized = normalize_text(full_context)

        # Search for FROM patterns
        from_matches = from_pattern.findall(normalized)
        for match in from_matches:
            results['from_count'] += 1
            results['from_patterns'].append({
                'freedom_word': match[0],
                'following': match[1],
                'context': full_context[:200],
            })

        # Search for TO patterns
        to_matches = to_pattern.findall(normalized)
        for match in to_matches:
            results['to_count'] += 1
            results['to_patterns'].append({
                'freedom_word': match[0],
                'following': match[1],
                'context': full_context[:200],
            })

    return results


def analyze_what_follows(patterns: list) -> dict:
    """Analyze what words follow 'from' or 'to' in the patterns."""
    word_counts = defaultdict(int)

    for p in patterns:
        following = p['following'].lower().strip()
        # Get first word
        first_word = following.split()[0] if following else ''
        if first_word:
            word_counts[first_word] += 1

    # Sort by frequency
    sorted_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_counts[:30])


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'eebo' / 'fulltext_corpus'
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("NEGATIVE vs POSITIVE FREEDOM ANALYSIS (EEBO-TCP 1500-1700)")
    print("=" * 70)
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Load and analyze each period
    periods = [
        '1500-1550',
        '1550-1600',
        '1600-1650',
        '1650-1700',
    ]

    all_results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'source': 'EEBO-TCP fulltext corpus',
            'coverage': '1500-1700',
        },
        'by_period': {},
        'totals': {
            'from_count': 0,
            'to_count': 0,
        },
        'all_from_patterns': [],
        'all_to_patterns': [],
    }

    for period in periods:
        filepath = data_dir / f'freedom_contexts_{period}.json'
        if not filepath.exists():
            print(f"  Skipping {period} - file not found")
            continue

        print(f"Analyzing {period}...")

        with open(filepath, 'r') as f:
            data = json.load(f)

        period_results = {
            'from_count': 0,
            'to_count': 0,
            'from_examples': [],
            'to_examples': [],
            'texts_analyzed': len(data),
        }

        for entry in data:
            contexts = entry.get('contexts', [])
            result = search_from_to_patterns(contexts)

            period_results['from_count'] += result['from_count']
            period_results['to_count'] += result['to_count']

            # Keep some examples
            if len(period_results['from_examples']) < 20:
                period_results['from_examples'].extend(result['from_patterns'][:5])

            if len(period_results['to_examples']) < 20:
                period_results['to_examples'].extend(result['to_patterns'][:5])

            # Add to overall
            all_results['all_from_patterns'].extend(result['from_patterns'])
            all_results['all_to_patterns'].extend(result['to_patterns'])

        # Calculate ratio
        total = period_results['from_count'] + period_results['to_count']
        if total > 0:
            period_results['pct_from'] = round(period_results['from_count'] / total * 100, 1)
            period_results['pct_to'] = round(period_results['to_count'] / total * 100, 1)
            period_results['ratio_from_to'] = round(
                period_results['from_count'] / period_results['to_count'], 3
            ) if period_results['to_count'] > 0 else float('inf')

        all_results['by_period'][period] = period_results
        all_results['totals']['from_count'] += period_results['from_count']
        all_results['totals']['to_count'] += period_results['to_count']

        print(f"  FROM: {period_results['from_count']}, TO: {period_results['to_count']}")

    # Analyze what follows FROM and TO
    print()
    print("Analyzing patterns...")

    all_results['from_following'] = analyze_what_follows(all_results['all_from_patterns'])
    all_results['to_following'] = analyze_what_follows(all_results['all_to_patterns'])

    # Calculate overall ratio
    total_from = all_results['totals']['from_count']
    total_to = all_results['totals']['to_count']
    total = total_from + total_to

    if total > 0:
        all_results['totals']['pct_from'] = round(total_from / total * 100, 1)
        all_results['totals']['pct_to'] = round(total_to / total * 100, 1)
        all_results['totals']['ratio_from_to'] = round(total_from / total_to, 3) if total_to > 0 else float('inf')

    # Remove large pattern lists before saving (keep just counts)
    del all_results['all_from_patterns']
    del all_results['all_to_patterns']

    # Print results
    print()
    print("=" * 70)
    print("RESULTS: Early Modern Negative vs Positive Freedom")
    print("=" * 70)

    print("\nBy Period:")
    print(f"{'Period':<15} {'FROM':>10} {'TO':>10} {'% FROM':>10} {'Ratio':>10}")
    print("-" * 55)

    for period in periods:
        if period in all_results['by_period']:
            data = all_results['by_period'][period]
            print(f"{period:<15} {data['from_count']:>10} {data['to_count']:>10} "
                  f"{data.get('pct_from', 0):>9.1f}% {data.get('ratio_from_to', 0):>10.2f}")

    print("-" * 55)
    print(f"{'TOTAL':<15} {total_from:>10} {total_to:>10} "
          f"{all_results['totals'].get('pct_from', 0):>9.1f}% "
          f"{all_results['totals'].get('ratio_from_to', 0):>10.2f}")

    print("\nTop words following 'freedom/liberty FROM':")
    for word, count in list(all_results['from_following'].items())[:10]:
        print(f"  {word}: {count}")

    print("\nTop words following 'freedom/liberty TO':")
    for word, count in list(all_results['to_following'].items())[:10]:
        print(f"  {word}: {count}")

    # Sample contexts
    print("\nSample FROM contexts:")
    for period in periods:
        if period in all_results['by_period']:
            examples = all_results['by_period'][period].get('from_examples', [])[:2]
            for ex in examples:
                ctx = ex['context'][:100] + "..." if len(ex['context']) > 100 else ex['context']
                print(f"  [{period}] ...{ctx}")

    print("\nSample TO contexts:")
    for period in periods:
        if period in all_results['by_period']:
            examples = all_results['by_period'][period].get('to_examples', [])[:2]
            for ex in examples:
                ctx = ex['context'][:100] + "..." if len(ex['context']) > 100 else ex['context']
                print(f"  [{period}] ...{ctx}")

    # Save results
    output_path = output_dir / 'negative_positive_eebo.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Key findings
    print()
    print("=" * 70)
    print("KEY FINDINGS (Early Modern Baseline)")
    print("=" * 70)

    print(f"""
1. Overall ratio: {all_results['totals'].get('ratio_from_to', 0):.1f}x more "freedom FROM" than "freedom TO"
   ({all_results['totals'].get('pct_from', 0):.0f}% negative framing)

2. This establishes the BASELINE for comparison with modern period:
   - Early Modern (1500-1700): {all_results['totals'].get('pct_from', 0):.0f}% negative
   - Modern (1800s from Ngrams): ~57% negative
   - Modern (2010s from Ngrams): ~35% negative

3. Top "freedom FROM" objects: {', '.join(list(all_results['from_following'].keys())[:5])}

4. Top "freedom TO" objects: {', '.join(list(all_results['to_following'].keys())[:5])}
""")


if __name__ == '__main__':
    main()
