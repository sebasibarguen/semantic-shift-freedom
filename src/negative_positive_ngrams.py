# ABOUTME: Analyzes "freedom from" vs "freedom to" phrase frequencies in Google Ngrams.
# ABOUTME: Tests hypothesis of shift from negative (FROM) to positive (TO) freedom framing.

"""
Negative vs Positive Freedom Analysis - Google Ngrams

Tests Isaiah Berlin's "Two Concepts of Liberty" empirically:
- Negative freedom: "freedom FROM X" (absence of constraint)
- Positive freedom: "freedom TO Y" (capacity/right to act)

Tracks the ratio of these framings over time (1800-2019).
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime


def query_ngram(phrase: str, start_year: int = 1800, end_year: int = 2019,
                corpus: int = 26, smoothing: int = 3) -> dict:
    """
    Query Google Ngram Viewer for phrase frequency.

    Corpus 26 = English (2019 version)
    """
    encoded_phrase = urllib.parse.quote(phrase)
    url = f"https://books.google.com/ngrams/json?content={encoded_phrase}&year_start={start_year}&year_end={end_year}&corpus={corpus}&smoothing={smoothing}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data:
                return {
                    'phrase': phrase,
                    'timeseries': data[0].get('timeseries', []),
                    'years': list(range(start_year, end_year + 1)),
                }
            return {'phrase': phrase, 'timeseries': [], 'years': []}
    except Exception as e:
        print(f"  Error querying '{phrase}': {e}")
        return {'phrase': phrase, 'error': str(e), 'timeseries': [], 'years': []}


def compute_ratio(data1: dict, data2: dict) -> list:
    """Compute ratio of two timeseries (data1 / data2)."""
    ts1 = data1.get('timeseries', [])
    ts2 = data2.get('timeseries', [])
    years = data1.get('years', [])

    if not ts1 or not ts2 or len(ts1) != len(ts2):
        return []

    ratios = []
    for i, (v1, v2) in enumerate(zip(ts1, ts2)):
        if v2 > 0:
            ratios.append({'year': years[i], 'ratio': v1 / v2})
        else:
            ratios.append({'year': years[i], 'ratio': None})

    return ratios


def compute_decade_averages(timeseries: list, years: list) -> dict:
    """Compute decade averages from timeseries."""
    decade_vals = {}
    for year, val in zip(years, timeseries):
        decade = (year // 10) * 10
        if decade not in decade_vals:
            decade_vals[decade] = []
        decade_vals[decade].append(val)

    return {str(d): sum(v)/len(v) for d, v in sorted(decade_vals.items())}


def find_inflection_points(timeseries: list, years: list, threshold: float = 0.1) -> list:
    """Find years with significant changes in trajectory."""
    if len(timeseries) < 3:
        return []

    inflections = []
    for i in range(1, len(timeseries) - 1):
        if timeseries[i-1] > 0:
            prev_change = (timeseries[i] - timeseries[i-1]) / timeseries[i-1]
            next_change = (timeseries[i+1] - timeseries[i]) / timeseries[i] if timeseries[i] > 0 else 0

            # Sign change or large magnitude change
            if (prev_change * next_change < 0 and abs(prev_change) > threshold) or abs(prev_change) > 0.3:
                inflections.append({
                    'year': years[i],
                    'change': round(prev_change * 100, 1)
                })

    return sorted(inflections, key=lambda x: abs(x['change']), reverse=True)[:10]


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("NEGATIVE vs POSITIVE FREEDOM ANALYSIS (Google Ngrams)")
    print("=" * 70)
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Core comparison phrases
    core_phrases = [
        "freedom from",
        "freedom to",
    ]

    # Specific negative freedom phrases (freedom FROM X)
    negative_phrases = [
        "freedom from slavery",
        "freedom from sin",
        "freedom from want",
        "freedom from fear",
        "freedom from oppression",
        "freedom from tyranny",
        "freedom from bondage",
        "freedom from debt",
        "freedom from persecution",
        "freedom from constraint",
    ]

    # Specific positive freedom phrases (freedom TO Y)
    positive_phrases = [
        "freedom to choose",
        "freedom to worship",
        "freedom to speak",
        "freedom to vote",
        "freedom to work",
        "freedom to travel",
        "freedom to marry",
        "freedom to think",
        "freedom to act",
        "freedom to live",
    ]

    # Related phrases
    related_phrases = [
        "right to freedom",
        "liberty from",
        "liberty to",
        "free from",
        "free to",
    ]

    results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'source': 'Google Ngrams',
            'corpus': 'English 2019',
            'coverage': '1800-2019',
            'smoothing': 3,
        },
        'core_comparison': {},
        'negative_phrases': {},
        'positive_phrases': {},
        'related_phrases': {},
        'analysis': {},
    }

    # Query core phrases
    print("Querying core phrases...")
    print("-" * 70)
    for phrase in core_phrases:
        print(f"  {phrase}...")
        data = query_ngram(phrase)
        results['core_comparison'][phrase] = {
            'timeseries': data.get('timeseries', []),
            'years': data.get('years', []),
            'decade_averages': compute_decade_averages(
                data.get('timeseries', []),
                data.get('years', [])
            ),
        }
        time.sleep(1)  # Rate limiting

    # Query negative phrases
    print("\nQuerying negative freedom phrases (freedom FROM X)...")
    print("-" * 70)
    for phrase in negative_phrases:
        print(f"  {phrase}...")
        data = query_ngram(phrase)
        ts = data.get('timeseries', [])
        years = data.get('years', [])

        results['negative_phrases'][phrase] = {
            'decade_averages': compute_decade_averages(ts, years),
            'max_year': years[ts.index(max(ts))] if ts else None,
            'max_value': max(ts) if ts else 0,
            'total_sum': sum(ts) if ts else 0,
        }
        time.sleep(1)

    # Query positive phrases
    print("\nQuerying positive freedom phrases (freedom TO Y)...")
    print("-" * 70)
    for phrase in positive_phrases:
        print(f"  {phrase}...")
        data = query_ngram(phrase)
        ts = data.get('timeseries', [])
        years = data.get('years', [])

        results['positive_phrases'][phrase] = {
            'decade_averages': compute_decade_averages(ts, years),
            'max_year': years[ts.index(max(ts))] if ts else None,
            'max_value': max(ts) if ts else 0,
            'total_sum': sum(ts) if ts else 0,
        }
        time.sleep(1)

    # Query related phrases
    print("\nQuerying related phrases...")
    print("-" * 70)
    for phrase in related_phrases:
        print(f"  {phrase}...")
        data = query_ngram(phrase)
        ts = data.get('timeseries', [])
        years = data.get('years', [])

        results['related_phrases'][phrase] = {
            'decade_averages': compute_decade_averages(ts, years),
        }
        time.sleep(1)

    # Compute core analysis
    print("\nComputing analysis...")
    print("-" * 70)

    freedom_from = results['core_comparison'].get('freedom from', {})
    freedom_to = results['core_comparison'].get('freedom to', {})

    if freedom_from.get('timeseries') and freedom_to.get('timeseries'):
        ts_from = freedom_from['timeseries']
        ts_to = freedom_to['timeseries']
        years = freedom_from['years']

        # Compute ratio over time
        ratio_data = []
        for i, (vf, vt) in enumerate(zip(ts_from, ts_to)):
            if vt > 0:
                ratio_data.append({
                    'year': years[i],
                    'from': vf,
                    'to': vt,
                    'ratio_from_to': vf / vt,
                    'pct_from': vf / (vf + vt) * 100,
                    'pct_to': vt / (vf + vt) * 100,
                })

        # Decade summary
        decade_ratios = {}
        for entry in ratio_data:
            decade = (entry['year'] // 10) * 10
            if decade not in decade_ratios:
                decade_ratios[decade] = {'from': [], 'to': [], 'ratio': []}
            decade_ratios[decade]['from'].append(entry['from'])
            decade_ratios[decade]['to'].append(entry['to'])
            decade_ratios[decade]['ratio'].append(entry['ratio_from_to'])

        decade_summary = {}
        for decade, vals in sorted(decade_ratios.items()):
            avg_from = sum(vals['from']) / len(vals['from'])
            avg_to = sum(vals['to']) / len(vals['to'])
            avg_ratio = sum(vals['ratio']) / len(vals['ratio'])
            pct_from = avg_from / (avg_from + avg_to) * 100

            decade_summary[str(decade)] = {
                'avg_freedom_from': avg_from,
                'avg_freedom_to': avg_to,
                'ratio_from_to': round(avg_ratio, 3),
                'pct_negative': round(pct_from, 1),
                'pct_positive': round(100 - pct_from, 1),
            }

        # Find crossover point (if any)
        crossover_year = None
        for entry in ratio_data:
            if entry['ratio_from_to'] < 1:  # "freedom to" exceeds "freedom from"
                crossover_year = entry['year']
                break

        # Compute totals for negative vs positive specific phrases
        total_negative = sum(p['total_sum'] for p in results['negative_phrases'].values())
        total_positive = sum(p['total_sum'] for p in results['positive_phrases'].values())

        results['analysis'] = {
            'decade_summary': decade_summary,
            'crossover_year': crossover_year,
            'inflection_points_from': find_inflection_points(ts_from, years),
            'inflection_points_to': find_inflection_points(ts_to, years),
            'specific_phrases': {
                'total_negative_phrases': total_negative,
                'total_positive_phrases': total_positive,
                'ratio_negative_positive': round(total_negative / total_positive, 3) if total_positive > 0 else None,
            },
            'growth_1800_to_2019': {
                'freedom_from': round(ts_from[-1] / ts_from[0], 2) if ts_from[0] > 0 else None,
                'freedom_to': round(ts_to[-1] / ts_to[0], 2) if ts_to[0] > 0 else None,
            },
            'growth_1900_to_2019': {
                'freedom_from': round(ts_from[100] / ts_from[0], 2) if len(ts_from) > 100 and ts_from[0] > 0 else None,
                'freedom_to': round(ts_to[100] / ts_to[0], 2) if len(ts_to) > 100 and ts_to[0] > 0 else None,
            },
        }

    # Print results
    print()
    print("=" * 70)
    print("RESULTS: Negative vs Positive Freedom Framing")
    print("=" * 70)

    if results['analysis'].get('decade_summary'):
        print("\nDecade-by-Decade Summary:")
        print(f"{'Decade':<10} {'% Negative':>12} {'% Positive':>12} {'Ratio (FROM/TO)':>16}")
        print("-" * 50)
        for decade, data in sorted(results['analysis']['decade_summary'].items()):
            print(f"{decade:<10} {data['pct_negative']:>11.1f}% {data['pct_positive']:>11.1f}% {data['ratio_from_to']:>16.3f}")

        print(f"\nCrossover year (when 'freedom to' > 'freedom from'): {results['analysis'].get('crossover_year', 'Never')}")

        print("\nGrowth Rates:")
        growth = results['analysis'].get('growth_1800_to_2019', {})
        print(f"  'freedom from' (1800→2019): {growth.get('freedom_from', 'N/A')}x")
        print(f"  'freedom to' (1800→2019): {growth.get('freedom_to', 'N/A')}x")

    print("\nTop Negative Phrases (by total frequency):")
    sorted_neg = sorted(results['negative_phrases'].items(),
                        key=lambda x: x[1]['total_sum'], reverse=True)
    for phrase, data in sorted_neg[:5]:
        print(f"  {phrase}: peak {data['max_year']}")

    print("\nTop Positive Phrases (by total frequency):")
    sorted_pos = sorted(results['positive_phrases'].items(),
                        key=lambda x: x[1]['total_sum'], reverse=True)
    for phrase, data in sorted_pos[:5]:
        print(f"  {phrase}: peak {data['max_year']}")

    # Save results
    output_path = output_dir / 'negative_positive_ngrams.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Key findings
    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if results['analysis'].get('decade_summary'):
        first_decade = list(results['analysis']['decade_summary'].values())[0]
        last_decade = list(results['analysis']['decade_summary'].values())[-1]

        print(f"""
1. In the 1800s, "freedom from" was {first_decade['ratio_from_to']:.1f}x more common than "freedom to"
   ({first_decade['pct_negative']:.0f}% negative framing)

2. By the 2010s, the ratio changed to {last_decade['ratio_from_to']:.1f}x
   ({last_decade['pct_negative']:.0f}% negative / {last_decade['pct_positive']:.0f}% positive)

3. Crossover year: {results['analysis'].get('crossover_year', 'No crossover - negative still dominates')}

4. "Freedom to" grew {results['analysis']['growth_1800_to_2019'].get('freedom_to', 'N/A')}x vs
   "freedom from" grew {results['analysis']['growth_1800_to_2019'].get('freedom_from', 'N/A')}x
""")


if __name__ == '__main__':
    main()
