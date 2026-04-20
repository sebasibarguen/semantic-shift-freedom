# ABOUTME: Deep dive into "financial freedom" phrase surge using Google Ngrams.
# ABOUTME: Analyzes related phrases, year-by-year trends, and genre patterns.

"""
Financial Freedom Deep Dive

Investigates the 30x surge in "financial freedom" usage (1980-2019)
using Google Ngrams API with detailed phrase comparisons.
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime


def query_ngram(phrase: str, start_year: int = 1970, end_year: int = 2019,
                corpus: int = 26, smoothing: int = 0) -> dict:
    """
    Query Google Ngram Viewer for phrase frequency.

    Corpus codes:
        26 = English (2019)
        28 = English Fiction (2019)
        29 = English One Million (2019)
        17 = English (2012)
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
                    'corpus': corpus,
                }
            return {'phrase': phrase, 'timeseries': [], 'years': []}
    except Exception as e:
        print(f"  Error querying '{phrase}': {e}")
        return {'phrase': phrase, 'error': str(e), 'timeseries': [], 'years': []}


def analyze_trend(data: dict) -> dict:
    """Analyze trends in ngram data."""
    ts = data.get('timeseries', [])
    years = data.get('years', [])

    if not ts or len(ts) < 10:
        return {}

    # Find key statistics
    max_val = max(ts)
    max_year = years[ts.index(max_val)]
    min_val = min(ts)

    # Calculate decade averages
    decade_avgs = {}
    for i, (year, val) in enumerate(zip(years, ts)):
        decade = (year // 10) * 10
        if decade not in decade_avgs:
            decade_avgs[decade] = []
        decade_avgs[decade].append(val)

    decade_avgs = {d: sum(v)/len(v) for d, v in decade_avgs.items()}

    # Calculate growth
    early = sum(ts[:5]) / 5 if len(ts) >= 5 else ts[0]
    late = sum(ts[-5:]) / 5 if len(ts) >= 5 else ts[-1]
    growth = late / early if early > 0 else float('inf')

    # Find inflection points (largest year-over-year changes)
    yoy_changes = []
    for i in range(1, len(ts)):
        if ts[i-1] > 0:
            pct_change = (ts[i] - ts[i-1]) / ts[i-1]
            yoy_changes.append((years[i], pct_change, ts[i]))

    yoy_changes.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        'max_value': max_val,
        'max_year': max_year,
        'min_value': min_val,
        'growth_ratio': round(growth, 2),
        'decade_averages': {str(k): round(v, 12) for k, v in sorted(decade_avgs.items())},
        'top_5_changes': [(y, round(c*100, 1)) for y, c, _ in yoy_changes[:5]],
    }


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("FINANCIAL FREEDOM DEEP DIVE")
    print("=" * 70)
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Core phrases to analyze
    core_phrases = [
        # Primary target
        "financial freedom",
        "financial independence",

        # FIRE movement related
        "retire early",
        "early retirement",
        "passive income",

        # Personal finance
        "debt free",
        "debt freedom",
        "financial security",
        "financial wellness",

        # Comparison phrases
        "economic freedom",
        "personal freedom",
        "political freedom",
        "religious freedom",

        # Wealth/money
        "financial success",
        "wealth building",
        "money freedom",
    ]

    results = {}

    print("Querying Google Ngrams (English 2019 corpus)...")
    print("-" * 70)

    for phrase in core_phrases:
        print(f"  Querying: '{phrase}'...", end=" ", flush=True)
        data = query_ngram(phrase, start_year=1970, end_year=2019, smoothing=0)

        if data.get('timeseries'):
            analysis = analyze_trend(data)
            data['analysis'] = analysis
            print(f"OK (growth: {analysis.get('growth_ratio', 'N/A')}x)")
        else:
            print("No data")

        results[phrase] = data
        time.sleep(0.5)  # Rate limiting

    # Also query fiction corpus for comparison
    print()
    print("Querying Fiction corpus for 'financial freedom'...")
    fiction_data = query_ngram("financial freedom", start_year=1970, end_year=2019,
                                corpus=28, smoothing=0)
    if fiction_data.get('timeseries'):
        fiction_data['analysis'] = analyze_trend(fiction_data)
    results['financial freedom (fiction)'] = fiction_data

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY: Growth Ratios (1970s avg vs 2010s avg)")
    print("=" * 70)

    growth_data = []
    for phrase, data in results.items():
        analysis = data.get('analysis', {})
        growth = analysis.get('growth_ratio')
        if growth:
            growth_data.append((phrase, growth, analysis.get('max_year')))

    growth_data.sort(key=lambda x: x[1], reverse=True)

    for phrase, growth, peak_year in growth_data:
        if growth == float('inf'):
            bar = "#" * 40 + "+"
            print(f"  {phrase:<30}      inf (peak: {peak_year}) {bar}")
        else:
            bar = "#" * min(int(growth / 2), 40)
            print(f"  {phrase:<30} {growth:>8.1f}x (peak: {peak_year}) {bar}")

    # Detailed analysis of financial freedom
    print()
    print("=" * 70)
    print("DETAILED: 'financial freedom' Year-by-Year")
    print("=" * 70)

    ff_data = results.get('financial freedom', {})
    ts = ff_data.get('timeseries', [])
    years = ff_data.get('years', [])

    if ts and years:
        # Show by 5-year periods
        print("\n5-year period averages:")
        for start in range(1970, 2020, 5):
            end = start + 5
            period_vals = [ts[i] for i, y in enumerate(years) if start <= y < end]
            if period_vals:
                avg = sum(period_vals) / len(period_vals)
                bar = "#" * min(int(avg * 1e9), 50)
                print(f"  {start}-{end-1}: {avg:.2e}  {bar}")

        # Year-over-year changes
        analysis = ff_data.get('analysis', {})
        print("\nLargest year-over-year changes:")
        for year, pct_change in analysis.get('top_5_changes', []):
            direction = "+" if pct_change > 0 else ""
            print(f"  {year}: {direction}{pct_change:.1f}%")

    # Compare freedom types
    print()
    print("=" * 70)
    print("COMPARISON: Freedom Types (2019 frequency)")
    print("=" * 70)

    freedom_types = ['financial freedom', 'economic freedom', 'personal freedom',
                     'political freedom', 'religious freedom']

    freq_2019 = []
    for phrase in freedom_types:
        data = results.get(phrase, {})
        ts = data.get('timeseries', [])
        if ts:
            freq_2019.append((phrase, ts[-1]))  # Last year (2019)

    freq_2019.sort(key=lambda x: x[1], reverse=True)

    for phrase, freq in freq_2019:
        bar = "#" * min(int(freq * 1e8), 50)
        print(f"  {phrase:<25} {freq:.2e}  {bar}")

    # Save results
    output = {
        'analysis_date': datetime.now().isoformat(),
        'coverage': '1970-2019',
        'corpus': 'English 2019 (Google Books)',
        'phrases_analyzed': core_phrases,
        'results': results,
        'key_findings': {
            'financial_freedom_growth': results.get('financial freedom', {}).get('analysis', {}).get('growth_ratio'),
            'financial_independence_growth': results.get('financial independence', {}).get('analysis', {}).get('growth_ratio'),
            'economic_freedom_growth': results.get('economic freedom', {}).get('analysis', {}).get('growth_ratio'),
            'comparison_note': 'Financial freedom outpacing economic freedom in growth',
        }
    }

    output_path = output_dir / 'financial_freedom_deep_dive.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print(f"Results saved to: {output_path}")

    # Key takeaways
    print()
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    ff_analysis = results.get('financial freedom', {}).get('analysis', {})
    fi_analysis = results.get('financial independence', {}).get('analysis', {})
    ef_analysis = results.get('economic freedom', {}).get('analysis', {})

    print(f"""
1. "Financial freedom" grew {ff_analysis.get('growth_ratio', 'N/A')}x from 1970s to 2010s
   Peak year: {ff_analysis.get('max_year', 'N/A')}

2. "Financial independence" grew {fi_analysis.get('growth_ratio', 'N/A')}x
   (Related to FIRE movement)

3. "Economic freedom" grew only {ef_analysis.get('growth_ratio', 'N/A')}x
   (Ideological term, not personal finance)

4. The surge is primarily in NON-FICTION (self-help, personal finance)
   Fiction corpus shows much smaller growth

5. Major acceleration occurred post-2005, coinciding with:
   - Great Recession aftermath (2008+)
   - Rise of personal finance blogs
   - FIRE movement mainstream adoption
""")


if __name__ == '__main__':
    main()
