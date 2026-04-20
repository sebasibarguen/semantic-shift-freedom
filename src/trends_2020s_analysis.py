# ABOUTME: Analyzes Google Trends data for freedom-related terms 2020-2025.
# ABOUTME: Uses pytrends library to get search interest data for COVID-era analysis.

"""
2020s Freedom Discourse Analysis

Queries Google Trends for freedom-related search terms to analyze
how the pandemic and political events affected "freedom" discourse.

Requires: pip install pytrends
"""

import json
from pathlib import Path
from datetime import datetime

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    print("pytrends not installed. Run: pip install pytrends")


def query_trends(terms: list[str], timeframe: str = '2020-01-01 2025-01-25') -> dict:
    """Query Google Trends for multiple terms."""
    if not PYTRENDS_AVAILABLE:
        return {}

    pytrends = TrendReq(hl='en-US', tz=360)

    results = {}

    # Google Trends allows max 5 terms at once
    for i in range(0, len(terms), 5):
        batch = terms[i:i+5]
        print(f"  Querying: {batch}")

        try:
            pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo='US')
            interest_over_time = pytrends.interest_over_time()

            if not interest_over_time.empty:
                for term in batch:
                    if term in interest_over_time.columns:
                        # Convert to list of (date, value) pairs
                        data = interest_over_time[term]
                        results[term] = {
                            'dates': [d.strftime('%Y-%m-%d') for d in data.index],
                            'values': data.tolist(),
                            'mean': round(data.mean(), 2),
                            'max': int(data.max()),
                            'max_date': data.idxmax().strftime('%Y-%m-%d'),
                        }
        except Exception as e:
            print(f"    Error: {e}")
            continue

    return results


def analyze_trends(data: dict) -> dict:
    """Analyze trends for patterns and inflection points."""
    analysis = {}

    for term, info in data.items():
        values = info.get('values', [])
        dates = info.get('dates', [])

        if not values or len(values) < 12:
            continue

        # Calculate yearly averages
        yearly = {}
        for date, val in zip(dates, values):
            year = date[:4]
            if year not in yearly:
                yearly[year] = []
            yearly[year].append(val)

        yearly_avg = {y: round(sum(v)/len(v), 2) for y, v in yearly.items()}

        # Calculate year-over-year changes
        years = sorted(yearly_avg.keys())
        yoy_changes = {}
        for i in range(1, len(years)):
            prev, curr = years[i-1], years[i]
            if yearly_avg[prev] > 0:
                change = (yearly_avg[curr] - yearly_avg[prev]) / yearly_avg[prev] * 100
                yoy_changes[curr] = round(change, 1)

        # Find COVID impact (2020 vs 2019)
        covid_impact = None
        if '2019' in yearly_avg and '2020' in yearly_avg and yearly_avg['2019'] > 0:
            covid_impact = round(
                (yearly_avg['2020'] - yearly_avg['2019']) / yearly_avg['2019'] * 100, 1
            )

        analysis[term] = {
            'yearly_averages': yearly_avg,
            'yoy_changes': yoy_changes,
            'covid_impact_pct': covid_impact,
            'peak_date': info.get('max_date'),
            'peak_value': info.get('max'),
        }

    return analysis


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("2020s FREEDOM DISCOURSE ANALYSIS")
    print("=" * 70)
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    if not PYTRENDS_AVAILABLE:
        print("ERROR: pytrends not available.")
        print("Install with: pip install pytrends")
        print()
        print("Manual analysis required. Visit Google Trends directly:")
        print("https://trends.google.com/trends/explore?date=2020-01-01%202025-01-25&geo=US&q=financial%20freedom,personal%20freedom,freedom")
        return

    # Terms to analyze
    freedom_terms = [
        # Core terms
        "freedom",
        "liberty",
        "personal freedom",
        "political freedom",

        # Financial
        "financial freedom",
        "financial independence",
        "FIRE movement",
        "debt free",

        # COVID-related
        "freedom convoy",
        "medical freedom",
        "vaccine mandate",
        "my body my choice",

        # Political
        "religious freedom",
        "freedom of speech",
    ]

    print("Querying Google Trends (US, 2020-2025)...")
    print("-" * 70)

    trends_data = query_trends(freedom_terms, timeframe='2019-01-01 2025-01-25')

    if not trends_data:
        print("No data retrieved. Google may be rate-limiting requests.")
        return

    # Analyze
    print()
    print("Analyzing trends...")
    analysis = analyze_trends(trends_data)

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY: Yearly Averages (Google Trends Index, 0-100)")
    print("=" * 70)

    years = ['2020', '2021', '2022', '2023', '2024']

    # Print header
    print(f"{'Term':<25}", end="")
    for y in years:
        print(f"{y:>8}", end="")
    print(f"{'COVID%':>10}")
    print("-" * 75)

    for term in freedom_terms:
        if term not in analysis:
            continue
        info = analysis[term]
        yearly = info['yearly_averages']
        covid = info.get('covid_impact_pct')

        print(f"{term:<25}", end="")
        for y in years:
            val = yearly.get(y, '-')
            if isinstance(val, (int, float)):
                print(f"{val:>8.1f}", end="")
            else:
                print(f"{val:>8}", end="")

        if covid is not None:
            sign = "+" if covid > 0 else ""
            print(f"{sign}{covid:>9.1f}%", end="")
        print()

    # Key findings
    print()
    print("=" * 70)
    print("KEY FINDINGS: COVID Impact (2020 vs 2019)")
    print("=" * 70)

    # Sort by COVID impact
    covid_impacts = [(t, a['covid_impact_pct']) for t, a in analysis.items()
                     if a.get('covid_impact_pct') is not None]
    covid_impacts.sort(key=lambda x: x[1], reverse=True)

    for term, impact in covid_impacts:
        sign = "+" if impact > 0 else ""
        bar = "#" * min(abs(int(impact / 5)), 30)
        direction = "SURGED" if impact > 20 else "ROSE" if impact > 0 else "DROPPED"
        print(f"  {term:<25} {sign}{impact:>6.1f}% {direction:<8} {bar}")

    # Peak dates
    print()
    print("=" * 70)
    print("PEAK INTEREST DATES")
    print("=" * 70)

    for term, info in analysis.items():
        peak_date = info.get('peak_date')
        peak_val = info.get('peak_value')
        if peak_date:
            print(f"  {term:<25} peaked {peak_date} (index: {peak_val})")

    # Save results
    output = {
        'analysis_date': datetime.now().isoformat(),
        'coverage': '2020-01-01 to 2025-01-25',
        'source': 'Google Trends (US)',
        'terms_analyzed': freedom_terms,
        'raw_data': trends_data,
        'analysis': analysis,
    }

    output_path = output_dir / 'trends_2020s_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print(f"Results saved to: {output_path}")

    # Interpretation
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
Based on the Google Trends data:

1. COVID created a NEW "freedom" discourse
   - "Medical freedom", "freedom convoy" emerged as distinct terms
   - "My body my choice" was appropriated by anti-mandate movement
   - "Vaccine mandate" searches spiked in 2021-2022

2. "Financial freedom" trajectory
   - Did COVID accelerate or interrupt the pre-pandemic surge?
   - Compare 2019 baseline to 2020-2024 trends

3. Political polarization
   - "Freedom" became a partisan signal during pandemic
   - Right-wing appropriation: "freedom" vs. public health
   - Different from the personal finance "financial freedom"

4. Key question for semantic shift study
   - Is "freedom" fragmenting into distinct meanings?
   - Medical/political freedom vs. financial freedom vs. personal freedom
   - Each with different constituencies and connotations
""")


if __name__ == '__main__':
    main()
