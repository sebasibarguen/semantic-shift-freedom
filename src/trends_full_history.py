# ABOUTME: Queries Google Trends for the full available history (2004-2025).
# ABOUTME: Maximizes the pytrends timeframe for freedom-related terms.

"""
Google Trends Full History Analysis (2004-2025)

Queries the maximum available timeframe from Google Trends.
Data available from January 2004 to present.
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


def query_trends_full(terms: list[str], geo: str = 'US') -> dict:
    """Query Google Trends for full history (2004-present)."""
    if not PYTRENDS_AVAILABLE:
        return {}

    pytrends = TrendReq(hl='en-US', tz=360)
    results = {}

    # Use 'all' for full available history (2004-present)
    timeframe = 'all'

    for i in range(0, len(terms), 5):
        batch = terms[i:i+5]
        print(f"  Querying: {batch}")

        try:
            pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo)
            interest_over_time = pytrends.interest_over_time()

            if not interest_over_time.empty:
                for term in batch:
                    if term in interest_over_time.columns:
                        data = interest_over_time[term]
                        results[term] = {
                            'dates': [d.strftime('%Y-%m') for d in data.index],
                            'values': [int(v) for v in data.tolist()],
                            'first_date': data.index[0].strftime('%Y-%m'),
                            'last_date': data.index[-1].strftime('%Y-%m'),
                            'mean': round(data.mean(), 2),
                            'max': int(data.max()),
                            'max_date': data.idxmax().strftime('%Y-%m'),
                            'min': int(data.min()),
                        }
        except Exception as e:
            print(f"    Error: {e}")
            continue

    return results


def analyze_by_year(data: dict) -> dict:
    """Aggregate data by year for trend analysis."""
    analysis = {}

    for term, info in data.items():
        dates = info.get('dates', [])
        values = info.get('values', [])

        if not dates:
            continue

        # Group by year
        yearly = {}
        for date, val in zip(dates, values):
            year = date[:4]
            if year not in yearly:
                yearly[year] = []
            yearly[year].append(val)

        yearly_avg = {y: round(sum(v)/len(v), 1) for y, v in sorted(yearly.items())}

        # Calculate growth periods
        years = sorted(yearly_avg.keys())

        # 2004 vs 2024
        if '2004' in yearly_avg and '2024' in yearly_avg:
            total_growth = yearly_avg['2024'] / yearly_avg['2004'] if yearly_avg['2004'] > 0 else float('inf')
        else:
            total_growth = None

        # Find peak year
        peak_year = max(yearly_avg, key=yearly_avg.get) if yearly_avg else None
        peak_value = yearly_avg.get(peak_year)

        analysis[term] = {
            'yearly_avg': yearly_avg,
            'total_growth': round(total_growth, 2) if total_growth and total_growth != float('inf') else total_growth,
            'peak_year': peak_year,
            'peak_value': peak_value,
            'coverage': f"{years[0]}-{years[-1]}" if years else None,
        }

    return analysis


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("GOOGLE TRENDS FULL HISTORY (2004-2025)")
    print("=" * 70)
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    if not PYTRENDS_AVAILABLE:
        print("ERROR: pytrends not available")
        return

    # Core terms - freedom types
    freedom_terms = [
        "financial freedom",
        "personal freedom",
        "political freedom",
        "religious freedom",
        "economic freedom",
    ]

    # Additional terms
    additional_terms = [
        "freedom",
        "liberty",
        "FIRE movement",
        "debt free",
        "financial independence",
    ]

    all_terms = freedom_terms + additional_terms

    print("Querying Google Trends (US, 2004-present, monthly)...")
    print("-" * 70)

    trends_data = query_trends_full(all_terms)

    if not trends_data:
        print("No data retrieved.")
        return

    # Analyze
    print()
    print("Analyzing trends by year...")
    analysis = analyze_by_year(trends_data)

    # Print yearly summary table
    print()
    print("=" * 70)
    print("YEARLY AVERAGES (Google Trends Index, 0-100)")
    print("=" * 70)

    # Select key years
    key_years = ['2004', '2008', '2012', '2016', '2020', '2024']

    # Header
    print(f"{'Term':<25}", end="")
    for y in key_years:
        print(f"{y:>8}", end="")
    print(f"{'Growth':>10}")
    print("-" * 85)

    for term in all_terms:
        if term not in analysis:
            continue
        info = analysis[term]
        yearly = info['yearly_avg']
        growth = info.get('total_growth')

        print(f"{term:<25}", end="")
        for y in key_years:
            val = yearly.get(y, '-')
            if isinstance(val, (int, float)):
                print(f"{val:>8.1f}", end="")
            else:
                print(f"{val:>8}", end="")

        if growth is not None:
            if growth == float('inf'):
                print(f"{'inf':>10}", end="")
            else:
                print(f"{growth:>9.1f}x", end="")
        print()

    # Peak years
    print()
    print("=" * 70)
    print("PEAK YEARS")
    print("=" * 70)

    for term in all_terms:
        if term not in analysis:
            continue
        info = analysis[term]
        peak_year = info.get('peak_year')
        peak_val = info.get('peak_value')
        coverage = info.get('coverage')
        print(f"  {term:<25} peaked in {peak_year} (index: {peak_val}) [{coverage}]")

    # Detailed time series for key terms
    print()
    print("=" * 70)
    print("DETAILED: 'financial freedom' Year-by-Year")
    print("=" * 70)

    if 'financial freedom' in analysis:
        yearly = analysis['financial freedom']['yearly_avg']
        for year in sorted(yearly.keys()):
            val = yearly[year]
            bar = "#" * int(val / 2)
            print(f"  {year}: {val:>5.1f}  {bar}")

    # Compare freedom types
    print()
    print("=" * 70)
    print("COMPARISON: Freedom Types Over Time")
    print("=" * 70)

    comparison_terms = ['financial freedom', 'personal freedom', 'economic freedom', 'religious freedom']
    decades = ['2004', '2009', '2014', '2019', '2024']

    print(f"{'Year':<8}", end="")
    for term in comparison_terms:
        short = term.replace(' freedom', '')
        print(f"{short:>12}", end="")
    print()
    print("-" * 60)

    for year in decades:
        print(f"{year:<8}", end="")
        for term in comparison_terms:
            if term in analysis:
                val = analysis[term]['yearly_avg'].get(year, '-')
                if isinstance(val, (int, float)):
                    print(f"{val:>12.1f}", end="")
                else:
                    print(f"{val:>12}", end="")
            else:
                print(f"{'-':>12}", end="")
        print()

    # Save results
    output = {
        'analysis_date': datetime.now().isoformat(),
        'source': 'Google Trends',
        'coverage': '2004-01 to present',
        'geo': 'US',
        'granularity': 'monthly',
        'terms_analyzed': all_terms,
        'raw_data': trends_data,
        'yearly_analysis': analysis,
    }

    output_path = output_dir / 'trends_full_history.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print(f"Results saved to: {output_path}")

    # Key findings
    print()
    print("=" * 70)
    print("KEY FINDINGS (2004-2024)")
    print("=" * 70)

    if 'financial freedom' in analysis:
        ff = analysis['financial freedom']
        print(f"""
1. "Financial freedom" trajectory:
   - 2004: {ff['yearly_avg'].get('2004', 'N/A')}
   - 2024: {ff['yearly_avg'].get('2024', 'N/A')}
   - Peak: {ff['peak_year']} (index {ff['peak_value']})
   - Growth: {ff.get('total_growth', 'N/A')}x
""")

    if 'personal freedom' in analysis:
        pf = analysis['personal freedom']
        print(f"""2. "Personal freedom" trajectory:
   - 2004: {pf['yearly_avg'].get('2004', 'N/A')}
   - 2024: {pf['yearly_avg'].get('2024', 'N/A')}
   - Peak: {pf['peak_year']} (index {pf['peak_value']})
""")

    if 'economic freedom' in analysis:
        ef = analysis['economic freedom']
        print(f"""3. "Economic freedom" trajectory:
   - 2004: {ef['yearly_avg'].get('2004', 'N/A')}
   - 2024: {ef['yearly_avg'].get('2024', 'N/A')}
   - Peak: {ef['peak_year']} (index {ef['peak_value']})
""")


if __name__ == '__main__':
    main()
