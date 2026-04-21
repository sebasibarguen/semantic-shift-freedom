# ABOUTME: Google Trends analysis for freedom-related terms via pytrends.
# ABOUTME: Supports full history (2004-present) and COVID-era (2020-2025) modes.

"""
Google Trends analysis for freedom discourse.

Two modes:
    --range full    2004-present, core freedom types + FIRE/debt terms
    --range 2020s   2020-2025 with COVID-era terms (mandates, convoys)

Usage:
    uv run python -m src.trends --range full
    uv run python -m src.trends --range 2020s
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Term lists per mode
# ---------------------------------------------------------------------------

FULL_TERMS = [
    # Core freedom types
    "financial freedom",
    "personal freedom",
    "political freedom",
    "religious freedom",
    "economic freedom",
    # Adjacent
    "freedom",
    "liberty",
    "FIRE movement",
    "debt free",
    "financial independence",
]

TERMS_2020S = [
    # Core
    "freedom",
    "liberty",
    "personal freedom",
    "political freedom",
    # Financial
    "financial freedom",
    "financial independence",
    "FIRE movement",
    "debt free",
    # COVID-era
    "freedom convoy",
    "medical freedom",
    "vaccine mandate",
    "my body my choice",
    # Political
    "religious freedom",
    "freedom of speech",
]

MODES = {
    "full": {
        "terms": FULL_TERMS,
        "timeframe": "all",
        "output_name": "trends_full_history.json",
        "label": "2004-present",
        "yoy_reference_year": "2004",
    },
    "2020s": {
        "terms": TERMS_2020S,
        "timeframe": "2019-01-01 2025-01-25",
        "output_name": "trends_2020s_analysis.json",
        "label": "2019-2025",
        "yoy_reference_year": "2019",  # enables COVID impact calc
    },
}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_trends(terms: list[str], timeframe: str, geo: str = "US") -> dict:
    """Query pytrends in batches of 5 (the API max)."""
    if not PYTRENDS_AVAILABLE:
        return {}
    pytrends = TrendReq(hl="en-US", tz=360)
    results = {}
    for i in range(0, len(terms), 5):
        batch = terms[i:i + 5]
        print(f"  Querying: {batch}")
        try:
            pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
            if df.empty:
                continue
            for term in batch:
                if term not in df.columns:
                    continue
                data = df[term]
                results[term] = {
                    "dates": [d.strftime("%Y-%m-%d") for d in data.index],
                    "values": [int(v) for v in data.tolist()],
                    "first_date": data.index[0].strftime("%Y-%m-%d"),
                    "last_date": data.index[-1].strftime("%Y-%m-%d"),
                    "mean": round(float(data.mean()), 2),
                    "max": int(data.max()),
                    "max_date": data.idxmax().strftime("%Y-%m-%d"),
                    "min": int(data.min()),
                }
        except Exception as e:
            print(f"    Error: {e}")
            continue
    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(data: dict, reference_year: str | None = None) -> dict:
    """Yearly averages, YoY changes, peak, optional reference-year % change."""
    analysis = {}
    for term, info in data.items():
        dates = info.get("dates", [])
        values = info.get("values", [])
        if not dates:
            continue

        yearly: dict[str, list[int]] = {}
        for date, val in zip(dates, values):
            yearly.setdefault(date[:4], []).append(val)
        yearly_avg = {y: round(sum(v) / len(v), 2) for y, v in sorted(yearly.items())}
        years = sorted(yearly_avg.keys())

        yoy_changes = {}
        for i in range(1, len(years)):
            prev, curr = years[i - 1], years[i]
            if yearly_avg[prev] > 0:
                yoy_changes[curr] = round(
                    (yearly_avg[curr] - yearly_avg[prev]) / yearly_avg[prev] * 100, 1
                )

        # Reference-year growth (e.g. 2019→2020 for COVID impact, or 2004→latest for full history)
        ref_growth_pct = None
        total_growth_x = None
        if reference_year and reference_year in yearly_avg and years:
            latest = years[-1]
            if yearly_avg[reference_year] > 0:
                ref_growth_pct = round(
                    (yearly_avg[latest] - yearly_avg[reference_year]) / yearly_avg[reference_year] * 100, 1
                )
                total_growth_x = round(yearly_avg[latest] / yearly_avg[reference_year], 2)

        analysis[term] = {
            "yearly_averages": yearly_avg,
            "yoy_changes": yoy_changes,
            "ref_growth_pct": ref_growth_pct,
            "total_growth_x": total_growth_x,
            "coverage": f"{years[0]}-{years[-1]}" if years else None,
            "peak_year": max(yearly_avg, key=yearly_avg.get) if yearly_avg else None,
            "peak_value": max(yearly_avg.values()) if yearly_avg else None,
            "peak_date": info.get("max_date"),
        }
    return analysis


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_yearly_table(analysis: dict, terms: list[str], years: list[str], growth_label: str):
    header = f"{'Term':<25}" + "".join(f"{y:>8}" for y in years) + f"{growth_label:>10}"
    print(header)
    print("-" * len(header))
    for term in terms:
        if term not in analysis:
            continue
        info = analysis[term]
        yearly = info["yearly_averages"]
        row = f"{term:<25}"
        for y in years:
            val = yearly.get(y, "-")
            row += f"{val:>8.1f}" if isinstance(val, (int, float)) else f"{val:>8}"
        growth = info.get("ref_growth_pct")
        if growth is not None:
            sign = "+" if growth > 0 else ""
            row += f"{sign}{growth:>8.1f}%"
        print(row)


def key_years_for_mode(mode: str) -> list[str]:
    return (
        ["2004", "2008", "2012", "2016", "2020", "2024"] if mode == "full"
        else ["2019", "2020", "2021", "2022", "2023", "2024"]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(mode: str):
    cfg = MODES[mode]
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"GOOGLE TRENDS — {cfg['label']}")
    print("=" * 70)
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    if not PYTRENDS_AVAILABLE:
        raise SystemExit(
            "pytrends not installed. Run: uv pip install pytrends (or `uv sync`)"
        )

    print(f"Querying Google Trends (US, timeframe={cfg['timeframe']})...")
    data = query_trends(cfg["terms"], cfg["timeframe"])
    if not data:
        raise SystemExit("No data retrieved. Google may be rate-limiting.")

    print("\nAnalyzing...")
    analysis = analyze(data, reference_year=cfg["yoy_reference_year"])

    print()
    growth_label = "Δvs'04" if mode == "full" else "COVID%"
    print_yearly_table(analysis, cfg["terms"], key_years_for_mode(mode), growth_label)

    print("\nPeak years:")
    for term in cfg["terms"]:
        if term not in analysis:
            continue
        info = analysis[term]
        print(f"  {term:<25} peaked {info['peak_year']} (index {info['peak_value']}) [{info['coverage']}]")

    output = {
        "analysis_date": datetime.now().isoformat(),
        "mode": mode,
        "coverage": cfg["label"],
        "source": "Google Trends (US)",
        "terms_analyzed": cfg["terms"],
        "raw_data": data,
        "analysis": analysis,
    }
    output_path = output_dir / cfg["output_name"]
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved to: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Google Trends analysis for freedom terms.")
    p.add_argument("--range", choices=list(MODES), default="full", help="Which date range.")
    args = p.parse_args()
    run(args.range)


if __name__ == "__main__":
    main()
