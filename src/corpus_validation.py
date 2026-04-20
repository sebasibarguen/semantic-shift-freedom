# ABOUTME: Validates findings across overlapping corpora (COHA vs Google Ngrams).
# ABOUTME: Checks that "freedom from" vs "freedom to" patterns agree across data sources.

import csv
import json
from pathlib import Path


def load_coha_collocates(csv_path):
    """Load COHA collocate frequencies by decade from CSV."""
    collocates = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)

        # Find decade columns (they start at index 6)
        # Header format: HELP, Picture, Picture, RE-USE WORDS, ALL, <blank>, 1820, 1830, ...
        decade_cols = {}
        for i, col in enumerate(header):
            col = col.strip()
            if col.isdigit() and 1800 <= int(col) <= 2020:
                decade_cols[int(col)] = i

        for row in reader:
            if len(row) < 5:
                continue
            word = row[3].strip()
            total = row[4].strip()
            if not word:
                continue

            freqs = {}
            for decade, col_idx in decade_cols.items():
                if col_idx < len(row):
                    try:
                        freqs[decade] = int(row[col_idx].strip()) if row[col_idx].strip() else 0
                    except ValueError:
                        freqs[decade] = 0
            collocates[word] = {"total": total, "by_decade": freqs}

    return collocates


def load_ngrams_results(json_path):
    """Load previously computed Google Ngrams results."""
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_coha_from_to_ratio(collocates):
    """Compute FROM/TO ratio from COHA collocate data."""
    from_data = collocates.get("FROM", {}).get("by_decade", {})
    to_data = collocates.get("TO", {}).get("by_decade", {})

    ratios = {}
    for decade in sorted(set(from_data.keys()) | set(to_data.keys())):
        from_count = from_data.get(decade, 0)
        to_count = to_data.get(decade, 0)
        total = from_count + to_count
        if total > 0:
            ratios[decade] = {
                "from_count": from_count,
                "to_count": to_count,
                "pct_from": round(from_count / total * 100, 1),
                "pct_to": round(to_count / total * 100, 1),
                "ratio_from_to": round(from_count / to_count, 3) if to_count > 0 else None,
            }
    return ratios


def compute_ngrams_decade_ratios(ngrams_data):
    """Extract decade ratios from Google Ngrams results."""
    decade_summary = ngrams_data.get("analysis", {}).get("decade_summary", {})
    ratios = {}
    for decade_str, data in decade_summary.items():
        decade = int(decade_str)
        ratios[decade] = {
            "pct_from": data.get("pct_negative", 0),
            "pct_to": data.get("pct_positive", 0),
            "ratio_from_to": data.get("ratio_from_to", None),
        }
    return ratios


def run_analysis():
    project_root = Path(__file__).parent.parent
    coha_path = project_root / "data" / "coha" / "collocates" / "coha_freedom_collocates_all.csv"
    ngrams_path = project_root / "outputs" / "negative_positive_ngrams.json"

    print("=" * 70)
    print("CORPUS OVERLAP VALIDATION")
    print("Comparing COHA collocates vs Google Ngrams")
    print("=" * 70)
    print()

    results = {"coha": {}, "ngrams": {}, "comparison": {}}

    # =========================================================================
    # LOAD COHA
    # =========================================================================
    print("Loading COHA collocate data...")
    coha = load_coha_collocates(coha_path)

    coha_ratios = compute_coha_from_to_ratio(coha)
    results["coha"] = {str(k): v for k, v in coha_ratios.items()}

    print(f"  COHA decades available: {sorted(coha_ratios.keys())}")
    print(f"  FROM total: {coha.get('FROM', {}).get('total', 'N/A')}")
    print(f"  TO total: {coha.get('TO', {}).get('total', 'N/A')}")
    print()

    # =========================================================================
    # LOAD NGRAMS
    # =========================================================================
    if ngrams_path.exists():
        print("Loading Google Ngrams results...")
        ngrams_data = load_ngrams_results(ngrams_path)
        ngrams_ratios = compute_ngrams_decade_ratios(ngrams_data)
        results["ngrams"] = {str(k): v for k, v in ngrams_ratios.items()}
        print(f"  Ngrams decades available: {sorted(ngrams_ratios.keys())}")
    else:
        print(f"  WARNING: {ngrams_path} not found. Run negative_positive_ngrams.py first.")
        ngrams_ratios = {}
    print()

    # =========================================================================
    # SIDE-BY-SIDE COMPARISON
    # =========================================================================
    print("=" * 70)
    print("SIDE-BY-SIDE: FROM/TO Ratio by Decade")
    print("=" * 70)
    print()

    overlap_decades = sorted(set(coha_ratios.keys()) & set(ngrams_ratios.keys()))

    print(f"{'Decade':<10} {'COHA %FROM':>12} {'COHA %TO':>10} {'Ngrams %FROM':>14} {'Ngrams %TO':>12} {'Agree?':>8}")
    print("-" * 70)

    agreements = 0
    disagreements = 0
    coha_pct_from_list = []
    ngrams_pct_from_list = []

    for decade in overlap_decades:
        c = coha_ratios[decade]
        n = ngrams_ratios[decade]

        # Do they agree on which is dominant?
        coha_dominant = "FROM" if c["pct_from"] > c["pct_to"] else "TO"
        ngrams_dominant = "FROM" if n["pct_from"] > n["pct_to"] else "TO"
        agree = coha_dominant == ngrams_dominant
        if agree:
            agreements += 1
        else:
            disagreements += 1

        coha_pct_from_list.append(c["pct_from"])
        ngrams_pct_from_list.append(n["pct_from"])

        marker = "  YES" if agree else "  NO ***"
        print(f"  {decade:<8} {c['pct_from']:>10.1f}% {c['pct_to']:>8.1f}% {n['pct_from']:>12.1f}% {n['pct_to']:>10.1f}%{marker}")

    print()
    print(f"  Agreement: {agreements}/{agreements + disagreements} decades ({agreements / (agreements + disagreements) * 100:.0f}%)")

    # =========================================================================
    # CORRELATION
    # =========================================================================
    if coha_pct_from_list and ngrams_pct_from_list:
        coha_arr = [x for x in coha_pct_from_list]
        ngrams_arr = [x for x in ngrams_pct_from_list]

        # Pearson correlation (manual to avoid scipy dependency)
        n = len(coha_arr)
        mean_c = sum(coha_arr) / n
        mean_n = sum(ngrams_arr) / n
        cov = sum((c - mean_c) * (ng - mean_n) for c, ng in zip(coha_arr, ngrams_arr)) / n
        std_c = (sum((c - mean_c) ** 2 for c in coha_arr) / n) ** 0.5
        std_n = (sum((ng - mean_n) ** 2 for ng in ngrams_arr) / n) ** 0.5

        if std_c > 0 and std_n > 0:
            correlation = cov / (std_c * std_n)
        else:
            correlation = 0

        results["comparison"] = {
            "overlap_decades": overlap_decades,
            "agreement_rate": round(agreements / (agreements + disagreements), 3),
            "pearson_r": round(correlation, 4),
            "mean_absolute_difference": round(
                sum(abs(c - n) for c, n in zip(coha_pct_from_list, ngrams_pct_from_list)) / len(coha_pct_from_list), 2
            ),
        }

        print()
        print(f"  Pearson r (COHA %FROM vs Ngrams %FROM): {correlation:.4f}")
        print(f"  Mean absolute difference: {results['comparison']['mean_absolute_difference']:.1f} percentage points")

    print()

    # =========================================================================
    # TREND COMPARISON
    # =========================================================================
    print("=" * 70)
    print("TREND COMPARISON")
    print("=" * 70)
    print()

    if overlap_decades:
        # Early vs late comparison
        early = [d for d in overlap_decades if d < 1900]
        late = [d for d in overlap_decades if d >= 1960]

        if early and late:
            coha_early = sum(coha_ratios[d]["pct_from"] for d in early) / len(early)
            coha_late = sum(coha_ratios[d]["pct_from"] for d in late) / len(late)
            ngrams_early = sum(ngrams_ratios[d]["pct_from"] for d in early) / len(early)
            ngrams_late = sum(ngrams_ratios[d]["pct_from"] for d in late) / len(late)

            coha_trend = "declining" if coha_late < coha_early else "increasing"
            ngrams_trend = "declining" if ngrams_late < ngrams_early else "increasing"

            print(f"  COHA %FROM trend:   {coha_early:.1f}% (pre-1900) → {coha_late:.1f}% (1960+) = {coha_trend}")
            print(f"  Ngrams %FROM trend: {ngrams_early:.1f}% (pre-1900) → {ngrams_late:.1f}% (1960+) = {ngrams_trend}")
            print()

            trends_agree = coha_trend == ngrams_trend
            print(f"  Trends agree: {'YES' if trends_agree else 'NO'}")
            results["comparison"]["trends_agree"] = trends_agree

    # =========================================================================
    # COHA TOP COLLOCATES
    # =========================================================================
    print()
    print("=" * 70)
    print("COHA TOP COLLOCATES (content words only)")
    print("=" * 70)
    print()

    stop_words = {
        "OF", "THE", ",", ".", "AND", "TO", "IN", "FOR", "A", "''", "IS", "THAT",
        "WITH", "FROM", "HIS", "'S", "THEIR", "AS", "WAS", "IT", "HE", ";",
        "WHICH", "I", "NOT", "?", "BUT", "THIS", "HAVE", "BE", "ARE", "HAS",
        "ALL", "OR", "ITS", "ON", "AN", "BY", "THEY", "BEEN", "WHO", "HAD",
        "WOULD", "AT", "WE", "HER", "WILL", "``", "NO", "THEM", "SHE", "DO",
        "OUR", "WERE", "MY", "ONE", "MAN", "THAN", "CAN", "IF", "COULD",
        "THERE", "SO", "WHAT", "SHOULD", "MAY", "MEN", "EVERY", "THOSE",
        "MORE", "OTHER", "ANY", "YOU", "THESE", "MUST", "ABOUT", "SOME",
        "SUCH", "ALSO", "MOST", "OWN", "NEW", "ONLY", "GREAT", "--", "VERY",
    }

    content_collocates = {
        k: v for k, v in coha.items()
        if k not in stop_words and k.isalpha()
    }

    # Show top 20 by total
    sorted_collocates = sorted(content_collocates.items(),
                                key=lambda x: int(x[1]["total"]) if x[1]["total"].isdigit() else 0,
                                reverse=True)[:20]

    print(f"{'Word':<20} {'Total':>8} {'1820':>6} {'1860':>6} {'1900':>6} {'1940':>6} {'1960':>6} {'1980':>6} {'2010':>6}")
    print("-" * 80)

    for word, data in sorted_collocates:
        decades_data = data["by_decade"]
        print(f"  {word:<18} {data['total']:>8} "
              f"{decades_data.get(1820, 0):>6} {decades_data.get(1860, 0):>6} "
              f"{decades_data.get(1900, 0):>6} {decades_data.get(1940, 0):>6} "
              f"{decades_data.get(1960, 0):>6} {decades_data.get(1980, 0):>6} "
              f"{decades_data.get(2010, 0):>6}")

    results["coha_top_collocates"] = [
        {"word": w, "total": d["total"]} for w, d in sorted_collocates
    ]

    # =========================================================================
    # SAVE
    # =========================================================================
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "corpus_validation.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {output_file}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    if results.get("comparison", {}).get("pearson_r") is not None:
        r = results["comparison"]["pearson_r"]
        agreement = results["comparison"]["agreement_rate"]
        if r > 0.7 and agreement > 0.7:
            print("STRONG CONVERGENCE: COHA and Google Ngrams tell a consistent story.")
            print(f"  Correlation: r={r:.3f}, Agreement: {agreement*100:.0f}%")
        elif r > 0.4:
            print("MODERATE CONVERGENCE: COHA and Ngrams partially agree.")
            print(f"  Correlation: r={r:.3f}, Agreement: {agreement*100:.0f}%")
            print("  Differences may reflect genre composition or corpus construction.")
        else:
            print("WEAK CONVERGENCE: COHA and Ngrams show different patterns.")
            print(f"  Correlation: r={r:.3f}, Agreement: {agreement*100:.0f}%")
            print("  Findings may be corpus-dependent — interpret with caution.")

    return results


if __name__ == "__main__":
    run_analysis()
