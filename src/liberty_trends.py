# ABOUTME: Tests decade-level trends in positive/negative liberty label proportions.
# ABOUTME: Uses sentence-level LLM labels instead of surface grammar patterns.

import argparse
import json
from collections import Counter, defaultdict
from math import erf, sqrt
from pathlib import Path


LABELS = ("positive_liberty", "negative_liberty", "ambiguous", "other", "error", "missing")
PRIMARY_DENOMINATOR = "positive_plus_negative"
SENSITIVITY_DENOMINATOR = "substantive"


def normal_two_sided_p(z: float) -> float:
    """Approximate two-sided p-value from a normal z-score."""
    return float(2 * (1 - (0.5 * (1 + erf(abs(z) / sqrt(2))))))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return {"point": None, "lower": None, "upper": None}

    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return {
        "point": round(p, 6),
        "lower": round(max(0.0, center - margin), 6),
        "upper": round(min(1.0, center + margin), 6),
    }


def get_llm_label(record: dict) -> str:
    """Return normalized LLM label for a sentence record."""
    label = record.get("methods", {}).get("llm", {}).get("label")
    if label in LABELS:
        return label
    if label is None:
        return "missing"
    return "error"


def load_records(data_dir: Path) -> list[dict]:
    """Load all decade sentence files from web/data-style JSON files."""
    records: list[dict] = []
    for path in sorted(data_dir.glob("sentences_*s.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")
        records.extend(data)
    return records


def aggregate_by_decade(records: list[dict]) -> dict[str, dict]:
    """Aggregate labels and derived proportions by decade."""
    counts: dict[int, Counter] = defaultdict(Counter)
    totals: Counter[int] = Counter()

    for record in records:
        year = record.get("year")
        if year is None:
            continue
        decade = (int(year) // 10) * 10
        label = get_llm_label(record)
        counts[decade][label] += 1
        totals[decade] += 1

    by_decade: dict[str, dict] = {}
    for decade in sorted(counts):
        c = counts[decade]
        positive = c["positive_liberty"]
        negative = c["negative_liberty"]
        ambiguous = c["ambiguous"]
        other = c["other"]
        error = c["error"]
        missing = c["missing"]
        primary_n = positive + negative
        substantive_n = positive + negative + ambiguous

        by_decade[str(decade)] = {
            "total": totals[decade],
            "counts": {label: c[label] for label in LABELS},
            "denominators": {
                PRIMARY_DENOMINATOR: primary_n,
                SENSITIVITY_DENOMINATOR: substantive_n,
            },
            "positive_share_of_positive_negative": wilson_interval(positive, primary_n),
            "positive_share_of_substantive": wilson_interval(positive, substantive_n),
            "negative_share_of_substantive": wilson_interval(negative, substantive_n),
            "ambiguous_share_of_substantive": wilson_interval(ambiguous, substantive_n),
            "other_error_missing_share_all": wilson_interval(other + error + missing, totals[decade]),
        }

    return by_decade


def weighted_linear_trend(points: list[tuple[int, float, int]]) -> dict | None:
    """
    Weighted linear trend over decade-level proportions.

    Decades are the units of analysis; weights are denominators for each
    decade. This is intentionally a trend test over proportions, not an
    assertion that one category absolutely replaced another.
    """
    points = [(d, p, n) for d, p, n in points if n > 0]
    if len(points) < 3:
        return None

    decades = [p[0] for p in points]
    y = [p[1] for p in points]
    weights = [p[2] for p in points]

    x = [(d - sum(decades) / len(decades)) / 100 for d in decades]
    w_sum = sum(weights)
    x_bar = sum(w * xi for w, xi in zip(weights, x)) / w_sum
    y_bar = sum(w * yi for w, yi in zip(weights, y)) / w_sum
    ss_xx = sum(w * (xi - x_bar) ** 2 for w, xi in zip(weights, x))
    if ss_xx == 0:
        return None

    slope = sum(
        w * (xi - x_bar) * (yi - y_bar)
        for w, xi, yi in zip(weights, x, y)
    ) / ss_xx
    intercept = y_bar - slope * x_bar
    residuals = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
    rss = sum(w * r**2 for w, r in zip(weights, residuals))
    df = len(points) - 2
    se = sqrt((rss / df) / ss_xx) if df > 0 and ss_xx > 0 else 0.0
    z = slope / se if se > 0 else 0.0

    first_decade, first_p, first_n = points[0]
    last_decade, last_p, last_n = points[-1]
    endpoint_change = last_p - first_p
    endpoint_se = sqrt(
        first_p * (1 - first_p) / first_n
        + last_p * (1 - last_p) / last_n
    )
    endpoint_z = endpoint_change / endpoint_se if endpoint_se > 0 else 0.0

    return {
        "first_decade": first_decade,
        "last_decade": last_decade,
        "n_decades": len(points),
        "slope_per_century": round(slope, 6),
        "std_error": round(se, 6),
        "z": round(z, 3),
        "p_value_approx": round(normal_two_sided_p(z), 6) if se > 0 else None,
        "first_proportion": round(first_p, 6),
        "last_proportion": round(last_p, 6),
        "endpoint_change": round(endpoint_change, 6),
        "endpoint_z": round(endpoint_z, 3),
        "endpoint_p_value_approx": round(normal_two_sided_p(endpoint_z), 6)
        if endpoint_se > 0 else None,
    }


def build_trend_points(by_decade: dict[str, dict], proportion_key: str, denominator_key: str,
                       min_denominator: int) -> list[tuple[int, float, int]]:
    """Build trend input tuples from aggregate decade records."""
    points = []
    for decade_str, row in sorted(by_decade.items(), key=lambda item: int(item[0])):
        prop = row[proportion_key]["point"]
        denom = row["denominators"][denominator_key]
        if prop is None or denom < min_denominator:
            continue
        points.append((int(decade_str), prop, denom))
    return points


def run_analysis(data_dir: Path, min_denominator: int = 30) -> dict:
    """Run decade-level trend analysis over LLM liberty labels."""
    records = load_records(data_dir)
    by_decade = aggregate_by_decade(records)

    primary_points = build_trend_points(
        by_decade,
        "positive_share_of_positive_negative",
        PRIMARY_DENOMINATOR,
        min_denominator,
    )
    substantive_points = build_trend_points(
        by_decade,
        "positive_share_of_substantive",
        SENSITIVITY_DENOMINATOR,
        min_denominator,
    )
    negative_points = build_trend_points(
        by_decade,
        "negative_share_of_substantive",
        SENSITIVITY_DENOMINATOR,
        min_denominator,
    )
    ambiguous_points = build_trend_points(
        by_decade,
        "ambiguous_share_of_substantive",
        SENSITIVITY_DENOMINATOR,
        min_denominator,
    )

    return {
        "metadata": {
            "source": str(data_dir),
            "total_records": len(records),
            "min_denominator": min_denominator,
            "primary_hypothesis": (
                "The proportion of positive-liberty labels changes over time "
                "among records classified as positive or negative liberty."
            ),
            "note": (
                "This analysis tests trends in label proportions. It does not "
                "require an absolute switch from negative to positive liberty."
            ),
        },
        "by_decade": by_decade,
        "trend_tests": {
            "positive_share_of_positive_negative": weighted_linear_trend(primary_points),
            "positive_share_of_substantive": weighted_linear_trend(substantive_points),
            "negative_share_of_substantive": weighted_linear_trend(negative_points),
            "ambiguous_share_of_substantive": weighted_linear_trend(ambiguous_points),
        },
    }


def print_summary(results: dict) -> None:
    """Print compact trend summary."""
    meta = results["metadata"]
    primary = results["trend_tests"]["positive_share_of_positive_negative"]
    sensitivity = results["trend_tests"]["positive_share_of_substantive"]

    print("=" * 70)
    print("LIBERTY LABEL PROPORTION TRENDS")
    print("=" * 70)
    print(f"Records: {meta['total_records']:,}")
    print(meta["note"])
    print()

    for name, trend in [
        ("Primary: positive / (positive + negative)", primary),
        ("Sensitivity: positive / (positive + negative + ambiguous)", sensitivity),
    ]:
        if not trend:
            print(f"{name}: insufficient data")
            continue
        print(name)
        print(
            f"  {trend['first_decade']}={trend['first_proportion']:.3f}, "
            f"{trend['last_decade']}={trend['last_proportion']:.3f}, "
            f"change={trend['endpoint_change']:+.3f}"
        )
        print(
            f"  slope/century={trend['slope_per_century']:+.4f}, "
            f"p≈{trend['p_value_approx']}"
        )
        print()


def main() -> None:
    project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "web" / "data")
    parser.add_argument("--output", type=Path, default=project_root / "outputs" / "liberty_trends.json")
    parser.add_argument("--min-denominator", type=int, default=30)
    args = parser.parse_args()

    results = run_analysis(args.data_dir, min_denominator=args.min_denominator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print_summary(results)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
