# ABOUTME: Shared significance and trend helpers used across the analyses.
# ABOUTME: Deliberately stdlib-only so the label-trend pipeline stays numpy-free.

from math import erf, sqrt


def normal_two_sided_p(z: float) -> float:
    """Approximate two-sided p-value from a normal z-score."""
    return float(2 * (1 - (0.5 * (1 + erf(abs(z) / sqrt(2))))))


def linear_trend(decades: list[int], values: list[float]) -> dict | None:
    """OLS trend over decades. Slope is reported per century.

    Returns None when a trend would be meaningless: fewer than three points,
    misaligned inputs, or no spread across decades.
    """
    n = len(decades)
    if n < 3 or n != len(values):
        return None

    decade_mean = sum(decades) / n
    x = [(d - decade_mean) / 100.0 for d in decades]
    y = [float(v) for v in values]

    ss_xx = sum(xi * xi for xi in x)
    if ss_xx == 0:
        return None

    y_bar = sum(y) / n
    slope = sum(xi * (yi - y_bar) for xi, yi in zip(x, y, strict=True)) / ss_xx
    intercept = y_bar
    rss = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y, strict=True))
    df = n - 2
    se = sqrt((rss / df) / ss_xx) if df > 0 else 0.0
    z = slope / se if se > 0 else 0.0

    return {
        "slope_per_century": round(slope, 6),
        "intercept_at_mean_decade": round(intercept, 6),
        "std_error": round(se, 6),
        "z": round(z, 3),
        "p_value_approx": round(normal_two_sided_p(z), 6) if se > 0 else None,
    }
