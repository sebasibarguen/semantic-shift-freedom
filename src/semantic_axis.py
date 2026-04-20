# ABOUTME: Projects freedom onto a constraint→agency semantic axis using the SemAxis method.
# ABOUTME: Tests when the legal-to-personal shift happened with change-point detection.

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from embeddings import TemporalEmbeddings
from metrics import cosine_similarity, cosine_distance

# Seed words for each pole — expanded via nearest neighbors
CONSTRAINT_SEEDS = ["slavery", "bondage", "servitude", "oppression"]
AGENCY_SEEDS = ["autonomy", "choice", "ability", "capacity"]

CONTROL_WORDS = [
    "liberty", "justice", "truth", "honor", "power",
    "virtue", "equality", "democracy", "authority", "dignity",
]

REFERENCE_DECADE = 1900  # midpoint of the data range
EXPANSION_K = 10  # expand each seed to this many neighbors
N_PERMUTATIONS = 1000
RANDOM_SEED = 42


def expand_pole(emb, seed_words, decade, k=EXPANSION_K):
    """Expand seed words by adding their nearest neighbors."""
    expanded = set(seed_words)
    for seed in seed_words:
        nn = emb.get_nearest_neighbors(seed, decade, k)
        for word, _ in nn:
            expanded.add(word)
    # Remove words that are in both poles' seeds (safety check done by caller)
    return sorted(expanded)


def build_axis(emb, constraint_words, agency_words, decade):
    """Build semantic axis as normalized difference of pole centroids."""
    constraint_vecs = []
    for w in constraint_words:
        v = emb.get_vector(w, decade)
        if v is not None:
            constraint_vecs.append(v)

    agency_vecs = []
    for w in agency_words:
        v = emb.get_vector(w, decade)
        if v is not None:
            agency_vecs.append(v)

    if not constraint_vecs or not agency_vecs:
        return None

    centroid_c = np.mean(constraint_vecs, axis=0)
    centroid_a = np.mean(agency_vecs, axis=0)

    axis = centroid_a - centroid_c
    norm = np.linalg.norm(axis)
    if norm == 0:
        return None
    return axis / norm


def project_onto_axis(emb, word, decade, axis):
    """Project a word's vector onto the semantic axis. Higher = more agency-like."""
    v = emb.get_vector(word, decade)
    if v is None:
        return None
    return float(np.dot(v, axis))


def linear_trend(decades, values):
    """Compute linear regression slope and R-squared."""
    x = np.array(decades, dtype=float)
    y = np.array(values, dtype=float)
    n = len(x)
    if n < 3:
        return None

    x_mean = x.mean()
    y_mean = y.mean()
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_yy = np.sum((y - y_mean) ** 2)

    if ss_xx == 0:
        return None

    slope = ss_xy / ss_xx
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 0 else 0

    return {
        "slope": float(slope),
        "r_squared": float(r_squared),
        "slope_per_century": float(slope * 100),
    }


def find_changepoint_bic(decades, values):
    """
    Find a single changepoint using BIC-penalized piecewise linear regression.
    Tests every possible split point; returns the one that minimizes BIC.
    """
    x = np.array(decades, dtype=float)
    y = np.array(values, dtype=float)
    n = len(x)

    if n < 6:  # Need at least 3 points per segment
        return None

    # Null model: single linear fit
    null_trend = linear_trend(decades, values)
    if null_trend is None:
        return None

    y_pred_null = null_trend["slope"] * (x - x.mean()) + y.mean()
    rss_null = np.sum((y - y_pred_null) ** 2)
    k_null = 2  # slope + intercept
    bic_null = n * np.log(rss_null / n + 1e-10) + k_null * np.log(n)

    best_bic = bic_null
    best_split = None

    for split_idx in range(3, n - 3):
        # Fit two separate linear models
        x1, y1 = x[:split_idx], y[:split_idx]
        x2, y2 = x[split_idx:], y[split_idx:]

        trend1 = linear_trend(x1.tolist(), y1.tolist())
        trend2 = linear_trend(x2.tolist(), y2.tolist())

        if trend1 is None or trend2 is None:
            continue

        y_pred1 = trend1["slope"] * (x1 - x1.mean()) + y1.mean()
        y_pred2 = trend2["slope"] * (x2 - x2.mean()) + y2.mean()
        rss = np.sum((y1 - y_pred1) ** 2) + np.sum((y2 - y_pred2) ** 2)

        k_split = 4  # two slopes + two intercepts
        bic = n * np.log(rss / n + 1e-10) + k_split * np.log(n)

        if bic < best_bic:
            best_bic = bic
            best_split = {
                "decade": int(x[split_idx]),
                "bic_improvement": float(bic_null - bic),
                "before_slope_per_century": trend1["slope_per_century"],
                "after_slope_per_century": trend2["slope_per_century"],
            }

    return best_split


def permutation_test_trend(emb, word, axis, decades, observed_slope, rng,
                            n_permutations=N_PERMUTATIONS):
    """
    Permutation test: shuffle decade labels for the word's projections
    and see how often the trend is as strong as observed.
    """
    # Get all projections
    projections = []
    valid_decades = []
    for decade in decades:
        p = project_onto_axis(emb, word, decade, axis)
        if p is not None:
            projections.append(p)
            valid_decades.append(decade)

    if len(projections) < 5:
        return None

    count_as_strong = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(projections)
        trend = linear_trend(valid_decades, shuffled.tolist())
        if trend and abs(trend["slope"]) >= abs(observed_slope):
            count_as_strong += 1

    return {
        "p_value": round(count_as_strong / n_permutations, 4),
        "n_permutations": n_permutations,
    }


def seed_sensitivity_test(emb, decades, rng, n_trials=100):
    """
    Test sensitivity to seed word choice. For each trial, randomly drop
    one seed from each pole and recompute the trend.
    """
    slopes = []

    for _ in range(n_trials):
        # Drop one random seed from each pole
        c_subset = list(rng.choice(CONSTRAINT_SEEDS, size=len(CONSTRAINT_SEEDS) - 1, replace=False))
        a_subset = list(rng.choice(AGENCY_SEEDS, size=len(AGENCY_SEEDS) - 1, replace=False))

        c_expanded = expand_pole(emb, c_subset, REFERENCE_DECADE, EXPANSION_K)
        a_expanded = expand_pole(emb, a_subset, REFERENCE_DECADE, EXPANSION_K)

        # Remove overlap
        overlap = set(c_expanded) & set(a_expanded)
        c_expanded = [w for w in c_expanded if w not in overlap]
        a_expanded = [w for w in a_expanded if w not in overlap]

        axis = build_axis(emb, c_expanded, a_expanded, REFERENCE_DECADE)
        if axis is None:
            continue

        projections = []
        valid_decades = []
        for decade in decades:
            p = project_onto_axis(emb, "freedom", decade, axis)
            if p is not None:
                projections.append(p)
                valid_decades.append(decade)

        if len(valid_decades) >= 5:
            trend = linear_trend(valid_decades, projections)
            if trend:
                slopes.append(trend["slope_per_century"])

    if not slopes:
        return None

    slopes = np.array(slopes)
    return {
        "n_trials": len(slopes),
        "mean_slope": round(float(slopes.mean()), 6),
        "std_slope": round(float(slopes.std()), 6),
        "pct_positive": round(float(np.mean(slopes > 0) * 100), 1),
        "pct_negative": round(float(np.mean(slopes < 0) * 100), 1),
        "ci_lower": round(float(np.percentile(slopes, 2.5)), 6),
        "ci_upper": round(float(np.percentile(slopes, 97.5)), 6),
    }


def run_analysis():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "sgns"

    print("=" * 70)
    print("SEMANTIC AXIS PROJECTION")
    print("Constraint → Agency axis (SemAxis method)")
    print("=" * 70)
    print()

    print("Loading HistWords embeddings...")
    emb = TemporalEmbeddings(str(data_dir))
    emb.load_decades(start=1800, end=1990, step=10)
    decades = emb.decades
    print()

    rng = np.random.default_rng(RANDOM_SEED)
    results = {}

    # =========================================================================
    # 1. BUILD THE AXIS
    # =========================================================================
    print("=" * 70)
    print(f"1. BUILDING CONSTRAINT→AGENCY AXIS (reference decade: {REFERENCE_DECADE})")
    print("=" * 70)
    print()

    print(f"  Constraint seeds: {CONSTRAINT_SEEDS}")
    print(f"  Agency seeds: {AGENCY_SEEDS}")
    print()

    constraint_expanded = expand_pole(emb, CONSTRAINT_SEEDS, REFERENCE_DECADE, EXPANSION_K)
    agency_expanded = expand_pole(emb, AGENCY_SEEDS, REFERENCE_DECADE, EXPANSION_K)

    # Remove overlap
    overlap = set(constraint_expanded) & set(agency_expanded)
    constraint_expanded = [w for w in constraint_expanded if w not in overlap]
    agency_expanded = [w for w in agency_expanded if w not in overlap]

    print(f"  Constraint pole ({len(constraint_expanded)} words): {constraint_expanded[:15]}...")
    print(f"  Agency pole ({len(agency_expanded)} words): {agency_expanded[:15]}...")
    if overlap:
        print(f"  Removed {len(overlap)} overlapping words: {sorted(overlap)[:10]}")
    print()

    results["axis_construction"] = {
        "reference_decade": REFERENCE_DECADE,
        "constraint_seeds": CONSTRAINT_SEEDS,
        "agency_seeds": AGENCY_SEEDS,
        "constraint_expanded": constraint_expanded,
        "agency_expanded": agency_expanded,
        "overlap_removed": sorted(overlap),
    }

    # Build the axis (FIXED across all decades)
    axis = build_axis(emb, constraint_expanded, agency_expanded, REFERENCE_DECADE)
    if axis is None:
        print("ERROR: Could not build axis")
        return

    # =========================================================================
    # 2. PROJECT FREEDOM AND CONTROL WORDS
    # =========================================================================
    print("=" * 70)
    print("2. PROJECTIONS ONTO CONSTRAINT→AGENCY AXIS")
    print("   (higher = more agency-like, lower = more constraint-like)")
    print("=" * 70)
    print()

    all_words = ["freedom"] + [w for w in CONTROL_WORDS if emb.word_exists(w, 1800)]
    projections = {word: {} for word in all_words}

    key_decades = [1800, 1830, 1850, 1870, 1880, 1900, 1920, 1950, 1970, 1990]

    print(f"{'Decade':<10}", end="")
    for word in all_words[:5]:
        print(f"{word:>12}", end="")
    print()
    print("-" * (10 + 12 * min(5, len(all_words))))

    for decade in key_decades:
        print(f"  {decade:<8}", end="")
        for word in all_words[:5]:
            p = project_onto_axis(emb, word, decade, axis)
            if p is not None:
                projections[word][str(decade)] = round(p, 4)
                print(f"{p:>12.4f}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()

    # Store all projections (not just the ones we printed)
    for word in all_words[5:]:
        for decade in decades:
            p = project_onto_axis(emb, word, decade, axis)
            if p is not None:
                projections[word][str(decade)] = round(p, 4)

    results["projections"] = projections
    print()

    # =========================================================================
    # 3. TREND ANALYSIS
    # =========================================================================
    print("=" * 70)
    print("3. LINEAR TREND ANALYSIS")
    print("=" * 70)
    print()

    trends = {}
    print(f"{'Word':<18} {'Slope/century':>14} {'R²':>8} {'Direction':>12}")
    print("-" * 55)

    for word in all_words:
        word_decades = [int(d) for d in sorted(projections[word].keys())]
        word_values = [projections[word][str(d)] for d in word_decades]

        trend = linear_trend(word_decades, word_values)
        if trend:
            trends[word] = trend
            direction = "→ agency" if trend["slope"] > 0 else "→ constraint"
            marker = " <--" if word == "freedom" else ""
            print(f"  {word:<16} {trend['slope_per_century']:>+12.4f} {trend['r_squared']:>8.3f} {direction:>12}{marker}")

    results["trends"] = trends
    print()

    # Compare freedom's trend to control distribution
    control_slopes = [t["slope_per_century"] for w, t in trends.items() if w != "freedom"]
    freedom_slope = trends.get("freedom", {}).get("slope_per_century")

    if control_slopes and freedom_slope is not None:
        mean_control = np.mean(control_slopes)
        std_control = np.std(control_slopes)
        z_score = (freedom_slope - mean_control) / std_control if std_control > 0 else 0

        print(f"  Freedom slope: {freedom_slope:+.4f} per century")
        print(f"  Control mean:  {mean_control:+.4f} (std: {std_control:.4f})")
        print(f"  Z-score: {z_score:.2f}")

        results["freedom_vs_controls"] = {
            "freedom_slope": round(freedom_slope, 6),
            "control_mean": round(float(mean_control), 6),
            "control_std": round(float(std_control), 6),
            "z_score": round(float(z_score), 2),
        }

        if abs(z_score) > 2:
            print(f"  → Freedom's trend is an OUTLIER (|z| > 2)")
        elif abs(z_score) > 1:
            print(f"  → Freedom's trend is NOTABLE but not extreme")
        else:
            print(f"  → Freedom's trend is WITHIN NORMAL RANGE")

    print()

    # =========================================================================
    # 4. CHANGE-POINT DETECTION
    # =========================================================================
    print("=" * 70)
    print("4. CHANGE-POINT DETECTION (BIC-penalized)")
    print("=" * 70)
    print()

    freedom_decades = [int(d) for d in sorted(projections["freedom"].keys())]
    freedom_values = [projections["freedom"][str(d)] for d in freedom_decades]

    changepoint = find_changepoint_bic(freedom_decades, freedom_values)
    results["changepoint"] = changepoint

    if changepoint:
        print(f"  Change-point detected at: {changepoint['decade']}")
        print(f"  BIC improvement over single-line model: {changepoint['bic_improvement']:.2f}")
        print(f"  Before: {changepoint['before_slope_per_century']:+.4f} per century")
        print(f"  After:  {changepoint['after_slope_per_century']:+.4f} per century")
    else:
        print("  No change-point detected (single linear model preferred by BIC)")

    print()

    # =========================================================================
    # 5. PERMUTATION TEST
    # =========================================================================
    print("=" * 70)
    print("5. PERMUTATION TEST: Is freedom's trend significant?")
    print("=" * 70)
    print()

    if freedom_slope is not None:
        perm_result = permutation_test_trend(
            emb, "freedom", axis, decades, trends["freedom"]["slope"], rng
        )
        results["permutation_test"] = perm_result

        if perm_result:
            print(f"  Observed slope: {freedom_slope:+.4f} per century")
            print(f"  p-value: {perm_result['p_value']}")
            if perm_result["p_value"] < 0.05:
                print(f"  → SIGNIFICANT at p < 0.05")
            else:
                print(f"  → NOT significant")
    print()

    # =========================================================================
    # 6. SEED SENSITIVITY
    # =========================================================================
    print("=" * 70)
    print("6. SEED WORD SENSITIVITY (dropping 1 seed per pole, 100 trials)")
    print("=" * 70)
    print()

    sensitivity = seed_sensitivity_test(emb, decades, rng, n_trials=100)
    results["seed_sensitivity"] = sensitivity

    if sensitivity:
        print(f"  Mean slope: {sensitivity['mean_slope']:+.6f} per century")
        print(f"  95% CI: [{sensitivity['ci_lower']:+.6f}, {sensitivity['ci_upper']:+.6f}]")
        print(f"  Positive trend in {sensitivity['pct_positive']:.0f}% of trials")
        print(f"  Negative trend in {sensitivity['pct_negative']:.0f}% of trials")
        print()
        if sensitivity["pct_positive"] > 90 or sensitivity["pct_negative"] > 90:
            dominant = "agency" if sensitivity["pct_positive"] > 90 else "constraint"
            print(f"  → Trend direction is ROBUST: consistently toward {dominant}")
        elif sensitivity["pct_positive"] > 70 or sensitivity["pct_negative"] > 70:
            print(f"  → Trend direction is MOSTLY CONSISTENT but not rock-solid")
        else:
            print(f"  → Trend direction is SENSITIVE to seed choice")

    # =========================================================================
    # SAVE
    # =========================================================================
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "semantic_axis.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {output_file}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    if freedom_slope is not None:
        direction = "toward agency" if freedom_slope > 0 else "toward constraint"
        print(f"  1. Freedom's trend: {direction} ({freedom_slope:+.4f}/century)")
    if results.get("freedom_vs_controls", {}).get("z_score"):
        print(f"  2. vs controls: z = {results['freedom_vs_controls']['z_score']}")
    if changepoint:
        print(f"  3. Change-point: {changepoint['decade']}")
    else:
        print(f"  3. Change-point: none detected (gradual trend)")
    if perm_result:
        print(f"  4. Permutation p-value: {perm_result['p_value']}")
    if sensitivity:
        dominant_pct = max(sensitivity["pct_positive"], sensitivity["pct_negative"])
        print(f"  5. Seed sensitivity: trend consistent in {dominant_pct:.0f}% of trials")

    return results


if __name__ == "__main__":
    run_analysis()
