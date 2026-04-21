# ABOUTME: Extends embedding analysis to 2000s using COHA HistWords data.
# ABOUTME: Compares COHA results with Google Books HistWords for cross-validation.

import json
import numpy as np
from pathlib import Path


def run_coha_analysis(coha_dir: str, gbooks_dir: str, output_path: str):
    """
    Run the full COHA analysis: freedom-liberty similarity, SemAxis projection,
    control words, and cross-validation against Google Books.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from embeddings import TemporalEmbeddings
    from metrics import cosine_similarity
    from semantic_axis import (
        CONSTRAINT_SEEDS, AGENCY_SEEDS, CONTROL_WORDS,
        expand_pole, build_axis, project_onto_axis, linear_trend,
        find_changepoint_bic, REFERENCE_DECADE, EXPANSION_K,
    )

    results = {"corpus_comparison": {}, "coha": {}, "gbooks": {}}

    # =========================================================================
    # 1. LOAD BOTH CORPORA
    # =========================================================================
    print("=" * 70)
    print("MODERN EMBEDDINGS ANALYSIS")
    print("COHA HistWords (1830s-2000s) + Google Books cross-validation")
    print("=" * 70)
    print()

    print("Loading COHA embeddings...")
    coha = TemporalEmbeddings(coha_dir)
    coha.load_decades(start=1830, end=2000, step=10)
    coha_decades = coha.decades
    print(f"COHA decades loaded: {coha_decades}")
    print()

    print("Loading Google Books embeddings...")
    gbooks = TemporalEmbeddings(gbooks_dir)
    gbooks.load_decades(start=1800, end=1990, step=10)
    gbooks_decades = gbooks.decades
    print(f"Google Books decades loaded: {gbooks_decades}")
    print()

    results["coha"]["decades"] = coha_decades
    results["gbooks"]["decades"] = gbooks_decades

    # =========================================================================
    # 2. FREEDOM-LIBERTY SIMILARITY TRAJECTORY
    # =========================================================================
    print("=" * 70)
    print("1. FREEDOM-LIBERTY SIMILARITY")
    print("=" * 70)
    print()

    for label, emb, decades in [("coha", coha, coha_decades), ("gbooks", gbooks, gbooks_decades)]:
        traj = {}
        for decade in decades:
            v_free = emb.get_vector("freedom", decade)
            v_lib = emb.get_vector("liberty", decade)
            if v_free is not None and v_lib is not None:
                traj[str(decade)] = round(float(cosine_similarity(v_free, v_lib)), 4)
        results[label]["freedom_liberty_similarity"] = traj
        print(f"  {label.upper()} freedom-liberty similarity:")
        for d in sorted(traj.keys()):
            print(f"    {d}: {traj[d]:.4f}")
        print()

    # Cross-validate: compare overlap decades
    overlap_decades = sorted(set(coha_decades) & set(gbooks_decades))
    fl_comparison = {}
    for decade in overlap_decades:
        coha_sim = results["coha"]["freedom_liberty_similarity"].get(str(decade))
        gbooks_sim = results["gbooks"]["freedom_liberty_similarity"].get(str(decade))
        if coha_sim is not None and gbooks_sim is not None:
            fl_comparison[str(decade)] = {
                "coha": coha_sim,
                "gbooks": gbooks_sim,
                "diff": round(coha_sim - gbooks_sim, 4),
            }
    results["corpus_comparison"]["freedom_liberty_similarity"] = fl_comparison

    print("  Cross-validation (overlap decades):")
    print(f"  {'Decade':<10} {'COHA':>8} {'GBooks':>8} {'Diff':>8}")
    print(f"  {'-'*36}")
    for d in sorted(fl_comparison.keys()):
        c = fl_comparison[d]
        print(f"  {d:<10} {c['coha']:>8.4f} {c['gbooks']:>8.4f} {c['diff']:>+8.4f}")
    print()

    # =========================================================================
    # 3. SEMAXIS PROJECTION — COHA
    # =========================================================================
    print("=" * 70)
    print("2. SEMAXIS PROJECTION (constraint → agency)")
    print("=" * 70)
    print()

    for label, emb, decades in [("coha", coha, coha_decades), ("gbooks", gbooks, gbooks_decades)]:
        # Find best reference decade (closest to 1900 that exists)
        ref_decade = min(decades, key=lambda d: abs(d - REFERENCE_DECADE))

        constraint_expanded = expand_pole(emb, CONSTRAINT_SEEDS, ref_decade, EXPANSION_K)
        agency_expanded = expand_pole(emb, AGENCY_SEEDS, ref_decade, EXPANSION_K)

        overlap = set(constraint_expanded) & set(agency_expanded)
        constraint_expanded = [w for w in constraint_expanded if w not in overlap]
        agency_expanded = [w for w in agency_expanded if w not in overlap]

        axis = build_axis(emb, constraint_expanded, agency_expanded, ref_decade)
        if axis is None:
            print(f"  WARNING: Could not build axis for {label}")
            continue

        # Project freedom + control words
        all_words = ["freedom"] + [w for w in CONTROL_WORDS if emb.word_exists(w, ref_decade)]
        projections = {}
        for word in all_words:
            word_proj = {}
            for decade in decades:
                p = project_onto_axis(emb, word, decade, axis)
                if p is not None:
                    word_proj[str(decade)] = round(p, 4)
            projections[word] = word_proj

        results[label]["semaxis_projections"] = projections
        results[label]["axis_construction"] = {
            "reference_decade": ref_decade,
            "constraint_words": len(constraint_expanded),
            "agency_words": len(agency_expanded),
            "overlap_removed": len(overlap),
        }

        # Trend analysis
        trends = {}
        for word in all_words:
            word_decades = [int(d) for d in sorted(projections[word].keys())]
            word_values = [projections[word][str(d)] for d in word_decades]
            trend = linear_trend(word_decades, word_values)
            if trend:
                trends[word] = trend
        results[label]["semaxis_trends"] = trends

        print(f"  {label.upper()} SemAxis projections for 'freedom':")
        if "freedom" in projections:
            for d in sorted(projections["freedom"].keys()):
                print(f"    {d}: {projections['freedom'][d]:>+.4f}")
        print()

        if "freedom" in trends:
            t = trends["freedom"]
            direction = "→ agency" if t["slope"] > 0 else "→ constraint"
            print(f"  Freedom trend: {t['slope_per_century']:+.4f}/century (R²={t['r_squared']:.3f}) {direction}")
        print()

    # =========================================================================
    # 4. THE 2000s DATA POINT (COHA only)
    # =========================================================================
    print("=" * 70)
    print("3. THE 2000s — WHAT HAPPENED?")
    print("=" * 70)
    print()

    coha_proj = results["coha"].get("semaxis_projections", {}).get("freedom", {})
    if "2000" in coha_proj:
        val_2000 = coha_proj["2000"]
        val_1990 = coha_proj.get("1990")
        val_1980 = coha_proj.get("1980")

        print(f"  Freedom's SemAxis projection in 2000s: {val_2000:+.4f}")
        if val_1990 is not None:
            delta = val_2000 - val_1990
            print(f"  Change from 1990s: {delta:+.4f}")
        if val_1980 is not None and val_1990 is not None:
            delta_prev = val_1990 - val_1980
            delta_curr = val_2000 - val_1990
            print(f"  1980→1990 change: {delta_prev:+.4f}")
            print(f"  1990→2000 change: {delta_curr:+.4f}")

        # Determine trend: continuing, reversing, or plateauing
        sorted_decades = sorted(coha_proj.keys())
        if len(sorted_decades) >= 3:
            last_3 = sorted_decades[-3:]
            vals = [coha_proj[d] for d in last_3]
            d1 = vals[1] - vals[0]
            d2 = vals[2] - vals[1]

            if abs(d2) < 0.005:
                verdict = "PLATEAU — minimal change in the 2000s"
            elif d1 > 0 and d2 > 0:
                verdict = "CONTINUING — agency-ward trend persists"
            elif d1 > 0 and d2 < 0:
                verdict = "REVERSING — freedom moved back toward constraint"
            elif d1 < 0 and d2 < 0:
                verdict = "CONTINUING — constraint-ward trend persists"
            elif d1 < 0 and d2 > 0:
                verdict = "REVERSING — freedom moved back toward agency"
            else:
                verdict = "UNCLEAR"

            print(f"\n  Verdict: {verdict}")
            results["coha"]["2000s_verdict"] = verdict
    else:
        print("  2000s data point not available in COHA embeddings")

    print()

    # =========================================================================
    # 5. CHANGE-POINT DETECTION ON COHA (longer series)
    # =========================================================================
    print("=" * 70)
    print("4. CHANGE-POINT DETECTION (COHA, 1830-2000s)")
    print("=" * 70)
    print()

    if "freedom" in results["coha"].get("semaxis_projections", {}):
        freedom_proj = results["coha"]["semaxis_projections"]["freedom"]
        cp_decades = [int(d) for d in sorted(freedom_proj.keys())]
        cp_values = [freedom_proj[str(d)] for d in cp_decades]

        changepoint = find_changepoint_bic(cp_decades, cp_values)
        results["coha"]["changepoint"] = changepoint

        if changepoint:
            print(f"  Change-point: {changepoint['decade']}")
            print(f"  BIC improvement: {changepoint['bic_improvement']:.2f}")
            print(f"  Before: {changepoint['before_slope_per_century']:+.4f}/century")
            print(f"  After:  {changepoint['after_slope_per_century']:+.4f}/century")
        else:
            print("  No change-point detected")

    print()

    # =========================================================================
    # 6. CROSS-VALIDATION SUMMARY
    # =========================================================================
    print("=" * 70)
    print("5. CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    print()

    coha_trend = results["coha"].get("semaxis_trends", {}).get("freedom", {})
    gbooks_trend = results["gbooks"].get("semaxis_trends", {}).get("freedom", {})

    if coha_trend and gbooks_trend:
        coha_slope = coha_trend.get("slope_per_century", 0)
        gbooks_slope = gbooks_trend.get("slope_per_century", 0)
        same_direction = (coha_slope > 0) == (gbooks_slope > 0)

        print(f"  COHA trend:   {coha_slope:+.4f}/century")
        print(f"  GBooks trend: {gbooks_slope:+.4f}/century")
        print(f"  Same direction: {'YES' if same_direction else 'NO'}")
        print(f"  Magnitude ratio: {abs(coha_slope / gbooks_slope):.2f}x" if gbooks_slope != 0 else "")

        results["corpus_comparison"]["trend_agreement"] = {
            "coha_slope": round(coha_slope, 6),
            "gbooks_slope": round(gbooks_slope, 6),
            "same_direction": same_direction,
        }

        if same_direction:
            print("\n  The two independent corpora AGREE on the direction of freedom's semantic shift.")
        else:
            print("\n  The two corpora DISAGREE — the finding may be corpus-specific.")

    print()

    # =========================================================================
    # 7. CONTROL WORD COMPARISON
    # =========================================================================
    print("=" * 70)
    print("6. CONTROL WORDS — IS FREEDOM SPECIAL?")
    print("=" * 70)
    print()

    for label in ["coha", "gbooks"]:
        trends = results[label].get("semaxis_trends", {})
        freedom_slope = trends.get("freedom", {}).get("slope_per_century")
        control_slopes = [
            t["slope_per_century"] for w, t in trends.items() if w != "freedom"
        ]

        if freedom_slope is not None and control_slopes:
            mean_ctrl = np.mean(control_slopes)
            std_ctrl = np.std(control_slopes)
            z = (freedom_slope - mean_ctrl) / std_ctrl if std_ctrl > 0 else 0

            results[label]["freedom_vs_controls"] = {
                "freedom_slope": round(freedom_slope, 6),
                "control_mean": round(float(mean_ctrl), 6),
                "control_std": round(float(std_ctrl), 6),
                "z_score": round(float(z), 2),
            }

            print(f"  {label.upper()}:")
            print(f"    Freedom: {freedom_slope:+.4f}/century")
            print(f"    Controls: {mean_ctrl:+.4f} ± {std_ctrl:.4f}")
            print(f"    Z-score: {z:.2f}")
            outlier = "OUTLIER" if abs(z) > 2 else "notable" if abs(z) > 1 else "within range"
            print(f"    → {outlier}")
            print()

    # =========================================================================
    # SAVE
    # =========================================================================
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    run_coha_analysis(
        coha_dir=str(project_root / "data" / "coha_sgns"),
        gbooks_dir=str(project_root / "data" / "sgns"),
        output_path=str(project_root / "outputs" / "modern_embeddings.json"),
    )
