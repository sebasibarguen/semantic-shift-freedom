# ABOUTME: Bootstrap confidence intervals and trend-oriented robustness analysis.
# ABOUTME: Tests freedom/liberty divergence and legal-vs-personal distance-gap trends.

import json
from pathlib import Path

import numpy as np

from .embeddings import TemporalEmbeddings
from .metrics import cosine_distance, cosine_similarity
from .stats import linear_trend

LEGAL_CLUSTER = ["slavery", "bondage", "emancipation", "rights", "law", "citizen", "slave", "servitude"]
PERSONAL_CLUSTER = ["choice", "autonomy", "independence", "self", "ability", "power", "individual", "personal"]

# The legal cluster is 5/8 abolition vocabulary. If the legal→personal gap trend
# is really "slavery talk faded from COHA" rather than "freedom moved toward
# capacity", removing these words should flatten the trend.
SLAVERY_TERMS = ["slavery", "bondage", "emancipation", "slave", "servitude"]

N_BOOTSTRAP = 1000
RANDOM_SEED = 42






def cluster_distance(emb, word, cluster_words, decade):
    """Average cosine distance from word to cluster members."""
    word_vec = emb.get_vector(word, decade)
    if word_vec is None:
        return None, []

    distances = []
    available_words = []
    for cw in cluster_words:
        cw_vec = emb.get_vector(cw, decade)
        if cw_vec is not None:
            distances.append(cosine_distance(word_vec, cw_vec))
            available_words.append(cw)

    if not distances:
        return None, []
    return sum(distances) / len(distances), available_words


def bootstrap_cluster_distance(emb, word, cluster_words, decade, n_bootstrap=N_BOOTSTRAP, rng=None):
    """Bootstrap CI on average distance from word to cluster by resampling cluster members."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    word_vec = emb.get_vector(word, decade)
    if word_vec is None:
        return None

    # Get individual distances to each cluster word
    individual_distances = []
    for cw in cluster_words:
        cw_vec = emb.get_vector(cw, decade)
        if cw_vec is not None:
            individual_distances.append(cosine_distance(word_vec, cw_vec))

    if len(individual_distances) < 2:
        return None

    distances = np.array(individual_distances)
    point_estimate = float(distances.mean())

    # Bootstrap: resample cluster members with replacement
    boot_means = np.array([
        distances[rng.choice(len(distances), size=len(distances), replace=True)].mean()
        for _ in range(n_bootstrap)
    ])

    return {
        "point_estimate": round(point_estimate, 4),
        "ci_lower": round(float(np.percentile(boot_means, 2.5)), 4),
        "ci_upper": round(float(np.percentile(boot_means, 97.5)), 4),
        "std": round(float(boot_means.std()), 4),
        "n_cluster_words": len(distances),
    }


def bootstrap_pair_similarity(emb, word1, word2, decade, n_bootstrap=N_BOOTSTRAP, rng=None):
    """
    Bootstrap CI on cosine similarity using neighborhood sampling.
    Resample from the nearest neighbors to estimate variability.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    v1 = emb.get_vector(word1, decade)
    v2 = emb.get_vector(word2, decade)
    if v1 is None or v2 is None:
        return None

    point_estimate = cosine_similarity(v1, v2)

    # Get top-k neighbors of each word to sample perturbation space
    nn1 = emb.get_nearest_neighbors(word1, decade, k=20)
    nn2 = emb.get_nearest_neighbors(word2, decade, k=20)

    if len(nn1) < 5 or len(nn2) < 5:
        return {"point_estimate": round(point_estimate, 4), "ci_lower": None, "ci_upper": None}

    # Bootstrap by averaging similarity with small neighborhood perturbations
    # For each iteration, add small noise proportional to the spread in neighbor space
    nn1_vecs = []
    for w, _ in nn1[:10]:
        v = emb.get_vector(w, decade)
        if v is not None:
            nn1_vecs.append(v)

    nn2_vecs = []
    for w, _ in nn2[:10]:
        v = emb.get_vector(w, decade)
        if v is not None:
            nn2_vecs.append(v)

    if not nn1_vecs or not nn2_vecs:
        return {"point_estimate": round(point_estimate, 4), "ci_lower": None, "ci_upper": None}

    nn1_matrix = np.array(nn1_vecs)
    nn2_matrix = np.array(nn2_vecs)

    # Compute spread (std of distances to neighbors)
    spread1 = np.std([cosine_distance(v1, nv) for nv in nn1_matrix])
    spread2 = np.std([cosine_distance(v2, nv) for nv in nn2_matrix])

    boot_sims = []
    for _ in range(n_bootstrap):
        # Perturb each vector by adding noise scaled to neighborhood spread
        noise1 = rng.normal(0, spread1 * 0.1, size=v1.shape)
        noise2 = rng.normal(0, spread2 * 0.1, size=v2.shape)
        boot_sims.append(cosine_similarity(v1 + noise1, v2 + noise2))

    boot_sims = np.array(boot_sims)

    return {
        "point_estimate": round(point_estimate, 4),
        "ci_lower": round(float(np.percentile(boot_sims, 2.5)), 4),
        "ci_upper": round(float(np.percentile(boot_sims, 97.5)), 4),
        "std": round(float(boot_sims.std()), 4),
    }




def cluster_gap_trajectory(emb, word, legal_cluster, personal_cluster, decades):
    """
    Compute personal-minus-legal cluster distance by decade.

    Positive gap means legal/status concepts are closer because the personal
    cluster is farther away. A declining gap means movement toward personal
    concepts relative to legal/status concepts.
    """
    trajectory = {}
    for decade in decades:
        legal_d, legal_words = cluster_distance(emb, word, legal_cluster, decade)
        personal_d, personal_words = cluster_distance(emb, word, personal_cluster, decade)
        if legal_d is None or personal_d is None:
            continue
        trajectory[str(decade)] = {
            "legal_distance": round(legal_d, 6),
            "personal_distance": round(personal_d, 6),
            "personal_minus_legal_gap": round(personal_d - legal_d, 6),
            "n_legal_words": len(legal_words),
            "n_personal_words": len(personal_words),
        }
    return trajectory


def gap_trend_test(trajectory, n_permutations=1000, rng=None):
    """Trend and permutation test for the legal-vs-personal distance gap."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    decades = [int(d) for d in sorted(trajectory.keys(), key=int)]
    gaps = [trajectory[str(d)]["personal_minus_legal_gap"] for d in decades]
    trend = linear_trend(decades, gaps)
    if trend is None:
        return None

    observed_slope = trend["slope_per_century"]
    count_as_strong = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(gaps)
        perm_trend = linear_trend(decades, shuffled.tolist())
        if perm_trend and abs(perm_trend["slope_per_century"]) >= abs(observed_slope):
            count_as_strong += 1

    early = decades[:3]
    late = decades[-3:]
    early_mean = float(np.mean([trajectory[str(d)]["personal_minus_legal_gap"] for d in early]))
    late_mean = float(np.mean([trajectory[str(d)]["personal_minus_legal_gap"] for d in late]))

    return {
        "metric": "personal_minus_legal_gap",
        "interpretation": "positive means legal/status concepts are closer; declining values mean movement toward personal/capacity concepts",
        "early_decades": early,
        "late_decades": late,
        "early_mean_gap": round(early_mean, 6),
        "late_mean_gap": round(late_mean, 6),
        "early_to_late_change": round(late_mean - early_mean, 6),
        "trend": trend,
        "permutation_p_value": round(count_as_strong / n_permutations, 4),
        "n_permutations": n_permutations,
    }


def gap_lexicon_sensitivity(emb, word, legal_cluster, personal_cluster, decades,
                            n_permutations=1000, rng=None):
    """Leave-one-out and slavery-vocabulary sensitivity for the legal→personal gap.

    Reports the gap-trend slope and permutation p when (a) each legal-cluster
    word is dropped one at a time, and (b) all abolition terms are dropped at
    once. If the trend depends entirely on the slavery words, the "no slavery
    terms" variant will lose slope and significance.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    def slope_and_p(legal_words):
        traj = cluster_gap_trajectory(emb, word, legal_words, personal_cluster, decades)
        test = gap_trend_test(traj, n_permutations=n_permutations, rng=rng)
        if not test:
            return None
        return {
            "slope_per_century": test["trend"]["slope_per_century"],
            "z": test["trend"]["z"],
            "permutation_p_value": test["permutation_p_value"],
            "early_to_late_change": test["early_to_late_change"],
            "legal_words": legal_words,
        }

    full = slope_and_p(legal_cluster)
    leave_one_out = {
        dropped: slope_and_p([w for w in legal_cluster if w != dropped])
        for dropped in legal_cluster
    }
    no_slavery = slope_and_p([w for w in legal_cluster if w not in SLAVERY_TERMS])

    return {
        "full_cluster": full,
        "leave_one_out": leave_one_out,
        "no_slavery_terms": no_slavery,
        "slavery_terms_removed": SLAVERY_TERMS,
        "interpretation": (
            "If 'no_slavery_terms' keeps a negative slope with permutation "
            "p < 0.05, the legal→personal shift is not merely abolition "
            "discourse fading."
        ),
    }


def run_analysis():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "sgns"

    print("=" * 70)
    print("ROBUSTNESS ANALYSIS")
    print("Bootstrap CIs and trend tests")
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
    # 1. BOOTSTRAP CIs ON FREEDOM-LIBERTY SIMILARITY
    # =========================================================================
    print("=" * 70)
    print("1. BOOTSTRAP CIs: Freedom-Liberty Similarity")
    print("=" * 70)
    print()

    key_decades = [1800, 1830, 1850, 1870, 1880, 1900, 1920, 1950, 1970, 1990]
    similarity_cis = {}

    print(f"{'Decade':<10} {'Similarity':>12} {'95% CI':>20} {'Width':>8}")
    print("-" * 55)

    for decade in key_decades:
        ci = bootstrap_pair_similarity(emb, "freedom", "liberty", decade, rng=rng)
        if ci and ci["ci_lower"] is not None:
            similarity_cis[str(decade)] = ci
            width = ci["ci_upper"] - ci["ci_lower"]
            print(f"  {decade:<8} {ci['point_estimate']:>10.4f}   [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]  {width:.4f}")

    results["freedom_liberty_similarity_cis"] = similarity_cis
    print()

    # Check if CIs overlap between 1800 and 1990
    if "1800" in similarity_cis and "1990" in similarity_cis:
        ci_1800 = similarity_cis["1800"]
        ci_1990 = similarity_cis["1990"]
        overlap = ci_1800["ci_lower"] < ci_1990["ci_upper"] and ci_1990["ci_lower"] < ci_1800["ci_upper"]
        print(f"  1800 CI: [{ci_1800['ci_lower']}, {ci_1800['ci_upper']}]")
        print(f"  1990 CI: [{ci_1990['ci_lower']}, {ci_1990['ci_upper']}]")
        print(f"  CIs overlap: {'YES — divergence NOT significant' if overlap else 'NO — divergence IS significant'}")
        results["divergence_significant"] = not overlap
    print()

    # =========================================================================
    # 2. BOOTSTRAP CIs ON CLUSTER DISTANCES
    # =========================================================================
    print("=" * 70)
    print("2. BOOTSTRAP CIs: Legal vs Personal Cluster Distance")
    print("=" * 70)
    print()

    cluster_cis = {}
    print(f"{'Decade':<8} {'Legal Dist':>12} {'Legal 95% CI':>22} {'Personal Dist':>14} {'Personal 95% CI':>22} {'Gap':>8}")
    print("-" * 90)

    for decade in key_decades:
        legal_ci = bootstrap_cluster_distance(emb, "freedom", LEGAL_CLUSTER, decade, rng=rng)
        personal_ci = bootstrap_cluster_distance(emb, "freedom", PERSONAL_CLUSTER, decade, rng=rng)

        if legal_ci and personal_ci:
            cluster_cis[str(decade)] = {"legal": legal_ci, "personal": personal_ci}
            gap = personal_ci["point_estimate"] - legal_ci["point_estimate"]
            # Check if CIs overlap (i.e., is the gap significant?)
            ci_overlap = legal_ci["ci_upper"] > personal_ci["ci_lower"] and personal_ci["ci_upper"] > legal_ci["ci_lower"]
            sig = "" if ci_overlap else " *"
            print(f"  {decade:<6} {legal_ci['point_estimate']:>10.4f}  [{legal_ci['ci_lower']:.4f}, {legal_ci['ci_upper']:.4f}]"
                  f"  {personal_ci['point_estimate']:>12.4f}  [{personal_ci['ci_lower']:.4f}, {personal_ci['ci_upper']:.4f}]"
                  f"  {gap:>+.4f}{sig}")

    results["cluster_distance_cis"] = cluster_cis
    print()
    print("  * = cluster distance CIs do NOT overlap (gap is significant)")
    print()

    # =========================================================================
    # 3. LEGAL-vs-PERSONAL GAP TREND
    # =========================================================================
    print("=" * 70)
    print("3. TREND TEST: Legal-vs-Personal Cluster Gap")
    print("   Gap = personal distance - legal distance")
    print("=" * 70)
    print()

    gap_trajectory = cluster_gap_trajectory(
        emb, "freedom", LEGAL_CLUSTER, PERSONAL_CLUSTER, decades
    )
    gap_test = gap_trend_test(gap_trajectory, n_permutations=1000, rng=rng)
    results["legal_personal_gap_trajectory"] = gap_trajectory
    results["legal_personal_gap_trend"] = gap_test

    sensitivity = gap_lexicon_sensitivity(
        emb, "freedom", LEGAL_CLUSTER, PERSONAL_CLUSTER, decades,
        n_permutations=1000, rng=rng,
    )
    results["legal_personal_gap_sensitivity"] = sensitivity

    print(f"{'Decade':<10} {'Legal':>10} {'Personal':>10} {'Gap':>10}")
    print("-" * 46)
    for decade in key_decades:
        row = gap_trajectory.get(str(decade))
        if not row:
            continue
        print(
            f"  {decade:<8} {row['legal_distance']:>10.4f} "
            f"{row['personal_distance']:>10.4f} {row['personal_minus_legal_gap']:>+10.4f}"
        )

    print()
    if gap_test:
        trend = gap_test["trend"]
        print(f"  Early mean gap ({gap_test['early_decades']}): {gap_test['early_mean_gap']:+.4f}")
        print(f"  Late mean gap ({gap_test['late_decades']}):  {gap_test['late_mean_gap']:+.4f}")
        print(f"  Early→late change: {gap_test['early_to_late_change']:+.4f}")
        print(f"  Slope/century: {trend['slope_per_century']:+.4f}")
        print(f"  Permutation p-value: {gap_test['permutation_p_value']}")
    print()

    # =========================================================================
    # SAVE
    # =========================================================================
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "robustness.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"1. Freedom-liberty divergence significant: {results.get('divergence_significant', 'N/A')}")
    if gap_test:
        print(f"2. Legal-personal gap slope/century: {gap_test['trend']['slope_per_century']:+.4f}")
        print(f"3. Legal-personal gap permutation p-value: {gap_test['permutation_p_value']}")

    full = sensitivity["full_cluster"]
    no_slav = sensitivity["no_slavery_terms"]
    if full and no_slav:
        print()
        print("4. Legal-cluster lexicon sensitivity (does the gap depend on slavery vocab?):")
        print(f"   full 8-word cluster : slope={full['slope_per_century']:+.4f}, "
              f"perm-p={full['permutation_p_value']}")
        print(f"   slavery terms removed: slope={no_slav['slope_per_century']:+.4f}, "
              f"perm-p={no_slav['permutation_p_value']}  "
              f"(kept: {', '.join(no_slav['legal_words'])})")
        loo_slopes = {d: v["slope_per_century"] for d, v in sensitivity["leave_one_out"].items() if v}
        if loo_slopes:
            lo = min(loo_slopes.values())
            hi = max(loo_slopes.values())
            print(f"   leave-one-out slope range: [{lo:+.4f}, {hi:+.4f}]")

    return results


if __name__ == "__main__":
    run_analysis()
