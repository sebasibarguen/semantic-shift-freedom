# ABOUTME: Bootstrap confidence intervals and cluster sensitivity analysis.
# ABOUTME: Tests whether the 1880 crossover and freedom/liberty divergence are statistically robust.

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from embeddings import TemporalEmbeddings

LEGAL_CLUSTER = ["slavery", "bondage", "emancipation", "rights", "law", "citizen", "slave", "servitude"]
PERSONAL_CLUSTER = ["choice", "autonomy", "independence", "self", "ability", "power", "individual", "personal"]

N_BOOTSTRAP = 1000
RANDOM_SEED = 42


def cosine_similarity(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def cosine_distance(v1, v2):
    return 1.0 - cosine_similarity(v1, v2)


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


def cluster_sensitivity(emb, word, full_legal, full_personal, decades, n_trials=100, rng=None):
    """
    Test sensitivity of crossover to cluster composition.
    For each trial, randomly drop 2 words from each cluster and find the crossover.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    crossover_counts = Counter()
    no_crossover_count = 0

    for _ in range(n_trials):
        # Drop 2 random words from each cluster
        legal_subset = list(rng.choice(full_legal, size=max(3, len(full_legal) - 2), replace=False))
        personal_subset = list(rng.choice(full_personal, size=max(3, len(full_personal) - 2), replace=False))

        prev_closer = None
        crossover = None
        for decade in decades:
            legal_d, _ = cluster_distance(emb, word, legal_subset, decade)
            personal_d, _ = cluster_distance(emb, word, personal_subset, decade)

            if legal_d is None or personal_d is None:
                continue

            closer = "legal" if legal_d < personal_d else "personal"
            if prev_closer and prev_closer != closer:
                crossover = decade
                break
            prev_closer = closer

        if crossover:
            crossover_counts[crossover] += 1
        else:
            no_crossover_count += 1

    return {
        "crossover_distribution": {str(k): v for k, v in sorted(crossover_counts.items())},
        "no_crossover_count": no_crossover_count,
        "n_trials": n_trials,
        "modal_crossover": max(crossover_counts, key=crossover_counts.get) if crossover_counts else None,
    }


def permutation_test_crossover(emb, word, legal_cluster, personal_cluster, decades,
                                 observed_crossover, n_permutations=1000, rng=None):
    """
    Permutation test: pool legal+personal cluster words, randomly split into two groups,
    see how often a crossover at or before the observed decade occurs by chance.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    pooled = legal_cluster + personal_cluster
    n_legal = len(legal_cluster)
    count_at_or_before = 0

    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        fake_legal = list(shuffled[:n_legal])
        fake_personal = list(shuffled[n_legal:])

        prev_closer = None
        crossover = None
        for decade in decades:
            legal_d, _ = cluster_distance(emb, word, fake_legal, decade)
            personal_d, _ = cluster_distance(emb, word, fake_personal, decade)

            if legal_d is None or personal_d is None:
                continue

            closer = "legal" if legal_d < personal_d else "personal"
            if prev_closer and prev_closer != closer:
                crossover = decade
                break
            prev_closer = closer

        if crossover is not None and crossover <= observed_crossover:
            count_at_or_before += 1

    p_value = count_at_or_before / n_permutations
    return {
        "observed_crossover": observed_crossover,
        "p_value": round(p_value, 4),
        "n_permutations": n_permutations,
        "crossovers_at_or_before": count_at_or_before,
    }


def run_analysis():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "sgns"

    print("=" * 70)
    print("ROBUSTNESS ANALYSIS")
    print("Bootstrap CIs, Cluster Sensitivity, Permutation Tests")
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
    # 3. CLUSTER SENSITIVITY ANALYSIS
    # =========================================================================
    print("=" * 70)
    print("3. CLUSTER SENSITIVITY: Crossover Date Distribution")
    print(f"   (Randomly dropping 2 words from each cluster, {100} trials)")
    print("=" * 70)
    print()

    sensitivity = cluster_sensitivity(
        emb, "freedom", LEGAL_CLUSTER, PERSONAL_CLUSTER, decades,
        n_trials=200, rng=rng
    )

    results["cluster_sensitivity"] = sensitivity

    print("  Crossover decade distribution:")
    for decade, count in sorted(sensitivity["crossover_distribution"].items()):
        bar = "#" * (count // 2)
        print(f"    {decade}: {count:>4} ({count/sensitivity['n_trials']*100:.0f}%)  {bar}")
    print(f"    No crossover: {sensitivity['no_crossover_count']}")
    print(f"  Modal crossover: {sensitivity['modal_crossover']}")
    print()

    # =========================================================================
    # 4. PERMUTATION TEST
    # =========================================================================
    print("=" * 70)
    print("4. PERMUTATION TEST: Is the crossover non-random?")
    print("   H0: Legal and personal clusters are interchangeable")
    print("=" * 70)
    print()

    perm_result = permutation_test_crossover(
        emb, "freedom", LEGAL_CLUSTER, PERSONAL_CLUSTER, decades,
        observed_crossover=1880, n_permutations=1000, rng=rng
    )

    results["permutation_test"] = perm_result

    print(f"  Observed crossover: {perm_result['observed_crossover']}")
    print(f"  Crossovers at or before 1880 under H0: {perm_result['crossovers_at_or_before']}/{perm_result['n_permutations']}")
    print(f"  p-value: {perm_result['p_value']}")
    if perm_result['p_value'] < 0.05:
        print("  → SIGNIFICANT at p < 0.05: the crossover is unlikely under random cluster assignment")
    else:
        print("  → NOT significant: random cluster splits produce similar crossovers")
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
    print(f"2. Modal crossover decade: {sensitivity['modal_crossover']} (from {sensitivity['n_trials']} random cluster subsets)")
    print(f"3. Permutation test p-value: {perm_result['p_value']}")

    return results


if __name__ == "__main__":
    run_analysis()
