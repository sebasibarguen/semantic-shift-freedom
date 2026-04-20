# ABOUTME: Tracks how freedom's semantic neighborhood restructures over time.
# ABOUTME: Tests the "stable word, changing context" finding with zero researcher degrees of freedom.

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from embeddings import TemporalEmbeddings
from metrics import cosine_similarity, cosine_distance
from domain_tagger import DomainTagger

CONTROL_WORDS = [
    "liberty", "justice", "truth", "honor", "power",
    "virtue", "equality", "democracy", "authority", "dignity",
    "right", "peace", "law", "independence", "knowledge",
]

K_NEIGHBORS = 50


def neighborhood_centroid(emb, word, decade, k=K_NEIGHBORS):
    """Compute centroid of a word's k nearest neighbors."""
    nn = emb.get_nearest_neighbors(word, decade, k)
    vecs = []
    for w, _ in nn:
        v = emb.get_vector(w, decade)
        if v is not None:
            vecs.append(v)
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def centroid_displacement(emb, word, decade1, decade2, k=K_NEIGHBORS):
    """Cosine distance between neighborhood centroids across two decades."""
    c1 = neighborhood_centroid(emb, word, decade1, k)
    c2 = neighborhood_centroid(emb, word, decade2, k)
    if c1 is None or c2 is None:
        return None
    return cosine_distance(c1, c2)


def neighbor_turnover(emb, word, decade1, decade2, k=K_NEIGHBORS):
    """Track which words enter and leave a word's neighborhood."""
    nn1 = set(w for w, _ in emb.get_nearest_neighbors(word, decade1, k))
    nn2 = set(w for w, _ in emb.get_nearest_neighbors(word, decade2, k))
    return {
        "entered": sorted(nn2 - nn1),
        "exited": sorted(nn1 - nn2),
        "stable": sorted(nn1 & nn2),
        "jaccard": len(nn1 & nn2) / len(nn1 | nn2) if nn1 | nn2 else 0,
    }


def second_order_similarity(emb, word, decade, k=K_NEIGHBORS):
    """Average pairwise similarity among a word's neighbors (neighborhood coherence)."""
    nn = emb.get_nearest_neighbors(word, decade, k)
    vecs = []
    for w, _ in nn[:30]:  # Use top 30 to keep computation tractable
        v = emb.get_vector(w, decade)
        if v is not None:
            vecs.append(v)

    if len(vecs) < 2:
        return None

    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(cosine_similarity(vecs[i], vecs[j]))
    return float(np.mean(sims))


def domain_tag_turnover(turnover, tagger):
    """Tag entered/exited words by domain."""
    entered_domains = tagger.get_domain_distribution(turnover["entered"])
    exited_domains = tagger.get_domain_distribution(turnover["exited"])
    return {"entered_domains": entered_domains, "exited_domains": exited_domains}


def run_analysis():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "sgns"

    print("=" * 70)
    print("NEIGHBORHOOD DYNAMICS ANALYSIS")
    print("How does freedom's semantic context restructure?")
    print("=" * 70)
    print()

    print("Loading HistWords embeddings...")
    emb = TemporalEmbeddings(str(data_dir))
    emb.load_decades(start=1800, end=1990, step=10)
    decades = emb.decades
    print()

    tagger = DomainTagger()
    all_words = ["freedom"] + [w for w in CONTROL_WORDS if emb.word_exists(w, 1800) and emb.word_exists(w, 1990)]

    results = {
        "centroid_displacement": {},
        "cumulative_displacement": {},
        "second_order_similarity": {},
        "freedom_turnover": {},
        "freedom_domain_shifts": {},
    }

    # =========================================================================
    # 1. CENTROID DISPLACEMENT: freedom vs control words
    # =========================================================================
    print("=" * 70)
    print("1. NEIGHBORHOOD CENTROID DISPLACEMENT (1800→1990)")
    print("   How much did each word's neighborhood move?")
    print("=" * 70)
    print()

    print(f"{'Word':<18} {'Displacement':>14} {'Rank':>6}")
    print("-" * 40)

    displacements = {}
    for word in all_words:
        d = centroid_displacement(emb, word, 1800, 1990)
        if d is not None:
            displacements[word] = round(d, 4)

    ranked = sorted(displacements.items(), key=lambda x: x[1], reverse=True)
    for rank, (word, d) in enumerate(ranked, 1):
        marker = " <-- TARGET" if word == "freedom" else ""
        print(f"  {word:<16} {d:>12.4f} {rank:>5}{marker}")

    results["centroid_displacement"] = displacements
    freedom_rank = next(i for i, (w, _) in enumerate(ranked, 1) if w == "freedom")
    results["freedom_displacement_rank"] = f"{freedom_rank}/{len(ranked)}"
    print()
    print(f"  'freedom' neighborhood displacement rank: #{freedom_rank}/{len(ranked)}")
    print()

    # =========================================================================
    # 2. DECADE-BY-DECADE DISPLACEMENT TRAJECTORY
    # =========================================================================
    print("=" * 70)
    print("2. DECADE-BY-DECADE DISPLACEMENT (adjacent decades)")
    print("=" * 70)
    print()

    trajectory = {}
    for word in ["freedom", "liberty", "justice", "power"]:
        trajectory[word] = {}
        for i in range(len(decades) - 1):
            d = centroid_displacement(emb, word, decades[i], decades[i + 1])
            if d is not None:
                trajectory[word][f"{decades[i]}-{decades[i+1]}"] = round(d, 4)

    # Print freedom's trajectory
    print("Freedom's neighborhood displacement per decade:")
    print("-" * 50)
    for period, d in trajectory["freedom"].items():
        bar = "#" * int(d * 200)
        print(f"  {period}: {d:.4f}  {bar}")

    results["displacement_trajectories"] = trajectory
    print()

    # Find the decade with max displacement for freedom
    max_period = max(trajectory["freedom"], key=trajectory["freedom"].get)
    print(f"  Largest neighborhood shift: {max_period} ({trajectory['freedom'][max_period]:.4f})")
    print()

    # =========================================================================
    # 3. CUMULATIVE DISPLACEMENT FROM 1800
    # =========================================================================
    print("=" * 70)
    print("3. CUMULATIVE DISPLACEMENT FROM 1800")
    print("=" * 70)
    print()

    key_decades = [1800, 1830, 1850, 1870, 1880, 1900, 1920, 1950, 1970, 1990]
    print(f"{'Decade':<10}", end="")
    for word in ["freedom", "liberty", "justice", "power"]:
        print(f"{word:>12}", end="")
    print()
    print("-" * 58)

    cumulative = {w: {} for w in ["freedom", "liberty", "justice", "power"]}
    for decade in key_decades:
        print(f"  {decade:<8}", end="")
        for word in ["freedom", "liberty", "justice", "power"]:
            d = centroid_displacement(emb, word, 1800, decade) if decade > 1800 else 0.0
            if d is not None:
                cumulative[word][str(decade)] = round(d, 4)
                print(f"{d:>12.4f}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()

    results["cumulative_displacement"] = cumulative
    print()

    # =========================================================================
    # 4. SECOND-ORDER SIMILARITY (neighborhood coherence)
    # =========================================================================
    print("=" * 70)
    print("4. NEIGHBORHOOD COHERENCE (second-order similarity)")
    print("   Are freedom's neighbors becoming more or less similar to each other?")
    print("=" * 70)
    print()

    coherence = {}
    for word in ["freedom", "liberty", "justice", "power"]:
        coherence[word] = {}
        for decade in key_decades:
            s = second_order_similarity(emb, word, decade)
            if s is not None:
                coherence[word][str(decade)] = round(s, 4)

    print(f"{'Decade':<10}", end="")
    for word in ["freedom", "liberty", "justice", "power"]:
        print(f"{word:>12}", end="")
    print()
    print("-" * 58)

    for decade in key_decades:
        print(f"  {decade:<8}", end="")
        for word in ["freedom", "liberty", "justice", "power"]:
            val = coherence[word].get(str(decade))
            if val is not None:
                print(f"{val:>12.4f}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()

    results["second_order_similarity"] = coherence
    print()

    # =========================================================================
    # 5. NEIGHBOR TURNOVER WITH DOMAIN TAGGING
    # =========================================================================
    print("=" * 70)
    print("5. FREEDOM'S NEIGHBOR TURNOVER (domain-tagged)")
    print("=" * 70)
    print()

    # Key transitions
    transitions = [
        (1800, 1850, "Early 19th century"),
        (1850, 1880, "Pre-crossover"),
        (1880, 1920, "New Liberalism era"),
        (1920, 1960, "Mid-20th century"),
        (1960, 1990, "Late 20th century"),
    ]

    for d1, d2, label in transitions:
        turnover = neighbor_turnover(emb, "freedom", d1, d2)
        domain_info = domain_tag_turnover(turnover, tagger)

        print(f"\n  {d1}→{d2} ({label}):")
        print(f"    Jaccard overlap: {turnover['jaccard']:.3f}")
        print(f"    Words entered ({len(turnover['entered'])}): {', '.join(turnover['entered'][:10])}")
        print(f"    Words exited ({len(turnover['exited'])}): {', '.join(turnover['exited'][:10])}")

        # Domain summary for entered/exited
        entered_doms = {k: v for k, v in domain_info["entered_domains"].items() if v > 0}
        exited_doms = {k: v for k, v in domain_info["exited_domains"].items() if v > 0}
        if entered_doms:
            top_entered = sorted(entered_doms.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"    Entered domains: {', '.join(f'{d}({n})' for d, n in top_entered)}")
        if exited_doms:
            top_exited = sorted(exited_doms.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"    Exited domains: {', '.join(f'{d}({n})' for d, n in top_exited)}")

        results["freedom_turnover"][f"{d1}-{d2}"] = {
            "label": label,
            "jaccard": round(turnover["jaccard"], 4),
            "n_entered": len(turnover["entered"]),
            "n_exited": len(turnover["exited"]),
            "entered": turnover["entered"][:20],
            "exited": turnover["exited"][:20],
            "entered_domains": domain_info["entered_domains"],
            "exited_domains": domain_info["exited_domains"],
        }

    print()

    # =========================================================================
    # 6. DOMAIN COMPOSITION OF FREEDOM'S NEIGHBORHOOD OVER TIME
    # =========================================================================
    print("=" * 70)
    print("6. DOMAIN COMPOSITION OF FREEDOM'S NEIGHBORHOOD")
    print("=" * 70)
    print()

    domain_trajectory = {}
    all_domains = list(tagger.lexicons.keys()) + ["untagged"]

    for decade in key_decades:
        nn = emb.get_nearest_neighbors("freedom", decade, K_NEIGHBORS)
        words = [w for w, _ in nn]
        dist = tagger.get_domain_distribution(words)
        domain_trajectory[str(decade)] = dist

    # Print as table
    short_names = {
        "servitude_bondage": "servitude",
        "constraint_liberation": "constraint",
        "political": "political",
        "economic": "economic",
        "personal": "personal",
        "religious": "religious",
        "legal": "legal",
        "abstract_philosophical": "abstract",
        "untagged": "untagged",
    }

    domains_to_show = ["servitude_bondage", "political", "economic", "personal", "religious", "legal", "abstract_philosophical", "untagged"]

    print(f"{'Decade':<10}", end="")
    for d in domains_to_show:
        print(f"{short_names[d]:>10}", end="")
    print()
    print("-" * (10 + 10 * len(domains_to_show)))

    for decade in key_decades:
        print(f"  {decade:<8}", end="")
        for d in domains_to_show:
            count = domain_trajectory[str(decade)].get(d, 0)
            print(f"{count:>10}", end="")
        print()

    results["freedom_domain_shifts"] = domain_trajectory
    print()

    # Track servitude vs personal over time
    servitude_trend = [domain_trajectory[str(d)].get("servitude_bondage", 0) for d in key_decades]
    personal_trend = [domain_trajectory[str(d)].get("personal", 0) for d in key_decades]

    print("  Servitude domain trend:", servitude_trend)
    print("  Personal domain trend: ", personal_trend)
    print()

    # =========================================================================
    # SAVE
    # =========================================================================
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "neighborhood_dynamics.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    freedom_disp = displacements.get("freedom", 0)
    control_disps = [v for k, v in displacements.items() if k != "freedom"]
    if control_disps:
        avg_control = sum(control_disps) / len(control_disps)
        more_displaced = sum(1 for d in control_disps if d < freedom_disp)
        print(f"  Freedom neighborhood displacement: {freedom_disp:.4f}")
        print(f"  Average control word displacement:  {avg_control:.4f}")
        print(f"  Freedom more displaced than {more_displaced}/{len(control_disps)} control words")
        print()
        if more_displaced > len(control_disps) * 0.8:
            print("  Freedom's neighborhood restructured MORE than most words.")
            print("  Combined with low individual drift, this confirms:")
            print("  the word stayed put, but the world around it changed.")
        elif more_displaced > len(control_disps) * 0.5:
            print("  Freedom's neighborhood displacement is above average but not exceptional.")
        else:
            print("  Freedom's neighborhood displacement is within normal range.")

    return results


if __name__ == "__main__":
    run_analysis()
