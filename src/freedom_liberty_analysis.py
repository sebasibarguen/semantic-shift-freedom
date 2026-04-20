# ABOUTME: Focused analysis of freedom vs liberty divergence and legal→personal shift
# ABOUTME: Uses HistWords embeddings (1800-1990) to answer two specific questions

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from embeddings import TemporalEmbeddings


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def cosine_distance(v1, v2):
    """Compute cosine distance (1 - similarity)."""
    return 1.0 - cosine_similarity(v1, v2)


def cluster_distance(embeddings, word, cluster_words, decade):
    """Compute average distance from word to a cluster of words."""
    word_vec = embeddings.get_vector(word, decade)
    if word_vec is None:
        return None
    
    distances = []
    for cw in cluster_words:
        cw_vec = embeddings.get_vector(cw, decade)
        if cw_vec is not None:
            distances.append(cosine_distance(word_vec, cw_vec))
    
    if not distances:
        return None
    return sum(distances) / len(distances)


def run_analysis():
    """Run the focused freedom/liberty analysis."""
    
    print("=" * 70)
    print("FREEDOM vs LIBERTY: Semantic Divergence Analysis (1800-1990)")
    print("=" * 70)
    print()
    
    # Load embeddings
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "sgns"
    
    print("Loading HistWords embeddings...")
    emb = TemporalEmbeddings(str(data_dir))
    emb.load_decades(start=1800, end=1990, step=10)
    print()
    
    # Key decades for summary
    key_decades = [1800, 1850, 1900, 1950, 1990]
    all_decades = emb.decades
    
    results = {
        "freedom_liberty_similarity": {},
        "legal_cluster_distance": {},
        "personal_cluster_distance": {},
        "freedom_neighbors": {},
        "liberty_neighbors": {},
    }
    
    # =========================================================================
    # QUESTION 1: Are freedom and liberty diverging?
    # =========================================================================
    print("=" * 70)
    print("QUESTION 1: Are 'freedom' and 'liberty' diverging?")
    print("=" * 70)
    print()
    
    print("Cosine SIMILARITY between 'freedom' and 'liberty' by decade:")
    print("(Higher = more similar, Lower = diverging)")
    print("-" * 50)
    
    for decade in all_decades:
        freedom_vec = emb.get_vector("freedom", decade)
        liberty_vec = emb.get_vector("liberty", decade)
        
        if freedom_vec is not None and liberty_vec is not None:
            sim = cosine_similarity(freedom_vec, liberty_vec)
            results["freedom_liberty_similarity"][decade] = sim
            marker = " <--" if decade in key_decades else ""
            print(f"  {decade}: {sim:.4f}{marker}")
    
    print()
    
    # Summary
    first_decade = min(results["freedom_liberty_similarity"].keys())
    last_decade = max(results["freedom_liberty_similarity"].keys())
    first_sim = results["freedom_liberty_similarity"][first_decade]
    last_sim = results["freedom_liberty_similarity"][last_decade]
    
    print(f"Change from {first_decade} to {last_decade}:")
    print(f"  {first_sim:.4f} → {last_sim:.4f} (Δ = {last_sim - first_sim:+.4f})")
    if last_sim < first_sim:
        print("  → DIVERGING: freedom and liberty are becoming less similar")
    else:
        print("  → CONVERGING: freedom and liberty are becoming more similar")
    print()
    
    # =========================================================================
    # QUESTION 2: Has freedom shifted from legal to personal?
    # =========================================================================
    print("=" * 70)
    print("QUESTION 2: Has 'freedom' shifted from legal/status to personal/capacity?")
    print("=" * 70)
    print()
    
    # Define clusters
    legal_cluster = ["slavery", "bondage", "emancipation", "rights", "law", "citizen", "slave", "servitude"]
    personal_cluster = ["choice", "autonomy", "independence", "self", "ability", "power", "individual", "personal"]
    
    print(f"Legal/Status cluster: {legal_cluster}")
    print(f"Personal/Capacity cluster: {personal_cluster}")
    print()
    
    print("Average cosine DISTANCE from 'freedom' to each cluster:")
    print("(Lower = closer to that meaning)")
    print("-" * 60)
    print(f"{'Decade':<10} {'Legal Dist':<15} {'Personal Dist':<15} {'Closer To'}")
    print("-" * 60)
    
    for decade in all_decades:
        legal_dist = cluster_distance(emb, "freedom", legal_cluster, decade)
        personal_dist = cluster_distance(emb, "freedom", personal_cluster, decade)
        
        if legal_dist is not None and personal_dist is not None:
            results["legal_cluster_distance"][decade] = legal_dist
            results["personal_cluster_distance"][decade] = personal_dist
            
            closer = "LEGAL" if legal_dist < personal_dist else "PERSONAL"
            marker = " <--" if decade in key_decades else ""
            print(f"  {decade:<8} {legal_dist:<15.4f} {personal_dist:<15.4f} {closer}{marker}")
    
    print()
    
    # Find crossover point
    crossover = None
    prev_closer = None
    for decade in sorted(results["legal_cluster_distance"].keys()):
        legal_d = results["legal_cluster_distance"][decade]
        personal_d = results["personal_cluster_distance"][decade]
        closer = "legal" if legal_d < personal_d else "personal"
        
        if prev_closer and prev_closer != closer:
            crossover = decade
            break
        prev_closer = closer
    
    if crossover:
        print(f"CROSSOVER DETECTED around {crossover}:")
        print(f"  Before: 'freedom' was closer to LEGAL/STATUS concepts")
        print(f"  After: 'freedom' became closer to PERSONAL/CAPACITY concepts")
    else:
        # Check which it's consistently closer to
        legal_closer_count = sum(1 for d in results["legal_cluster_distance"] 
                                  if results["legal_cluster_distance"][d] < results["personal_cluster_distance"][d])
        print(f"No clear crossover. Freedom closer to LEGAL in {legal_closer_count}/{len(results['legal_cluster_distance'])} decades")
    print()
    
    # =========================================================================
    # NEIGHBOR ANALYSIS
    # =========================================================================
    print("=" * 70)
    print("NEAREST NEIGHBORS EVOLUTION")
    print("=" * 70)
    print()
    
    for word in ["freedom", "liberty"]:
        print(f"Top 10 neighbors of '{word}' over time:")
        print("-" * 60)
        for decade in key_decades:
            neighbors = emb.get_nearest_neighbors(word, decade, k=10)
            results[f"{word}_neighbors"][decade] = [(w, float(s)) for w, s in neighbors]
            neighbor_str = ", ".join([w for w, s in neighbors[:10]])
            print(f"  {decade}: {neighbor_str}")
        print()
    
    # =========================================================================
    # SPECIFIC WORD DISTANCES
    # =========================================================================
    print("=" * 70)
    print("DISTANCE FROM 'FREEDOM' TO KEY CONCEPTS")
    print("=" * 70)
    print()
    
    key_concepts = ["liberty", "slavery", "autonomy", "choice", "rights", "independence", "power", "self"]
    
    print(f"{'Concept':<15}", end="")
    for decade in key_decades:
        print(f"{decade:<10}", end="")
    print()
    print("-" * 65)
    
    concept_trajectories = {}
    for concept in key_concepts:
        print(f"{concept:<15}", end="")
        concept_trajectories[concept] = {}
        for decade in key_decades:
            freedom_vec = emb.get_vector("freedom", decade)
            concept_vec = emb.get_vector(concept, decade)
            if freedom_vec is not None and concept_vec is not None:
                dist = cosine_distance(freedom_vec, concept_vec)
                concept_trajectories[concept][decade] = dist
                print(f"{dist:<10.3f}", end="")
            else:
                print(f"{'N/A':<10}", end="")
        print()
    
    print()
    print("Key observations:")
    
    # Check autonomy trajectory
    if "autonomy" in concept_trajectories and 1800 in concept_trajectories["autonomy"] and 1990 in concept_trajectories["autonomy"]:
        auto_1800 = concept_trajectories["autonomy"][1800]
        auto_1990 = concept_trajectories["autonomy"][1990]
        print(f"  - 'autonomy': {auto_1800:.3f} → {auto_1990:.3f} (Δ = {auto_1990 - auto_1800:+.3f})")
        if auto_1990 < auto_1800:
            print(f"    → Freedom moved CLOSER to autonomy")
    
    # Check slavery trajectory
    if "slavery" in concept_trajectories and 1800 in concept_trajectories["slavery"] and 1990 in concept_trajectories["slavery"]:
        slav_1800 = concept_trajectories["slavery"][1800]
        slav_1990 = concept_trajectories["slavery"][1990]
        print(f"  - 'slavery': {slav_1800:.3f} → {slav_1990:.3f} (Δ = {slav_1990 - slav_1800:+.3f})")
        if slav_1990 > slav_1800:
            print(f"    → Freedom moved AWAY from slavery")
    
    print()
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Convert numpy types for JSON
    json_results = {}
    for key, val in results.items():
        if isinstance(val, dict):
            json_results[key] = {str(k): v for k, v in val.items()}
        else:
            json_results[key] = val
    
    json_results["concept_distances"] = {
        concept: {str(k): v for k, v in traj.items()}
        for concept, traj in concept_trajectories.items()
    }
    
    output_file = output_dir / "freedom_liberty_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Q1: Are 'freedom' and 'liberty' diverging?")
    print(f"    Similarity: {first_sim:.3f} ({first_decade}) → {last_sim:.3f} ({last_decade})")
    if last_sim < first_sim:
        print("    ANSWER: Yes, they are diverging")
    else:
        print("    ANSWER: No clear divergence")
    print()
    print("Q2: Has 'freedom' shifted from legal/status to personal/capacity?")
    if crossover:
        print(f"    ANSWER: Yes, crossover around {crossover}")
    else:
        print("    ANSWER: No clear crossover detected")
    print()
    
    return results


if __name__ == "__main__":
    run_analysis()
