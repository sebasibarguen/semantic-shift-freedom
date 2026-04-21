# ABOUTME: Extends embedding analysis using pre-trained GloVe models (2014, 2024).
# ABOUTME: Procrustes-aligns GloVe to COHA 2000s and runs SemAxis projection.

import json
import pickle
import numpy as np
from pathlib import Path


VOCAB_LIMIT = 50_000


def load_glove_txt(glove_path: str, vocab_limit: int = VOCAB_LIMIT):
    """Load a GloVe text file into vocab list + numpy matrix."""
    vocab = []
    vectors = []
    with open(glove_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 10:  # skip malformed lines
                continue
            word = parts[0]
            vec = [float(x) for x in parts[1:]]
            vocab.append(word)
            vectors.append(vec)
            if len(vocab) >= vocab_limit:
                break

    matrix = np.array(vectors, dtype=np.float32)
    print(f"  Loaded {len(vocab)} words, {matrix.shape[1]}d from {glove_path}")
    return vocab, matrix


def save_histwords_format(vocab, matrix, output_dir: str, label: str):
    """Save vocab + matrix in HistWords format."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{label}-vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    np.save(out / f"{label}-w.npy", matrix)
    print(f"  Saved {label}-vocab.pkl ({len(vocab)} words), {label}-w.npy {matrix.shape}")


def procrustes_align(target_vocab_path, target_matrix_path, source_vocab_path, source_matrix_path, output_matrix_path):
    """Align source embeddings to target space using orthogonal Procrustes."""
    from scipy.linalg import orthogonal_procrustes

    with open(target_vocab_path, "rb") as f:
        target_vocab = pickle.load(f)
    target_matrix = np.load(target_matrix_path)
    target_w2i = {w: i for i, w in enumerate(target_vocab)}

    with open(source_vocab_path, "rb") as f:
        source_vocab = pickle.load(f)
    source_matrix = np.load(source_matrix_path)
    source_w2i = {w: i for i, w in enumerate(source_vocab)}

    shared = [w for w in source_vocab if w in target_w2i]
    print(f"  Shared vocabulary: {len(shared)} words")

    target_shared = np.array([target_matrix[target_w2i[w]] for w in shared])
    source_shared = np.array([source_matrix[source_w2i[w]] for w in shared])

    R, _ = orthogonal_procrustes(source_shared, target_shared)

    aligned = source_matrix @ R
    np.save(output_matrix_path, aligned.astype(np.float32))
    print(f"  Aligned matrix saved to {output_matrix_path}")
    return len(shared)


def run_glove_analysis(coha_dir: str, glove_dir: str, output_path: str):
    """
    Run SemAxis analysis on GloVe embeddings aligned to COHA.
    Expects glove_dir to have aligned HistWords-format files labeled 2014 and 2024.
    """
    from .embeddings import TemporalEmbeddings
    from .metrics import cosine_similarity
    from .semantic_axis import (
        CONSTRAINT_SEEDS, AGENCY_SEEDS, CONTROL_WORDS,
        expand_pole, build_axis, project_onto_axis, linear_trend,
        EXPANSION_K,
    )

    results = {"coha": {}, "glove": {}, "combined": {}}

    # Load COHA
    print("Loading COHA embeddings...")
    coha = TemporalEmbeddings(coha_dir)
    coha.load_decades(start=1830, end=2000, step=10)
    coha_decades = coha.decades

    # Load GloVe (aligned)
    print("Loading GloVe embeddings...")
    glove = TemporalEmbeddings(glove_dir)
    for label in [2014, 2024]:
        try:
            glove.load_decade(label)
        except FileNotFoundError:
            print(f"  Skipping {label} — not found")
    glove_labels = glove.decades
    print(f"GloVe snapshots: {glove_labels}")

    # Build axis from COHA 2000s
    ref = 2000
    constraint_expanded = expand_pole(coha, CONSTRAINT_SEEDS, ref, EXPANSION_K)
    agency_expanded = expand_pole(coha, AGENCY_SEEDS, ref, EXPANSION_K)
    overlap = set(constraint_expanded) & set(agency_expanded)
    constraint_expanded = [w for w in constraint_expanded if w not in overlap]
    agency_expanded = [w for w in agency_expanded if w not in overlap]
    coha_axis = build_axis(coha, constraint_expanded, agency_expanded, ref)

    if coha_axis is None:
        print("ERROR: Could not build axis from COHA 2000s")
        return results

    results["axis_source"] = "COHA 2000s"
    results["axis_words"] = {
        "constraint": len(constraint_expanded),
        "agency": len(agency_expanded),
    }

    # Project COHA decades
    all_words = ["freedom"] + [w for w in CONTROL_WORDS if coha.word_exists(w, ref)]

    coha_projections = {}
    for word in all_words:
        word_proj = {}
        for decade in coha_decades:
            v = coha.get_vector(word, decade)
            if v is not None:
                word_proj[str(decade)] = round(float(np.dot(v, coha_axis)), 4)
        coha_projections[word] = word_proj
    results["coha"]["projections"] = coha_projections

    # Project GloVe using the SAME axis (Procrustes-aligned to COHA space)
    glove_projections = {}
    for word in all_words:
        word_proj = {}
        for label in glove_labels:
            v = glove.get_vector(word, label)
            if v is not None:
                word_proj[str(label)] = round(float(np.dot(v, coha_axis)), 4)
        glove_projections[word] = word_proj
    results["glove"]["projections"] = glove_projections

    # Combined timeline
    print()
    print("=" * 70)
    print("COMBINED TIMELINE: Freedom on constraint→agency axis")
    print("(COHA 1830-2000, GloVe 2014/2024)")
    print("=" * 70)
    print()

    combined_freedom = {}
    for d, v in coha_projections.get("freedom", {}).items():
        combined_freedom[d] = {"value": v, "source": "COHA"}
    for d, v in glove_projections.get("freedom", {}).items():
        combined_freedom[d] = {"value": v, "source": "GloVe"}

    print(f"  {'Year':<10} {'Projection':>12} {'Source':<12}")
    print(f"  {'-'*34}")
    for d in sorted(combined_freedom.keys()):
        entry = combined_freedom[d]
        print(f"  {d:<10} {entry['value']:>+12.4f} {entry['source']:<12}")

    results["combined"]["freedom_timeline"] = combined_freedom

    # Freedom-liberty similarity
    print()
    print("=" * 70)
    print("FREEDOM-LIBERTY SIMILARITY")
    print("=" * 70)
    print()

    fl_sim = {}
    for label in glove_labels:
        v_free = glove.get_vector("freedom", label)
        v_lib = glove.get_vector("liberty", label)
        if v_free is not None and v_lib is not None:
            sim = float(cosine_similarity(v_free, v_lib))
            fl_sim[str(label)] = round(sim, 4)
            print(f"  GloVe {label}: {sim:.4f}")
    results["glove"]["freedom_liberty_similarity"] = fl_sim

    # Add COHA for comparison
    coha_fl = {}
    for decade in coha_decades:
        v_free = coha.get_vector("freedom", decade)
        v_lib = coha.get_vector("liberty", decade)
        if v_free is not None and v_lib is not None:
            coha_fl[str(decade)] = round(float(cosine_similarity(v_free, v_lib)), 4)
    results["coha"]["freedom_liberty_similarity"] = coha_fl
    print(f"  COHA 2000: {coha_fl.get('2000', 'N/A')}")

    # Trend analysis
    print()
    print("=" * 70)
    print("TREND ANALYSIS")
    print("=" * 70)
    print()

    coha_freedom = coha_projections.get("freedom", {})
    coha_d = [int(d) for d in sorted(coha_freedom.keys())]
    coha_v = [coha_freedom[str(d)] for d in coha_d]
    coha_trend = linear_trend(coha_d, coha_v)
    if coha_trend:
        print(f"  COHA only (1830-2000): {coha_trend['slope_per_century']:+.4f}/century (R²={coha_trend['r_squared']:.3f})")
        results["coha"]["trend"] = coha_trend

    # Combined
    all_decades = {}
    for d, v in coha_projections.get("freedom", {}).items():
        all_decades[int(d)] = v
    for d, v in glove_projections.get("freedom", {}).items():
        all_decades[int(d)] = v

    combined_d = sorted(all_decades.keys())
    combined_v = [all_decades[d] for d in combined_d]
    combined_trend = linear_trend(combined_d, combined_v)
    if combined_trend:
        print(f"  Combined (1830-2024):  {combined_trend['slope_per_century']:+.4f}/century (R²={combined_trend['r_squared']:.3f})")
        results["combined"]["trend"] = combined_trend

    # Control words
    print()
    print(f"  {'Word':<18} {'COHA slope':>14} {'Combined slope':>16}")
    print(f"  {'-'*50}")
    for word in all_words:
        coha_wd = coha_projections.get(word, {})
        cd = [int(d) for d in sorted(coha_wd.keys())]
        cv = [coha_wd[str(d)] for d in cd]
        ct = linear_trend(cd, cv)

        glove_wd = glove_projections.get(word, {})
        all_wd = dict(coha_wd)
        all_wd.update(glove_wd)
        ad = [int(d) for d in sorted(all_wd.keys())]
        av = [all_wd[str(d)] for d in ad]
        at = linear_trend(ad, av)

        cs = f"{ct['slope_per_century']:+.4f}" if ct else "N/A"
        as_ = f"{at['slope_per_century']:+.4f}" if at else "N/A"
        marker = " <--" if word == "freedom" else ""
        print(f"  {word:<18} {cs:>14} {as_:>16}{marker}")

    # Methodology note
    print()
    print("=" * 70)
    print("METHODOLOGY NOTE")
    print("=" * 70)
    print()
    print("  GloVe ≠ SGNS (word2vec). The COHA HistWords uses SGNS, while GloVe")
    print("  uses a different factorization method. Procrustes alignment rotates")
    print("  the GloVe space to match COHA, but the underlying geometry differs.")
    print("  Interpret the GloVe data points as directional indicators, not as")
    print("  precise continuations of the COHA trend line.")

    results["methodology_note"] = (
        "GloVe and SGNS are different embedding methods. Procrustes alignment "
        "matches the spaces but the underlying geometry differs. The GloVe "
        "data points are directional indicators, not precise COHA continuations."
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results
