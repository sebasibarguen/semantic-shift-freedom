# ABOUTME: Implements semantic change detection metrics.
# ABOUTME: Provides cosine distance and neighbor overlap (Jaccard) calculations.

import numpy as np
from typing import Optional
from .embeddings import TemporalEmbeddings


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity) between two vectors."""
    return 1.0 - cosine_similarity(v1, v2)


def semantic_change_score(
    embeddings: TemporalEmbeddings,
    word: str,
    decade1: int,
    decade2: int
) -> Optional[float]:
    """
    Compute semantic change score for a word between two time periods.
    Returns cosine distance (higher = more change).

    Note: HistWords embeddings are pre-aligned using orthogonal Procrustes,
    so we can directly compare vectors across decades.
    """
    v1 = embeddings.get_vector(word, decade1)
    v2 = embeddings.get_vector(word, decade2)

    if v1 is None or v2 is None:
        return None

    return cosine_distance(v1, v2)


def neighbor_overlap(
    embeddings: TemporalEmbeddings,
    word: str,
    decade1: int,
    decade2: int,
    k: int = 50
) -> Optional[float]:
    """
    Compute Jaccard overlap of k nearest neighbors between two time periods.
    Higher value = more stable semantics (neighbors are preserved).
    """
    nn1 = embeddings.get_nearest_neighbors(word, decade1, k)
    nn2 = embeddings.get_nearest_neighbors(word, decade2, k)

    if not nn1 or not nn2:
        return None

    words1 = set(w for w, _ in nn1)
    words2 = set(w for w, _ in nn2)

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    if union == 0:
        return None

    return intersection / union


def compute_trajectory(
    embeddings: TemporalEmbeddings,
    word: str,
    metric: str = "cosine"
) -> dict:
    """
    Compute semantic change trajectory for a word across all loaded decades.

    Returns dict with:
    - 'decades': list of decades
    - 'pairwise_scores': dict of (decade1, decade2) -> score for adjacent decades
    - 'cumulative_scores': dict of decade -> score relative to earliest decade
    - 'total_change': score between first and last decade
    """
    decades = embeddings.decades

    # Check word exists in all decades
    available_decades = [d for d in decades if embeddings.word_exists(word, d)]

    if len(available_decades) < 2:
        return {'error': f"Word '{word}' not found in enough decades"}

    results = {
        'word': word,
        'metric': metric,
        'decades': available_decades,
        'pairwise_scores': {},
        'cumulative_scores': {},
    }

    # Compute pairwise scores (adjacent decades)
    for i in range(len(available_decades) - 1):
        d1, d2 = available_decades[i], available_decades[i + 1]
        if metric == "cosine":
            score = semantic_change_score(embeddings, word, d1, d2)
        elif metric == "neighbor":
            # For neighbor overlap, we compute 1 - overlap so higher = more change
            overlap = neighbor_overlap(embeddings, word, d1, d2)
            score = 1 - overlap if overlap is not None else None
        else:
            raise ValueError(f"Unknown metric: {metric}")

        results['pairwise_scores'][(d1, d2)] = score

    # Compute cumulative scores (relative to first decade)
    first_decade = available_decades[0]
    results['cumulative_scores'][first_decade] = 0.0

    for decade in available_decades[1:]:
        if metric == "cosine":
            score = semantic_change_score(embeddings, word, first_decade, decade)
        elif metric == "neighbor":
            overlap = neighbor_overlap(embeddings, word, first_decade, decade)
            score = 1 - overlap if overlap is not None else None
        results['cumulative_scores'][decade] = score

    # Total change (first to last)
    last_decade = available_decades[-1]
    results['total_change'] = results['cumulative_scores'].get(last_decade)

    return results


def compare_words(
    embeddings: TemporalEmbeddings,
    words: list[str],
    metric: str = "cosine"
) -> dict:
    """
    Compare semantic change trajectories for multiple words.

    Returns dict mapping word -> trajectory results.
    """
    return {word: compute_trajectory(embeddings, word, metric) for word in words}
