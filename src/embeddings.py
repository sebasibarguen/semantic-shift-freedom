# ABOUTME: Loads and manages HistWords temporal word embeddings.
# ABOUTME: Provides methods to access word vectors across different decades.

import pickle
import numpy as np
from pathlib import Path
from typing import Optional


class TemporalEmbeddings:
    """Manages word embeddings across multiple time periods."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.embeddings = {}  # {decade: {'vocab': list, 'word2idx': dict, 'matrix': np.array}}
        self.decades = []

    def load_decade(self, decade: int) -> None:
        """Load embeddings for a specific decade."""
        vocab_path = self.data_dir / f"{decade}-vocab.pkl"
        matrix_path = self.data_dir / f"{decade}-w.npy"

        if not vocab_path.exists() or not matrix_path.exists():
            raise FileNotFoundError(f"Embeddings for {decade} not found")

        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        matrix = np.load(matrix_path, mmap_mode='r')
        word2idx = {word: idx for idx, word in enumerate(vocab)}

        self.embeddings[decade] = {
            'vocab': vocab,
            'word2idx': word2idx,
            'matrix': matrix
        }

        if decade not in self.decades:
            self.decades.append(decade)
            self.decades.sort()

    def load_decades(self, start: int = 1800, end: int = 1990, step: int = 10) -> None:
        """Load embeddings for a range of decades."""
        for decade in range(start, end + 1, step):
            try:
                self.load_decade(decade)
                print(f"Loaded {decade}")
            except FileNotFoundError:
                print(f"Skipping {decade} - not found")

    def get_vector(self, word: str, decade: int) -> Optional[np.ndarray]:
        """Get the embedding vector for a word in a specific decade."""
        if decade not in self.embeddings:
            return None

        emb = self.embeddings[decade]
        if word not in emb['word2idx']:
            return None

        idx = emb['word2idx'][word]
        return np.array(emb['matrix'][idx])  # Copy from mmap

    def word_exists(self, word: str, decade: int) -> bool:
        """Check if a word exists in a specific decade's vocabulary."""
        if decade not in self.embeddings:
            return False
        return word in self.embeddings[decade]['word2idx']

    def get_nearest_neighbors(self, word: str, decade: int, k: int = 50) -> list[tuple[str, float]]:
        """Get the k nearest neighbors of a word in a specific decade."""
        vec = self.get_vector(word, decade)
        if vec is None:
            return []

        emb = self.embeddings[decade]
        matrix = emb['matrix']
        vocab = emb['vocab']

        # Normalize vectors for cosine similarity
        vec_norm = vec / np.linalg.norm(vec)

        # Compute similarities (matrix is large, so we do this efficiently)
        # Note: matrix rows may not be pre-normalized
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        similarities = matrix @ vec_norm / norms

        # Get top k+1 (includes the word itself)
        top_indices = np.argsort(similarities)[::-1][:k + 1]

        results = []
        for idx in top_indices:
            if vocab[idx] != word:
                results.append((vocab[idx], float(similarities[idx])))
            if len(results) >= k:
                break

        return results

    def get_vocab_size(self, decade: int) -> int:
        """Get vocabulary size for a decade."""
        if decade not in self.embeddings:
            return 0
        return len(self.embeddings[decade]['vocab'])
