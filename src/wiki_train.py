# ABOUTME: Extracts text from Wikipedia bz2 dumps and trains word2vec.
# ABOUTME: Saves trained models in HistWords format for use with TemporalEmbeddings.

import re
import bz2
import pickle
import numpy as np
from pathlib import Path
from xml.etree.ElementTree import iterparse


# HistWords-matching hyperparameters
W2V_PARAMS = {
    "vector_size": 300,
    "window": 5,
    "negative": 15,
    "min_count": 100,
    "sg": 1,  # skip-gram
    "workers": 8,
    "epochs": 1,
}

VOCAB_LIMIT = 50_000

MARKUP_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
TAG_RE = re.compile(r"<[^>]+>")
REF_RE = re.compile(r"\{\{[^}]*\}\}")
TOKENIZE_RE = re.compile(r"[a-z]+")


def strip_markup(text):
    """Remove common MediaWiki markup."""
    text = MARKUP_RE.sub(r"\1", text)
    text = TAG_RE.sub("", text)
    text = REF_RE.sub("", text)
    return text


class WikiDumpCorpus:
    """Streams tokenized sentences from a bz2-compressed MediaWiki dump."""

    def __init__(self, dump_path: str):
        self.dump_path = dump_path

    def __iter__(self):
        count = 0
        with bz2.open(self.dump_path, "rt", errors="replace") as f:
            for event, elem in iterparse(f, events=("end",)):
                if elem.tag.endswith("}text"):
                    text = elem.text
                    if text:
                        text = strip_markup(text.lower())
                        for line in text.split("\n"):
                            words = TOKENIZE_RE.findall(line)
                            if len(words) >= 5:
                                yield words
                    count += 1
                    if count % 100_000 == 0:
                        print(f"  Processed {count:,} articles...")
                elem.clear()


def extract_to_text(dump_path: str, text_path: str):
    """Extract Wikipedia dump to a plain text file (one sentence per line)."""
    print(f"Extracting {dump_path} to {text_path}...")
    count = 0
    with open(text_path, "w") as out:
        for words in WikiDumpCorpus(dump_path):
            out.write(" ".join(words) + "\n")
            count += 1
    print(f"  Extracted {count:,} sentences")
    return text_path


def train_from_text(text_path: str, model_path: str):
    """Train word2vec from a pre-extracted plain text file."""
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence

    print(f"Training word2vec from {text_path}...")
    corpus = LineSentence(text_path)
    model = Word2Vec(corpus, **W2V_PARAMS)
    model.save(model_path)
    print(f"  Vocabulary: {len(model.wv)} words")
    return model


def save_histwords_format(model, output_dir: str, label: int):
    """Convert a gensim word2vec model to HistWords format."""
    wv = model.wv
    vocab = sorted(
        wv.key_to_index.keys(),
        key=lambda w: wv.get_vecattr(w, "count"),
        reverse=True,
    )[:VOCAB_LIMIT]

    matrix = np.array([wv[w] for w in vocab], dtype=np.float32)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{label}-vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    np.save(out / f"{label}-w.npy", matrix)
    print(f"  Saved {label}: {len(vocab)} words, {matrix.shape}")
