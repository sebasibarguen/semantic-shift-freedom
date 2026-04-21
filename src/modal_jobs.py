# ABOUTME: Modal cloud app for heavy jobs: Wiki GloVe training + Hansard XML extraction.
# ABOUTME: Imports logic from src.* local modules — single source of truth.

"""
Heavy jobs that benefit from Modal's RAM/CPU/time budget.

All business logic lives in the local src.* modules; this file is the
thin cloud runner that provisions containers, mounts a persistent volume,
and dispatches one of the jobs.

Usage:
    # Train word2vec on a Wikipedia dump and save HistWords-format vectors
    modal run src/modal_jobs.py --job wiki \\
        --dump-url https://dumps.wikimedia.org/enwiki/20250101/enwiki-20250101-pages-articles.xml.bz2 \\
        --label 2024

    # Parse Historic Hansard XML (1803-1918)
    modal run src/modal_jobs.py --job hansard-archive

    # Parse ParlParse XML (1919-2025)
    modal run src/modal_jobs.py --job parlparse

    # Classify freedom/liberty sentences from hansard-speeches-v310.csv
    modal run src/modal_jobs.py --job hansard-sentences

Data in /vol/ persists across runs via a named Modal volume ("freedom-jobs").
Upload source data to the volume first with `modal volume put freedom-jobs ...`.
"""

import modal

app = modal.App("freedom-semantic-shift-jobs")

# Persistent volume for corpora and model artifacts.
volume = modal.Volume.from_name("freedom-jobs", create_if_missing=True)

# Container image. Local `src/` package is mounted so modules are importable.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=2.4.3",
        "pandas>=2.2.0",
        "gensim>=4.4.0",
    )
    .add_local_python_source("src")
)


# -----------------------------------------------------------------------------
# Wikipedia GloVe training (biggest job: multi-hour, GB-scale RAM)
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=14400,           # 4h
    memory=32 * 1024,        # 32 GB
    cpu=8.0,
)
def train_wiki_glove(dump_url: str, label: int = 2024) -> dict:
    """Download a Wikipedia bz2 dump, train word2vec, save HistWords-format vectors."""
    import urllib.request
    from pathlib import Path
    from gensim.models import Word2Vec

    from src.wiki_train import (
        extract_to_text,
        train_from_text,
        save_histwords_format,
    )

    dump_path = Path("/vol/wiki") / "dump.xml.bz2"
    text_path = Path("/vol/wiki") / "corpus.txt"
    model_path = Path("/vol/wiki") / "wiki.model"
    output_dir = Path("/vol/wiki") / "sgns"
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    if not dump_path.exists():
        print(f"Downloading dump from {dump_url}...")
        urllib.request.urlretrieve(dump_url, dump_path)
    else:
        print(f"Dump already on volume: {dump_path}")

    if not text_path.exists():
        print("Extracting text from bz2 dump...")
        extract_to_text(str(dump_path), str(text_path))
    else:
        print(f"Corpus already extracted: {text_path}")

    if not model_path.exists():
        print("Training word2vec (skip-gram, 300d, window=5)...")
        model = train_from_text(str(text_path), str(model_path))
    else:
        print(f"Loading existing model: {model_path}")
        model = Word2Vec.load(str(model_path))

    output_dir.mkdir(exist_ok=True, parents=True)
    save_histwords_format(model, str(output_dir), label=label)
    volume.commit()
    return {
        "status": "ok",
        "output_dir": str(output_dir),
        "label": label,
        "vocab_size": len(model.wv),
    }


# -----------------------------------------------------------------------------
# Hansard Archive XML (1803-1918) — GB-scale XML, single pass
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=7200,            # 2h
    memory=16 * 1024,        # 16 GB
    cpu=4.0,
)
def extract_hansard_archive() -> dict:
    """Parse Historic Hansard XML volumes at /vol/hansard_archive/."""
    from src.hansard_archive_extractor import run_archive_extraction
    from src.domain_tagger import DomainTagger

    tagger = DomainTagger()
    result = run_archive_extraction(
        archive_dir="/vol/hansard_archive",
        output_dir="/vol/sentences/archive",
        domain_tagger=tagger,
    )
    volume.commit()
    return result


# -----------------------------------------------------------------------------
# ParlParse XML (1919-2025)
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=7200,
    memory=16 * 1024,
    cpu=4.0,
)
def extract_parlparse() -> dict:
    """Parse ParlParse debate XML at /vol/parlparse/debates/."""
    from src.parlparse_extractor import extract_from_parlparse
    from src.domain_tagger import DomainTagger

    tagger = DomainTagger()
    result = extract_from_parlparse(
        debates_dir="/vol/parlparse/debates",
        output_dir="/vol/sentences/parlparse",
        domain_tagger=tagger,
    )
    volume.commit()
    return result


# -----------------------------------------------------------------------------
# Hansard Speeches CSV — classify every freedom/liberty sentence
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=7200,
    memory=16 * 1024,
    cpu=4.0,
)
def extract_hansard_sentences(csv_name: str = "hansard-speeches-v310.csv") -> dict:
    """Classify freedom/liberty sentences from the Hansard speeches CSV."""
    from src.hansard_sentence_extractor import run_extraction

    result = run_extraction(
        csv_path=f"/vol/{csv_name}",
        output_dir="/vol/sentences/classified",
    )
    volume.commit()
    return result


# -----------------------------------------------------------------------------
# CLI dispatcher
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def main(job: str, dump_url: str = "", label: int = 2024, csv_name: str = "hansard-speeches-v310.csv"):
    """
    Dispatch one of the heavy jobs.

        job: wiki | hansard-archive | parlparse | hansard-sentences
    """
    if job == "wiki":
        if not dump_url:
            raise SystemExit("--dump-url required for `wiki` job")
        result = train_wiki_glove.remote(dump_url=dump_url, label=label)
    elif job == "hansard-archive":
        result = extract_hansard_archive.remote()
    elif job == "parlparse":
        result = extract_parlparse.remote()
    elif job == "hansard-sentences":
        result = extract_hansard_sentences.remote(csv_name=csv_name)
    else:
        raise SystemExit(
            f"Unknown job: {job}. "
            "Choose: wiki, hansard-archive, parlparse, hansard-sentences"
        )
    print(result)
