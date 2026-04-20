# ABOUTME: Modal app for Tier 2 — GloVe pre-trained embeddings (2014, 2024).
# ABOUTME: Downloads GloVe models, aligns to COHA, runs SemAxis analysis.

import modal

app = modal.App("freedom-glove-embeddings")

volume = modal.Volume.from_name("wiki-embeddings-data", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy")
    .apt_install("wget", "unzip")
    # COHA embeddings for alignment
    .run_commands("mkdir -p /data/coha_sgns && wget -q http://snap.stanford.edu/historical_embeddings/coha-word_sgns.zip -O /tmp/coha.zip && unzip -o /tmp/coha.zip -d /tmp/coha_extract && find /tmp/coha_extract -name '*.pkl' -exec mv {} /data/coha_sgns/ \\; && find /tmp/coha_extract -name '*.npy' -exec mv {} /data/coha_sgns/ \\; && rm -rf /tmp/coha.zip /tmp/coha_extract")
    # GloVe 2014 (Wikipedia 2014 + Gigaword 5, 300d)
    .run_commands("mkdir -p /data/glove && wget -q https://nlp.stanford.edu/data/glove.6B.zip -O /tmp/glove6b.zip && unzip -o /tmp/glove6b.zip glove.6B.300d.txt -d /data/glove && rm /tmp/glove6b.zip")
    # GloVe 2024 (Wikipedia 2024 + Gigaword 5, 300d)
    .run_commands("wget -q https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.300d.zip -O /tmp/glove2024.zip && unzip -o /tmp/glove2024.zip -d /data/glove && rm /tmp/glove2024.zip")
    .add_local_dir("src", remote_path="/app/src")
)


@app.function(
    image=base_image,
    volumes={"/vol": volume},
    timeout=1800,
    memory=16384,
)
def run_pipeline():
    """Convert GloVe to HistWords format, Procrustes-align, and analyze."""
    import sys
    import shutil
    from pathlib import Path

    sys.path.insert(0, "/app/src")
    from wiki_embeddings import (
        load_glove_txt, save_histwords_format, procrustes_align, run_glove_analysis,
    )

    coha_dir = "/data/coha_sgns"
    glove_raw_dir = "/data/glove"
    glove_hw_dir = "/vol/glove_sgns"
    aligned_dir = "/vol/glove_sgns_aligned"

    Path(glove_hw_dir).mkdir(parents=True, exist_ok=True)
    Path(aligned_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Convert GloVe text files to HistWords format
    print("=" * 70)
    print("STEP 1: Converting GloVe to HistWords format")
    print("=" * 70)
    print()

    glove_files = {
        2014: f"{glove_raw_dir}/glove.6B.300d.txt",
        2024: f"{glove_raw_dir}/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt",
    }

    for label, path in glove_files.items():
        if not Path(path).exists():
            print(f"  WARNING: {path} not found, listing dir:")
            for p in Path(glove_raw_dir).iterdir():
                print(f"    {p.name} ({p.stat().st_size / 1024**2:.0f} MB)")
            continue

        print(f"  Loading GloVe {label}...")
        vocab, matrix = load_glove_txt(path)
        save_histwords_format(vocab, matrix, glove_hw_dir, label)

    volume.commit()

    # Step 2: Procrustes-align to COHA 2000s
    print()
    print("=" * 70)
    print("STEP 2: Procrustes alignment to COHA 2000s")
    print("=" * 70)
    print()

    for label in [2014, 2024]:
        source_vocab = f"{glove_hw_dir}/{label}-vocab.pkl"
        source_matrix = f"{glove_hw_dir}/{label}-w.npy"

        if not Path(source_vocab).exists():
            print(f"  Skipping {label} — not found")
            continue

        target_vocab = f"{coha_dir}/2000-vocab.pkl"
        target_matrix = f"{coha_dir}/2000-w.npy"
        output_matrix = f"{aligned_dir}/{label}-w.npy"

        print(f"  Aligning {label}...")
        procrustes_align(target_vocab, target_matrix, source_vocab, source_matrix, output_matrix)
        shutil.copy2(source_vocab, f"{aligned_dir}/{label}-vocab.pkl")

    volume.commit()

    # Step 3: Run SemAxis analysis
    print()
    print("=" * 70)
    print("STEP 3: Running analysis")
    print("=" * 70)
    print()

    results = run_glove_analysis(
        coha_dir=coha_dir,
        glove_dir=aligned_dir,
        output_path="/vol/glove_analysis.json",
    )

    volume.commit()
    return results


@app.function(
    image=base_image,
    volumes={"/vol": volume},
    timeout=60,
)
def fetch_results():
    """Fetch results from the volume."""
    import json
    from pathlib import Path

    for name in ["glove_analysis.json", "wiki_embeddings.json"]:
        p = Path(f"/vol/{name}")
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return {"error": "No results found in volume."}


@app.local_entrypoint()
def main(fetch_only: bool = False):
    import json
    from pathlib import Path

    if fetch_only:
        results = fetch_results.remote()
        if "error" in results:
            print(results["error"])
            return
    else:
        print("Running GloVe analysis pipeline on Modal...")
        results = run_pipeline.remote()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "wiki_embeddings.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved locally to: {output_path}")
