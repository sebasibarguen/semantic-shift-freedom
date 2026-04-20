# ABOUTME: Modal app that runs embedding analysis on cloud infrastructure.
# ABOUTME: Downloads COHA HistWords data, runs analysis alongside Google Books data.

import modal

app = modal.App("freedom-semantic-shift")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy")
    .apt_install("wget", "unzip")
    # COHA HistWords (1830s-2000s)
    .run_commands("mkdir -p /data/coha_sgns && wget -q http://snap.stanford.edu/historical_embeddings/coha-word_sgns.zip -O /tmp/coha.zip && unzip -o /tmp/coha.zip -d /tmp/coha_extract && find /tmp/coha_extract -name '*.pkl' -exec mv {} /data/coha_sgns/ \\; && find /tmp/coha_extract -name '*.npy' -exec mv {} /data/coha_sgns/ \\; && rm -rf /tmp/coha.zip /tmp/coha_extract && ls /data/coha_sgns/*.pkl | head -3")
    # Google Books HistWords (1800s-1990s)
    .run_commands("mkdir -p /data/sgns && wget -q http://snap.stanford.edu/historical_embeddings/eng-all_sgns.zip -O /tmp/gbooks.zip && unzip -o /tmp/gbooks.zip -d /tmp/gbooks_extract && find /tmp/gbooks_extract -name '*.pkl' -exec mv {} /data/sgns/ \\; && find /tmp/gbooks_extract -name '*.npy' -exec mv {} /data/sgns/ \\; && rm -rf /tmp/gbooks.zip /tmp/gbooks_extract && ls /data/sgns/*.pkl | head -3")
    .add_local_dir("src", remote_path="/app/src")
)


@app.function(
    image=image,
    timeout=600,
    memory=8192,
)
def run_analysis():
    """Run the full modern embeddings analysis on Modal."""
    import sys
    sys.path.insert(0, "/app/src")

    from modern_embeddings import run_coha_analysis

    results = run_coha_analysis(
        coha_dir="/data/coha_sgns",
        gbooks_dir="/data/sgns",
        output_path="/tmp/modern_embeddings.json",
    )
    return results


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("Launching analysis on Modal...")
    results = run_analysis.remote()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "modern_embeddings.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved locally to: {output_path}")
