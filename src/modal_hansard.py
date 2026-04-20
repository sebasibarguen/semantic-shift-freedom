# ABOUTME: Modal runner for Hansard analysis of freedom/liberty in UK Parliament.
# ABOUTME: Downloads dataset from HuggingFace and runs collocate + frequency analysis.

import modal

app = modal.App("freedom-hansard")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas")
    .apt_install("wget", "unzip")
    .run_commands("mkdir -p /data && wget -q https://zenodo.org/records/4843485/files/hansard-speeches-v310.csv.zip -O /data/hansard.csv.zip && unzip -o /data/hansard.csv.zip -d /data && rm /data/hansard.csv.zip")
    .add_local_dir("src", remote_path="/app/src")
)


@app.function(image=image, timeout=3600, memory=16384)
def run_analysis():
    import sys
    sys.path.insert(0, "/app/src")
    from hansard_analysis import run_hansard_analysis
    return run_hansard_analysis(output_path="/tmp/hansard_analysis.json")


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("Running Hansard analysis on Modal...")
    results = run_analysis.remote()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    out = output_dir / "hansard_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out}")
