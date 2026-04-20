# ABOUTME: Modal runner for Hansard sentence extraction.
# ABOUTME: Extracts + classifies all freedom/liberty sentences, writes chunked JSON.

import modal

app = modal.App("freedom-hansard-sentences")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas")
    .apt_install("wget", "unzip")
    .run_commands("mkdir -p /data && wget -q https://zenodo.org/records/4843485/files/hansard-speeches-v310.csv.zip -O /data/hansard.csv.zip && unzip -o /data/hansard.csv.zip -d /data && rm /data/hansard.csv.zip")
    .add_local_dir("src", remote_path="/app/src")
    .add_local_file("outputs/hansard_analysis.json", remote_path="/app/outputs/hansard_analysis.json")
)


@app.function(image=image, timeout=3600, memory=16384)
def run_extraction():
    import sys
    import json
    import shutil
    from pathlib import Path

    sys.path.insert(0, "/app/src")
    from hansard_sentence_extractor import run_extraction as extract

    output_dir = "/tmp/hansard_data"
    extract(csv_path="/data/hansard-speeches-v310.csv", output_dir=output_dir)

    # Collect all output files into a single result dict
    result = {"files": {}}
    for p in Path(output_dir).iterdir():
        with open(p) as f:
            if p.name == "index.json":
                result["index"] = json.load(f)
            else:
                result["files"][p.name] = json.load(f)

    return result


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("Extracting Hansard sentences on Modal...")
    result = run_extraction.remote()

    # Write files locally
    data_dir = Path("web/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "index.json", "w") as f:
        json.dump(result["index"], f, indent=2)
    print(f"  index.json: {result['index']['total_sentences']:,} sentences")

    for filename, data in result["files"].items():
        with open(data_dir / filename, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"  {filename}: {len(data):,} sentences")

    print(f"\nAll files written to web/data/")
