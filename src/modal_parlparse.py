# ABOUTME: Modal runner that downloads ParlParse, extracts freedom/liberty sentences.
# ABOUTME: Produces decade-chunked JSON for the full 1919-2025 Hansard corpus.

import modal

app = modal.App("freedom-parlparse")

volume = modal.Volume.from_name("hansard-parlparse", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .apt_install("rsync")
    .add_local_dir("src", remote_path="/app/src")
)


@app.function(image=image, volumes={"/vol": volume}, timeout=7200, memory=8192)
def download_and_extract():
    """Download ParlParse via rsync, extract freedom/liberty sentences."""
    import subprocess
    import sys
    from pathlib import Path

    sys.path.insert(0, "/app/src")
    from domain_tagger import DomainTagger
    from parlparse_extractor import extract_from_parlparse

    debates_dir = "/vol/debates"
    output_dir = "/vol/sentences"

    # Download if not already present
    existing = list(Path(debates_dir).glob("*.xml")) if Path(debates_dir).exists() else []
    if len(existing) < 19000:
        print(f"Downloading ParlParse debates (have {len(existing)} files)...")
        Path(debates_dir).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-a", "--progress",
             "data.theyworkforyou.com::parldata/scrapedxml/debates/",
             debates_dir + "/"],
            check=True,
        )
        volume.commit()
        xml_count = len(list(Path(debates_dir).glob("*.xml")))
        print(f"Downloaded {xml_count} XML files")
    else:
        print(f"Already have {len(existing)} XML files, skipping download")

    # Extract sentences
    print("\nExtracting freedom/liberty sentences...")
    tagger = DomainTagger()
    manifest = extract_from_parlparse(
        debates_dir=debates_dir,
        output_dir=output_dir,
        domain_tagger=tagger,
    )
    volume.commit()

    return manifest


@app.function(image=image, volumes={"/vol": volume}, timeout=120)
def get_file(filename):
    """Read a single output file from the volume."""
    import json
    from pathlib import Path
    fpath = Path(f"/vol/sentences/{filename}")
    if not fpath.exists():
        return None
    with open(fpath) as f:
        return json.load(f)


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("Running ParlParse extraction on Modal...")
    manifest = download_and_extract.remote()

    print(f"\nTotal: {manifest['total_sentences']:,} sentences")

    # Download each file
    data_dir = Path("web/data_full")
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "index.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for decade_str, info in manifest["files"].items():
        print(f"  Fetching {info['file']}...")
        data = get_file.remote(info["file"])
        if data:
            with open(data_dir / info["file"], "w") as f:
                json.dump(data, f, separators=(",", ":"))
            print(f"    {len(data):,} sentences")

    print(f"\nAll files written to {data_dir}/")
