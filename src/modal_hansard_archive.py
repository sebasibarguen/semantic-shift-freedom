# ABOUTME: Modal runner for downloading and extracting Hansard Archive (1803-1918).
# ABOUTME: Downloads 2,493 ZIPs from Parliament, extracts freedom/liberty sentences.

import modal

app = modal.App("freedom-hansard-archive")

volume = modal.Volume.from_name("hansard-archive", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .apt_install("wget", "unzip")
    .add_local_dir("src", remote_path="/app/src")
)


@app.function(image=image, volumes={"/vol": volume}, timeout=14400, memory=8192)
def download_and_extract():
    """Download Hansard Archive ZIPs, extract XML, find freedom/liberty sentences."""
    import subprocess
    import sys
    from pathlib import Path

    sys.path.insert(0, "/app/src")
    from domain_tagger import DomainTagger
    from hansard_archive_extractor import run_archive_extraction

    zip_dir = "/vol/archive_zips"
    xml_dir = "/vol/archive_xml"
    output_dir = "/vol/archive_sentences"

    Path(zip_dir).mkdir(parents=True, exist_ok=True)
    Path(xml_dir).mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_xml = list(Path(xml_dir).glob("*.xml"))
    if len(existing_xml) < 2000:
        # Download URL list
        print("Downloading URL list...")
        url_list = "/tmp/urls.txt"
        subprocess.run([
            "wget", "-q", "-O", url_list,
            "https://raw.githubusercontent.com/econandrew/uk-hansard-archive-urls/master/urls.txt",
        ], check=True)

        with open(url_list) as f:
            urls = [line.strip() for line in f if line.strip()]
        print(f"Found {len(urls)} URLs")

        # Download all ZIPs in parallel using wget
        print("Downloading ZIPs...")
        subprocess.run([
            "wget", "-q", "-P", zip_dir, "-i", url_list,
            "--no-check-certificate", "--timeout=30", "--tries=2",
        ])

        zip_count = len(list(Path(zip_dir).glob("*.zip")))
        print(f"Downloaded {zip_count} ZIPs")

        # Extract all
        print("Extracting XML from ZIPs...")
        for zf in sorted(Path(zip_dir).glob("*.zip")):
            try:
                subprocess.run(
                    ["unzip", "-o", "-q", str(zf), "-d", xml_dir],
                    check=True, capture_output=True,
                )
            except subprocess.CalledProcessError:
                pass  # Some ZIPs may be corrupt

        xml_count = len(list(Path(xml_dir).glob("*.xml")))
        print(f"Extracted {xml_count} XML files")

        # Clean up ZIPs to save volume space
        for zf in Path(zip_dir).glob("*.zip"):
            zf.unlink()

        volume.commit()
    else:
        print(f"Already have {len(existing_xml)} XML files, skipping download")

    # Extract sentences
    print("\nExtracting freedom/liberty sentences...")
    tagger = DomainTagger()
    manifest = run_archive_extraction(
        archive_dir=xml_dir,
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
    fpath = Path(f"/vol/archive_sentences/{filename}")
    if not fpath.exists():
        return None
    with open(fpath) as f:
        return json.load(f)


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("Running Hansard Archive extraction on Modal...")
    manifest = download_and_extract.remote()

    print(f"\nTotal: {manifest['total_sentences']:,} sentences")

    data_dir = Path("web/data_archive")
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
