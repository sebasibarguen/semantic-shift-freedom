# ABOUTME: Enriches extracted Hansard sentences with SBERT and LLM classifications.
# ABOUTME: Adds sentence embedding projection and few-shot LLM liberty classification.

import modal

app = modal.App("freedom-enrich-sentences")

volume = modal.Volume.from_name("hansard-sentences-v2", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "sentence-transformers", "anthropic")
    .add_local_dir("web/data", remote_path="/app/data")
    .add_local_file("outputs/improved_haiku_prompt.txt", remote_path="/app/outputs/improved_haiku_prompt.txt")
)

# Improved prompt — calibrated against Opus 4.6 gold standard (85% agreement)
IMPROVED_SYSTEM_PROMPT = None  # loaded from file at runtime


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=7200,
    memory=8192,
    secrets=[modal.Secret.from_name("anthropic-secret")],
)
def enrich_with_sbert():
    """Add SBERT embeddings projected onto constraint↔agency axis."""
    import json
    import numpy as np
    from pathlib import Path
    from sentence_transformers import SentenceTransformer

    print("Loading SBERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build axis from pole sentences
    constraint_sentences = [
        "slavery bondage oppression coercion tyranny servitude",
        "freedom from fear want discrimination persecution",
        "restraint restriction constraint interference domination",
        "removal of barriers and obstacles to freedom",
        "protection against arbitrary power and abuse",
    ]
    agency_sentences = [
        "autonomy choice capacity ability opportunity empowerment",
        "freedom to choose decide act speak worship",
        "self-determination individual development fulfillment",
        "enabling people to achieve their goals and potential",
        "providing resources and opportunities for human flourishing",
    ]

    print("Building SBERT axis...")
    c_vecs = model.encode(constraint_sentences)
    a_vecs = model.encode(agency_sentences)
    c_centroid = np.mean(c_vecs, axis=0)
    a_centroid = np.mean(a_vecs, axis=0)
    axis = a_centroid - c_centroid
    axis = axis / np.linalg.norm(axis)

    # Process each decade file
    data_dir = Path("/app/data")
    results = {}

    for fpath in sorted(data_dir.glob("sentences_*.json")):
        with open(fpath) as f:
            sentences = json.load(f)

        # Skip if already enriched
        if sentences and "sbert" in sentences[0].get("methods", {}):
            print(f"\n{fpath.name}: already has SBERT, copying as-is")
            results[fpath.name] = sentences
            continue

        print(f"\nProcessing {fpath.name} ({len(sentences)} sentences)...")
        texts = [s["sentence"] for s in sentences]

        # Batch encode
        embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)

        # Project onto axis
        projections = embeddings @ axis

        for i, s in enumerate(sentences):
            score = round(float(projections[i]), 4)
            s["methods"]["sbert"] = {
                "score": score,
                "label": "agency" if score > 0.05 else ("constraint" if score < -0.05 else "neutral"),
            }

        results[fpath.name] = sentences
        print(f"  {len(sentences)} sentences enriched")

    # Save to volume
    out_dir = Path("/vol/enriched")
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, data in results.items():
        with open(out_dir / fname, "w") as f:
            json.dump(data, f, separators=(",", ":"))
    volume.commit()
    print("\nSBERT enrichment saved to volume")
    return {fname: len(data) for fname, data in results.items()}


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=14400,
    memory=4096,
    secrets=[modal.Secret.from_name("anthropic-secret")],
)
def enrich_with_llm():
    """Add LLM few-shot classification using Claude Haiku."""
    import json
    import os
    import time
    from pathlib import Path
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Load improved prompt from file
    prompt_path = Path("/app/outputs/improved_haiku_prompt.txt")
    if prompt_path.exists():
        system_prompt = prompt_path.read_text()
        print("Using improved prompt (Opus-calibrated)")
    else:
        print("WARNING: improved prompt not found, using fallback")
        system_prompt = "Classify as positive_liberty, negative_liberty, ambiguous, or other."

    # Load SBERT-enriched data from volume
    vol_dir = Path("/vol/enriched")
    if not vol_dir.exists():
        print("ERROR: Run enrich_with_sbert first")
        return {}

    results = {}
    total_classified = 0

    for fpath in sorted(vol_dir.glob("sentences_*.json")):
        with open(fpath) as f:
            sentences = json.load(f)

        print(f"\nClassifying {fpath.name} ({len(sentences)} sentences)...")

        batch_size = 20
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            # Build batch prompt
            batch_text = "\n".join(
                f"{j+1}. \"{s['sentence']}\""
                for j, s in enumerate(batch)
            )

            try:
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=200,
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": f"Classify each sentence. Respond with one category per line, numbered to match.\n\n{batch_text}",
                    }],
                )

                # Parse response
                lines = resp.content[0].text.strip().split("\n")
                for j, line in enumerate(lines):
                    if j >= len(batch):
                        break
                    # Extract category from line like "1. negative_liberty" or just "negative_liberty"
                    cat = line.strip().lower()
                    for prefix in ["positive_liberty", "negative_liberty", "ambiguous", "other"]:
                        if prefix in cat:
                            cat = prefix
                            break
                    else:
                        cat = "ambiguous"  # fallback

                    batch[j]["methods"]["llm"] = {
                        "label": cat,
                        "model": "claude-haiku-4-5-20251001",
                    }

                total_classified += len(batch)

            except Exception as e:
                print(f"  Error at batch {i}: {e}")
                for s in batch:
                    s["methods"]["llm"] = {"label": "error", "model": "claude-haiku-4-5-20251001"}
                time.sleep(2)

            if total_classified % 1000 == 0 and total_classified > 0:
                print(f"  Classified {total_classified:,}...")

        results[fpath.name] = sentences

        # Save incrementally
        with open(fpath, "w") as f:
            json.dump(sentences, f, separators=(",", ":"))
        volume.commit()
        print(f"  {fpath.name}: {len(sentences)} classified, saved")

    return {fname: len(data) for fname, data in results.items()}


@app.function(image=image, volumes={"/vol": volume}, timeout=60)
def fetch_results():
    """Download enriched files from volume."""
    import json
    from pathlib import Path

    vol_dir = Path("/vol/enriched")
    if not vol_dir.exists():
        return {"error": "No enriched data found"}

    result = {}
    for fpath in sorted(vol_dir.glob("*.json")):
        with open(fpath) as f:
            result[fpath.name] = json.load(f)
    return result


@app.local_entrypoint()
def main(step: str = "all"):
    import json
    from pathlib import Path

    if step in ("all", "sbert"):
        print("=== Step 1: SBERT enrichment ===")
        sbert_result = enrich_with_sbert.remote()
        for fname, count in sbert_result.items():
            print(f"  {fname}: {count}")

    if step in ("all", "llm"):
        print("\n=== Step 2: LLM classification ===")
        llm_result = enrich_with_llm.remote()
        for fname, count in llm_result.items():
            print(f"  {fname}: {count}")

    if step in ("all", "fetch"):
        print("\n=== Fetching results ===")
        result = fetch_results.remote()
        if "error" in result:
            print(result["error"])
            return

        data_dir = Path("web/data")
        for fname, data in result.items():
            with open(data_dir / fname, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            print(f"  {fname}: {len(data)} sentences")

        # Update index
        idx_path = data_dir / "index.json"
        with open(idx_path) as f:
            index = json.load(f)
        index["methods"] = ["from_to", "domains", "pole_score", "freq", "sbert", "llm"]
        with open(idx_path, "w") as f:
            json.dump(index, f, indent=2)

        print("\nAll files updated in web/data/")
