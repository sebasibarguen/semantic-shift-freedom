# ABOUTME: Modal app that trains word2vec on Wikipedia dumps (2008, 2020, 2025).
# ABOUTME: Processes each dump sequentially to manage disk, commits after each.

import modal

app = modal.App("freedom-wiki-train")

volume = modal.Volume.from_name("wiki-embeddings-data", create_if_missing=True)

WIKI_DUMPS = {
    2008: "https://archive.org/download/enwiki-20080103/enwiki-20080103-pages-articles.xml.bz2",
    2020: "https://archive.org/download/enwiki-20200101/enwiki-20200101-pages-articles-multistream.xml.bz2",
    2025: "https://dumps.wikimedia.org/enwiki/20260301/enwiki-20260301-pages-articles-multistream.xml.bz2",
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "gensim", "scipy")
    .apt_install("wget", "unzip")
    # COHA for alignment
    .run_commands("mkdir -p /data/coha_sgns && wget -q http://snap.stanford.edu/historical_embeddings/coha-word_sgns.zip -O /tmp/coha.zip && unzip -o /tmp/coha.zip -d /tmp/coha_extract && find /tmp/coha_extract -name '*.pkl' -exec mv {} /data/coha_sgns/ \\; && find /tmp/coha_extract -name '*.npy' -exec mv {} /data/coha_sgns/ \\; && rm -rf /tmp/coha.zip /tmp/coha_extract")
    .add_local_dir("src", remote_path="/app/src")
)


@app.function(image=image, volumes={"/vol": volume}, timeout=36000, memory=32768, cpu=8, nonpreemptible=True)
def process_one_dump(year: int):
    """Download, extract, train, and save one Wikipedia year."""
    import subprocess
    import sys
    from pathlib import Path

    sys.path.insert(0, "/app/src")
    from wiki_train import extract_to_text, train_from_text, save_histwords_format

    wiki_dir = "/vol/wiki_sgns"
    Path(wiki_dir).mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if (Path(wiki_dir) / f"{year}-vocab.pkl").exists():
        print(f"[{year}] Already in volume, skipping")
        return {"year": year, "status": "cached"}

    url = WIKI_DUMPS[year]
    dump_path = f"/tmp/dump_{year}.xml.bz2"
    text_path = f"/tmp/wiki_{year}.txt"
    model_path = f"/tmp/w2v_{year}.model"

    # Download
    print(f"[{year}] Downloading {url}...")
    subprocess.run(
        ["wget", "-q", "--no-check-certificate", "--timeout=600", "--tries=3",
         "-O", dump_path, url],
        check=True,
    )
    size_gb = Path(dump_path).stat().st_size / (1024**3)
    print(f"[{year}] Downloaded ({size_gb:.1f} GB)")

    # Extract text
    print(f"[{year}] Extracting text...")
    extract_to_text(dump_path, text_path)
    Path(dump_path).unlink()  # Free disk

    text_size_gb = Path(text_path).stat().st_size / (1024**3)
    print(f"[{year}] Text file: {text_size_gb:.1f} GB")

    # Train
    print(f"[{year}] Training word2vec...")
    model = train_from_text(text_path, model_path)
    Path(text_path).unlink()  # Free disk

    # Save to volume
    print(f"[{year}] Saving to volume...")
    save_histwords_format(model, wiki_dir, year)

    # Cleanup model files
    for p in Path("/tmp").glob(f"w2v_{year}*"):
        p.unlink(missing_ok=True)

    volume.commit()
    print(f"[{year}] Done!")
    return {"year": year, "status": "trained"}


@app.function(image=image, volumes={"/vol": volume}, timeout=1800, memory=16384)
def align_and_analyze():
    """Procrustes-align wiki embeddings to COHA 2000s and run SemAxis analysis."""
    import sys
    import shutil
    from pathlib import Path

    sys.path.insert(0, "/app/src")
    from wiki_embeddings import procrustes_align

    coha_dir = "/data/coha_sgns"
    wiki_dir = "/vol/wiki_sgns"
    aligned_dir = "/vol/wiki_sgns_aligned"
    Path(aligned_dir).mkdir(parents=True, exist_ok=True)

    # Align each year to COHA 2000s
    print("Aligning Wikipedia embeddings to COHA 2000s...")
    for year in [2008, 2020, 2025]:
        src_vocab = f"{wiki_dir}/{year}-vocab.pkl"
        src_matrix = f"{wiki_dir}/{year}-w.npy"
        if not Path(src_vocab).exists():
            print(f"  Skipping {year} — not found")
            continue

        print(f"\n  Aligning {year}...")
        procrustes_align(
            f"{coha_dir}/2000-vocab.pkl", f"{coha_dir}/2000-w.npy",
            src_vocab, src_matrix,
            f"{aligned_dir}/{year}-w.npy",
        )
        shutil.copy2(src_vocab, f"{aligned_dir}/{year}-vocab.pkl")

    volume.commit()

    # Run analysis using the same function as GloVe but adapted for wiki labels
    from wiki_embeddings import run_glove_analysis
    from embeddings import TemporalEmbeddings
    from metrics import cosine_similarity
    from semantic_axis import (
        CONSTRAINT_SEEDS, AGENCY_SEEDS, CONTROL_WORDS,
        expand_pole, build_axis, project_onto_axis, linear_trend,
        EXPANSION_K,
    )
    import json
    import numpy as np

    results = {"coha": {}, "wiki": {}, "combined": {}}

    print("\nLoading COHA embeddings...")
    coha = TemporalEmbeddings(coha_dir)
    coha.load_decades(start=1830, end=2000, step=10)

    print("Loading Wikipedia embeddings...")
    wiki = TemporalEmbeddings(aligned_dir)
    for year in [2008, 2020, 2025]:
        try:
            wiki.load_decade(year)
        except FileNotFoundError:
            print(f"  Skipping {year}")
    wiki_labels = wiki.decades
    print(f"Wiki snapshots: {wiki_labels}")

    # Build axis from COHA 2000s
    ref = 2000
    c_exp = expand_pole(coha, CONSTRAINT_SEEDS, ref, EXPANSION_K)
    a_exp = expand_pole(coha, AGENCY_SEEDS, ref, EXPANSION_K)
    overlap = set(c_exp) & set(a_exp)
    c_exp = [w for w in c_exp if w not in overlap]
    a_exp = [w for w in a_exp if w not in overlap]
    axis = build_axis(coha, c_exp, a_exp, ref)

    if axis is None:
        print("ERROR: Could not build axis")
        return results

    all_words = ["freedom"] + [w for w in CONTROL_WORDS if coha.word_exists(w, ref)]

    # Project COHA
    coha_proj = {}
    for word in all_words:
        wp = {}
        for decade in coha.decades:
            v = coha.get_vector(word, decade)
            if v is not None:
                wp[str(decade)] = round(float(np.dot(v, axis)), 4)
        coha_proj[word] = wp
    results["coha"]["projections"] = coha_proj

    # Project Wiki
    wiki_proj = {}
    for word in all_words:
        wp = {}
        for label in wiki_labels:
            v = wiki.get_vector(word, label)
            if v is not None:
                wp[str(label)] = round(float(np.dot(v, axis)), 4)
        wiki_proj[word] = wp
    results["wiki"]["projections"] = wiki_proj

    # Combined timeline
    print()
    print("=" * 70)
    print("COMBINED TIMELINE: Freedom on constraint→agency axis")
    print("(COHA 1830-2000, Wikipedia word2vec 2008/2020/2025)")
    print("=" * 70)
    combined = {}
    for d, v in coha_proj.get("freedom", {}).items():
        combined[d] = {"value": v, "source": "COHA"}
    for d, v in wiki_proj.get("freedom", {}).items():
        combined[d] = {"value": v, "source": "Wikipedia"}

    print(f"\n  {'Year':<10} {'Projection':>12} {'Source':<12}")
    print(f"  {'-'*34}")
    for d in sorted(combined.keys()):
        e = combined[d]
        print(f"  {d:<10} {e['value']:>+12.4f} {e['source']:<12}")
    results["combined"]["freedom_timeline"] = combined

    # Freedom-liberty similarity
    print()
    print("FREEDOM-LIBERTY SIMILARITY:")
    fl = {}
    for label in wiki_labels:
        vf = wiki.get_vector("freedom", label)
        vl = wiki.get_vector("liberty", label)
        if vf is not None and vl is not None:
            s = float(cosine_similarity(vf, vl))
            fl[str(label)] = round(s, 4)
            print(f"  Wiki {label}: {s:.4f}")
    results["wiki"]["freedom_liberty_similarity"] = fl

    # Trends
    print()
    coha_f = coha_proj.get("freedom", {})
    cd = [int(d) for d in sorted(coha_f.keys())]
    cv = [coha_f[str(d)] for d in cd]
    ct = linear_trend(cd, cv)
    if ct:
        print(f"  COHA trend: {ct['slope_per_century']:+.4f}/century (R²={ct['r_squared']:.3f})")
        results["coha"]["trend"] = ct

    ad = {}
    for d, v in coha_f.items():
        ad[int(d)] = v
    for d, v in wiki_proj.get("freedom", {}).items():
        ad[int(d)] = v
    cd2 = sorted(ad.keys())
    cv2 = [ad[d] for d in cd2]
    ct2 = linear_trend(cd2, cv2)
    if ct2:
        print(f"  Combined trend: {ct2['slope_per_century']:+.4f}/century (R²={ct2['r_squared']:.3f})")
        results["combined"]["trend"] = ct2

    # Save
    out_path = "/vol/wiki_w2v_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nResults saved to {out_path}")
    return results


@app.function(image=image, volumes={"/vol": volume}, timeout=36000, memory=32768, cpu=8, nonpreemptible=True)
def run_all():
    """Orchestrator: process all dumps sequentially, then align and analyze.
    Runs entirely server-side — no local dependency."""
    import subprocess
    import sys
    import shutil
    import json
    import numpy as np
    from pathlib import Path

    sys.path.insert(0, "/app/src")
    from wiki_train import extract_to_text, train_from_text, save_histwords_format
    from wiki_embeddings import procrustes_align
    from embeddings import TemporalEmbeddings
    from metrics import cosine_similarity
    from semantic_axis import (
        CONSTRAINT_SEEDS, AGENCY_SEEDS, CONTROL_WORDS,
        expand_pole, build_axis, linear_trend, EXPANSION_K,
    )

    coha_dir = "/data/coha_sgns"
    wiki_dir = "/vol/wiki_sgns"
    aligned_dir = "/vol/wiki_sgns_aligned"
    Path(wiki_dir).mkdir(parents=True, exist_ok=True)
    Path(aligned_dir).mkdir(parents=True, exist_ok=True)

    # === PHASE 1: Download, extract, train each dump ===
    for year, url in sorted(WIKI_DUMPS.items()):
        if (Path(wiki_dir) / f"{year}-vocab.pkl").exists():
            print(f"[{year}] Already in volume, skipping")
            continue

        dump_path = f"/tmp/dump_{year}.xml.bz2"
        text_path = f"/tmp/wiki_{year}.txt"
        model_path = f"/tmp/w2v_{year}.model"

        print(f"\n{'='*70}")
        print(f"[{year}] Downloading {url}...")
        subprocess.run(
            ["wget", "-q", "--no-check-certificate", "--timeout=600", "--tries=3",
             "-O", dump_path, url],
            check=True,
        )
        size_gb = Path(dump_path).stat().st_size / (1024**3)
        print(f"[{year}] Downloaded ({size_gb:.1f} GB)")

        print(f"[{year}] Extracting text...")
        extract_to_text(dump_path, text_path)
        Path(dump_path).unlink()

        print(f"[{year}] Training word2vec...")
        model = train_from_text(text_path, model_path)
        Path(text_path).unlink()

        print(f"[{year}] Saving to volume...")
        save_histwords_format(model, wiki_dir, year)
        for p in Path("/tmp").glob(f"w2v_{year}*"):
            p.unlink(missing_ok=True)

        volume.commit()
        print(f"[{year}] Done!")

    # === PHASE 2: Procrustes alignment ===
    print(f"\n{'='*70}")
    print("Aligning Wikipedia embeddings to COHA 2000s...")
    for year in [2008, 2020, 2025]:
        src_vocab = f"{wiki_dir}/{year}-vocab.pkl"
        if not Path(src_vocab).exists():
            print(f"  Skipping {year} — not found")
            continue
        print(f"  Aligning {year}...")
        procrustes_align(
            f"{coha_dir}/2000-vocab.pkl", f"{coha_dir}/2000-w.npy",
            src_vocab, f"{wiki_dir}/{year}-w.npy",
            f"{aligned_dir}/{year}-w.npy",
        )
        shutil.copy2(src_vocab, f"{aligned_dir}/{year}-vocab.pkl")
    volume.commit()

    # === PHASE 3: Analysis ===
    print(f"\n{'='*70}")
    print("Running SemAxis analysis...")

    coha = TemporalEmbeddings(coha_dir)
    coha.load_decades(start=1830, end=2000, step=10)

    wiki = TemporalEmbeddings(aligned_dir)
    for year in [2008, 2020, 2025]:
        try:
            wiki.load_decade(year)
        except FileNotFoundError:
            pass
    wiki_labels = wiki.decades
    print(f"Wiki snapshots: {wiki_labels}")

    ref = 2000
    c_exp = expand_pole(coha, CONSTRAINT_SEEDS, ref, EXPANSION_K)
    a_exp = expand_pole(coha, AGENCY_SEEDS, ref, EXPANSION_K)
    overlap = set(c_exp) & set(a_exp)
    c_exp = [w for w in c_exp if w not in overlap]
    a_exp = [w for w in a_exp if w not in overlap]
    axis = build_axis(coha, c_exp, a_exp, ref)

    results = {"coha": {}, "wiki": {}, "combined": {}}
    if axis is None:
        print("ERROR: Could not build axis")
        return results

    all_words = ["freedom"] + [w for w in CONTROL_WORDS if coha.word_exists(w, ref)]

    coha_proj = {}
    for word in all_words:
        wp = {}
        for decade in coha.decades:
            v = coha.get_vector(word, decade)
            if v is not None:
                wp[str(decade)] = round(float(np.dot(v, axis)), 4)
        coha_proj[word] = wp
    results["coha"]["projections"] = coha_proj

    wiki_proj = {}
    for word in all_words:
        wp = {}
        for label in wiki_labels:
            v = wiki.get_vector(word, label)
            if v is not None:
                wp[str(label)] = round(float(np.dot(v, axis)), 4)
        wiki_proj[word] = wp
    results["wiki"]["projections"] = wiki_proj

    combined = {}
    for d, v in coha_proj.get("freedom", {}).items():
        combined[d] = {"value": v, "source": "COHA"}
    for d, v in wiki_proj.get("freedom", {}).items():
        combined[d] = {"value": v, "source": "Wikipedia"}
    results["combined"]["freedom_timeline"] = combined

    print(f"\n  {'Year':<10} {'Projection':>12} {'Source':<12}")
    print(f"  {'-'*34}")
    for d in sorted(combined.keys()):
        e = combined[d]
        print(f"  {d:<10} {e['value']:>+12.4f} {e['source']:<12}")

    fl = {}
    for label in wiki_labels:
        vf = wiki.get_vector("freedom", label)
        vl = wiki.get_vector("liberty", label)
        if vf is not None and vl is not None:
            s = float(cosine_similarity(vf, vl))
            fl[str(label)] = round(s, 4)
            print(f"\n  Freedom-liberty similarity Wiki {label}: {s:.4f}")
    results["wiki"]["freedom_liberty_similarity"] = fl

    coha_f = coha_proj.get("freedom", {})
    cd = [int(d) for d in sorted(coha_f.keys())]
    cv = [coha_f[str(d)] for d in cd]
    ct = linear_trend(cd, cv)
    if ct:
        print(f"\n  COHA trend: {ct['slope_per_century']:+.4f}/century")
        results["coha"]["trend"] = ct

    ad = {int(d): v for d, v in coha_f.items()}
    for d, v in wiki_proj.get("freedom", {}).items():
        ad[int(d)] = v
    cd2 = sorted(ad.keys())
    cv2 = [ad[d] for d in cd2]
    ct2 = linear_trend(cd2, cv2)
    if ct2:
        print(f"  Combined trend: {ct2['slope_per_century']:+.4f}/century")
        results["combined"]["trend"] = ct2

    with open("/vol/wiki_w2v_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print("\nDone! Results saved to volume.")
    return results


@app.local_entrypoint()
def main(fetch_only: bool = False):
    import json
    from pathlib import Path

    if fetch_only:
        # Just read results from volume
        from pathlib import Path as P
        results = align_and_analyze.remote()
    else:
        # Single server-side call — survives local disconnects with --detach
        results = run_all.remote()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    out = output_dir / "wiki_w2v_embeddings.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved locally to: {out}")
