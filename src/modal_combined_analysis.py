# ABOUTME: Runs the final combined analysis across COHA + Wiki SGNS + GloVe.
# ABOUTME: Aligns Wiki 2008 to COHA, then projects everything onto the SemAxis.

import modal

app = modal.App("freedom-combined-analysis")

volume = modal.Volume.from_name("wiki-embeddings-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy")
    .apt_install("wget", "unzip")
    .run_commands("mkdir -p /data/coha_sgns && wget -q http://snap.stanford.edu/historical_embeddings/coha-word_sgns.zip -O /tmp/coha.zip && unzip -o /tmp/coha.zip -d /tmp/coha_extract && find /tmp/coha_extract -name '*.pkl' -exec mv {} /data/coha_sgns/ \\; && find /tmp/coha_extract -name '*.npy' -exec mv {} /data/coha_sgns/ \\; && rm -rf /tmp/coha.zip /tmp/coha_extract")
    .add_local_dir("src", remote_path="/app/src")
)


@app.function(image=image, volumes={"/vol": volume}, timeout=1800, memory=16384)
def run_combined():
    import sys
    import json
    import shutil
    import numpy as np
    from pathlib import Path

    sys.path.insert(0, "/app/src")
    from wiki_embeddings import procrustes_align
    from embeddings import TemporalEmbeddings
    from metrics import cosine_similarity
    from semantic_axis import (
        CONSTRAINT_SEEDS, AGENCY_SEEDS, CONTROL_WORDS,
        expand_pole, build_axis, linear_trend, EXPANSION_K,
    )

    coha_dir = "/data/coha_sgns"
    wiki_dir = "/vol/wiki_sgns"
    glove_dir = "/vol/glove_sgns_aligned"
    aligned_dir = "/vol/wiki_sgns_aligned"
    Path(aligned_dir).mkdir(parents=True, exist_ok=True)

    # Align Wiki 2008 SGNS to COHA 2000s
    print("Aligning Wiki 2008 SGNS to COHA 2000s...")
    procrustes_align(
        f"{coha_dir}/2000-vocab.pkl", f"{coha_dir}/2000-w.npy",
        f"{wiki_dir}/2008-vocab.pkl", f"{wiki_dir}/2008-w.npy",
        f"{aligned_dir}/2008-w.npy",
    )
    shutil.copy2(f"{wiki_dir}/2008-vocab.pkl", f"{aligned_dir}/2008-vocab.pkl")
    volume.commit()

    # Load COHA
    print("\nLoading COHA...")
    coha = TemporalEmbeddings(coha_dir)
    coha.load_decades(start=1830, end=2000, step=10)

    # Load aligned Wiki 2008
    print("Loading Wiki 2008 (aligned)...")
    wiki = TemporalEmbeddings(aligned_dir)
    wiki.load_decade(2008)

    # Load aligned GloVe 2014 + 2024
    print("Loading GloVe 2014 + 2024 (aligned)...")
    glove = TemporalEmbeddings(glove_dir)
    glove.load_decade(2014)
    glove.load_decade(2024)

    # Build axis from COHA 2000s
    ref = 2000
    c_exp = expand_pole(coha, CONSTRAINT_SEEDS, ref, EXPANSION_K)
    a_exp = expand_pole(coha, AGENCY_SEEDS, ref, EXPANSION_K)
    overlap = set(c_exp) & set(a_exp)
    c_exp = [w for w in c_exp if w not in overlap]
    a_exp = [w for w in a_exp if w not in overlap]
    axis = build_axis(coha, c_exp, a_exp, ref)

    if axis is None:
        return {"error": "Could not build axis"}

    all_words = ["freedom"] + [w for w in CONTROL_WORDS if coha.word_exists(w, ref)]

    results = {
        "axis_source": "COHA 2000s",
        "axis_words": {"constraint": len(c_exp), "agency": len(a_exp)},
        "coha": {},
        "wiki_sgns": {},
        "glove": {},
        "combined": {},
    }

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

    # Project Wiki 2008
    wiki_proj = {}
    for word in all_words:
        v = wiki.get_vector(word, 2008)
        if v is not None:
            wiki_proj[word] = {"2008": round(float(np.dot(v, axis)), 4)}
    results["wiki_sgns"]["projections"] = wiki_proj

    # Project GloVe 2014 + 2024
    glove_proj = {}
    for word in all_words:
        wp = {}
        for label in [2014, 2024]:
            v = glove.get_vector(word, label)
            if v is not None:
                wp[str(label)] = round(float(np.dot(v, axis)), 4)
        glove_proj[word] = wp
    results["glove"]["projections"] = glove_proj

    # Combined timeline for freedom
    combined = {}
    for d, v in coha_proj.get("freedom", {}).items():
        combined[d] = {"value": v, "source": "COHA SGNS", "method": "SGNS"}
    if "freedom" in wiki_proj:
        for d, v in wiki_proj["freedom"].items():
            combined[d] = {"value": v, "source": "Wikipedia SGNS", "method": "SGNS"}
    for d, v in glove_proj.get("freedom", {}).items():
        combined[d] = {"value": v, "source": "GloVe (Wikipedia)", "method": "GloVe"}

    results["combined"]["freedom_timeline"] = combined

    # Print timeline
    print()
    print("=" * 70)
    print("COMBINED TIMELINE: Freedom on constraint→agency axis")
    print("=" * 70)
    print()
    print(f"  {'Year':<8} {'Projection':>12} {'Source':<25} {'Method':<8}")
    print(f"  {'-'*55}")
    for d in sorted(combined.keys()):
        e = combined[d]
        print(f"  {d:<8} {e['value']:>+12.4f} {e['source']:<25} {e['method']:<8}")

    # Freedom-liberty similarity
    print()
    print("=" * 70)
    print("FREEDOM-LIBERTY SIMILARITY")
    print("=" * 70)
    print()
    fl = {}
    # COHA
    for decade in coha.decades:
        vf = coha.get_vector("freedom", decade)
        vl = coha.get_vector("liberty", decade)
        if vf is not None and vl is not None:
            fl[str(decade)] = {"sim": round(float(cosine_similarity(vf, vl)), 4), "source": "COHA"}
    # Wiki 2008
    vf = wiki.get_vector("freedom", 2008)
    vl = wiki.get_vector("liberty", 2008)
    if vf is not None and vl is not None:
        fl["2008"] = {"sim": round(float(cosine_similarity(vf, vl)), 4), "source": "Wiki SGNS"}
    # GloVe
    for label in [2014, 2024]:
        vf = glove.get_vector("freedom", label)
        vl = glove.get_vector("liberty", label)
        if vf is not None and vl is not None:
            fl[str(label)] = {"sim": round(float(cosine_similarity(vf, vl)), 4), "source": "GloVe"}

    results["combined"]["freedom_liberty_similarity"] = fl
    for d in sorted(fl.keys()):
        print(f"  {d}: {fl[d]['sim']:.4f} ({fl[d]['source']})")

    # Trends
    print()
    print("=" * 70)
    print("TREND ANALYSIS")
    print("=" * 70)
    print()

    # COHA-only
    cf = coha_proj.get("freedom", {})
    cd = [int(d) for d in sorted(cf.keys())]
    cv = [cf[str(d)] for d in cd]
    ct = linear_trend(cd, cv)
    if ct:
        print(f"  COHA only (1830-2000):     {ct['slope_per_century']:+.4f}/century (R²={ct['r_squared']:.3f})")
        results["coha"]["trend"] = ct

    # COHA + Wiki SGNS (same method)
    sgns_all = dict(cf)
    if "freedom" in wiki_proj:
        sgns_all.update(wiki_proj["freedom"])
    sd = [int(d) for d in sorted(sgns_all.keys())]
    sv = [sgns_all[str(d)] for d in sd]
    st = linear_trend(sd, sv)
    if st:
        print(f"  SGNS series (1830-2008):   {st['slope_per_century']:+.4f}/century (R²={st['r_squared']:.3f})")
        results["combined"]["sgns_trend"] = st

    # Full combined (all sources)
    all_d = {}
    for d, v in cf.items():
        all_d[int(d)] = v
    if "freedom" in wiki_proj:
        for d, v in wiki_proj["freedom"].items():
            all_d[int(d)] = v
    for d, v in glove_proj.get("freedom", {}).items():
        all_d[int(d)] = v
    fd = sorted(all_d.keys())
    fv = [all_d[d] for d in fd]
    ft = linear_trend(fd, fv)
    if ft:
        print(f"  All sources (1830-2024):   {ft['slope_per_century']:+.4f}/century (R²={ft['r_squared']:.3f})")
        results["combined"]["full_trend"] = ft

    # Control words comparison
    print()
    print(f"  {'Word':<18} {'COHA':>10} {'SGNS+Wiki':>12} {'All sources':>14}")
    print(f"  {'-'*56}")
    for word in all_words:
        # COHA only
        wc = coha_proj.get(word, {})
        wcd = [int(d) for d in sorted(wc.keys())]
        wcv = [wc[str(d)] for d in wcd]
        wct = linear_trend(wcd, wcv)

        # SGNS series
        ws = dict(wc)
        if word in wiki_proj:
            ws.update(wiki_proj[word])
        wsd = [int(d) for d in sorted(ws.keys())]
        wsv = [ws[str(d)] for d in wsd]
        wst = linear_trend(wsd, wsv)

        # All
        wa = dict(ws)
        if word in glove_proj:
            wa.update(glove_proj[word])
        wad = [int(d) for d in sorted(wa.keys())]
        wav = [wa[str(d)] for d in wad]
        wat = linear_trend(wad, wav)

        c = f"{wct['slope_per_century']:+.4f}" if wct else "N/A"
        s = f"{wst['slope_per_century']:+.4f}" if wst else "N/A"
        a = f"{wat['slope_per_century']:+.4f}" if wat else "N/A"
        m = " <--" if word == "freedom" else ""
        print(f"  {word:<18} {c:>10} {s:>12} {a:>14}{m}")

    # Save
    out_path = "/vol/combined_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nResults saved to volume: {out_path}")
    return results


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    results = run_combined.remote()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    out = output_dir / "combined_modern_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved locally to: {out}")
