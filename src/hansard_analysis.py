# ABOUTME: Analyzes freedom/liberty usage in UK Parliamentary debates (Hansard).
# ABOUTME: Tracks collocates, contexts, and frequency over time in political speech.

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_hansard(csv_path: str = "/data/hansard-speeches-v310.csv"):
    """Load Hansard speeches from local CSV."""
    import pandas as pd
    print(f"Loading Hansard from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False, usecols=lambda c: c in [
        "speech_date", "date", "speech", "text", "display_as", "speaker",
        "party", "mnis_id",
    ])
    print(f"  Loaded {len(df):,} speeches")
    print(f"  Columns: {list(df.columns)}")

    # Normalize columns
    if "speech_date" in df.columns:
        df["year"] = df["speech_date"].astype(str).str[:4].astype(int, errors="ignore")
    elif "date" in df.columns:
        df["year"] = df["date"].astype(str).str[:4].astype(int, errors="ignore")

    text_col = "speech" if "speech" in df.columns else "text"
    df["text_lower"] = df[text_col].fillna("").str.lower()
    df = df[df["text_lower"].str.len() > 10].copy()

    print(f"  Year range: {df['year'].min()} - {df['year'].max()}")
    return df


def analyze_frequency(df):
    """Track frequency of freedom/liberty mentions per year."""
    print("\n=== FREQUENCY ANALYSIS ===")

    df["has_freedom"] = df["text_lower"].str.contains("freedom", na=False)
    df["has_liberty"] = df["text_lower"].str.contains("liberty", na=False)

    yearly = df.groupby("year").agg(
        total=("text_lower", "count"),
        freedom=("has_freedom", "sum"),
        liberty=("has_liberty", "sum"),
    ).reset_index()

    yearly = yearly[yearly["total"] >= 100]
    results = {}

    print(f"\n  {'Year':<8} {'Total':>8} {'Freedom':>10} {'Liberty':>10} {'F/1000':>8} {'L/1000':>8}")
    print(f"  {'-'*56}")
    for _, row in yearly.iterrows():
        y = int(row["year"])
        results[y] = {
            "total_speeches": int(row["total"]),
            "freedom_count": int(row["freedom"]),
            "liberty_count": int(row["liberty"]),
            "freedom_rate": round(row["freedom"] / row["total"] * 1000, 2),
            "liberty_rate": round(row["liberty"] / row["total"] * 1000, 2),
        }
        r = results[y]
        print(f"  {y:<8} {r['total_speeches']:>8,} {r['freedom_count']:>10,} {r['liberty_count']:>10,} {r['freedom_rate']:>8.1f} {r['liberty_rate']:>8.1f}")

    return results


def analyze_collocates(df, window=10):
    """Find words that frequently appear near freedom/liberty."""
    print("\n=== COLLOCATE ANALYSIS ===")
    STOPWORDS = {
        "the", "of", "and", "to", "a", "in", "is", "it", "that", "for",
        "was", "on", "are", "be", "has", "have", "had", "with", "this",
        "not", "but", "by", "from", "or", "an", "they", "which", "as",
        "at", "we", "he", "she", "his", "her", "its", "their", "been",
        "will", "would", "could", "should", "may", "can", "do", "did",
        "does", "if", "my", "our", "your", "all", "no", "so", "there",
        "than", "very", "who", "what", "when", "where", "how", "i",
        "me", "him", "them", "us", "you", "about", "up", "out", "into",
        "just", "also", "more", "some", "any", "other", "these", "those",
        "one", "two", "only", "such", "then", "most", "over", "after",
        "before", "now", "here", "many", "must", "well", "much", "own",
        "even", "being", "made", "make", "going", "said", "hon", "right",
        "member", "members", "house", "government", "minister", "shall",
        "think", "people", "time", "way", "say", "like", "new", "see",
        "take", "come", "get", "give", "know", "first", "year", "years",
        "good", "great", "case", "point", "order", "bill", "clause",
    }
    TOKENIZE = re.compile(r"[a-z]+")

    freedom_collocates = defaultdict(Counter)
    liberty_collocates = defaultdict(Counter)

    relevant = df[df["has_freedom"] | df["has_liberty"]]
    for _, row in relevant.iterrows():
        decade = (int(row["year"]) // 10) * 10
        words = TOKENIZE.findall(row["text_lower"])

        for i, word in enumerate(words):
            if word == "freedom":
                context = words[max(0, i - window):i] + words[i + 1:i + window + 1]
                for w in context:
                    if w not in STOPWORDS and w != "freedom" and len(w) > 2:
                        freedom_collocates[decade][w] += 1
            elif word == "liberty":
                context = words[max(0, i - window):i] + words[i + 1:i + window + 1]
                for w in context:
                    if w not in STOPWORDS and w != "liberty" and len(w) > 2:
                        liberty_collocates[decade][w] += 1

    results = {"freedom": {}, "liberty": {}}
    for decade in sorted(freedom_collocates.keys()):
        results["freedom"][str(decade)] = dict(freedom_collocates[decade].most_common(20))
        results["liberty"][str(decade)] = dict(liberty_collocates[decade].most_common(20))

        print(f"\n  {decade}s Freedom: {', '.join(w for w, _ in freedom_collocates[decade].most_common(10))}")
        print(f"  {decade}s Liberty:  {', '.join(w for w, _ in liberty_collocates[decade].most_common(10))}")

    return results


def extract_contexts(df, max_per_decade=5):
    """Extract sample sentences containing freedom/liberty for qualitative review."""
    print("\n=== CONTEXT EXTRACTION ===")
    FREEDOM_RE = re.compile(r"[^.]*\bfreedom\b[^.]*\.", re.IGNORECASE)
    LIBERTY_RE = re.compile(r"[^.]*\bliberty\b[^.]*\.", re.IGNORECASE)

    freedom_contexts = defaultdict(list)
    liberty_contexts = defaultdict(list)

    speaker_col = "display_as" if "display_as" in df.columns else "speaker"

    relevant = df[df["has_freedom"] | df["has_liberty"]]
    for _, row in relevant.iterrows():
        decade = (int(row["year"]) // 10) * 10
        text = str(row.get("speech") or row.get("text") or "")
        speaker = str(row.get(speaker_col, "Unknown"))

        if len(freedom_contexts[decade]) < max_per_decade and "freedom" in text.lower():
            for m in FREEDOM_RE.finditer(text):
                sent = m.group(0).strip()
                if 20 < len(sent) < 300:
                    freedom_contexts[decade].append({
                        "sentence": sent,
                        "year": int(row["year"]),
                        "speaker": speaker,
                    })
                    break

        if len(liberty_contexts[decade]) < max_per_decade and "liberty" in text.lower():
            for m in LIBERTY_RE.finditer(text):
                sent = m.group(0).strip()
                if 20 < len(sent) < 300:
                    liberty_contexts[decade].append({
                        "sentence": sent,
                        "year": int(row["year"]),
                        "speaker": speaker,
                    })
                    break

        # Early exit if we have enough
        all_done = all(
            len(freedom_contexts[d]) >= max_per_decade and len(liberty_contexts[d]) >= max_per_decade
            for d in range(1970, 2030, 10)
        )
        if all_done:
            break

    results = {"freedom": {}, "liberty": {}}
    for decade in sorted(freedom_contexts.keys()):
        results["freedom"][str(decade)] = freedom_contexts[decade]
        results["liberty"][str(decade)] = liberty_contexts[decade]
        print(f"\n  {decade}s examples:")
        for ctx in freedom_contexts[decade][:2]:
            print(f"    F: \"{ctx['sentence'][:100]}...\" — {ctx['speaker']} ({ctx['year']})")
        for ctx in liberty_contexts[decade][:2]:
            print(f"    L: \"{ctx['sentence'][:100]}...\" — {ctx['speaker']} ({ctx['year']})")

    return results


def run_hansard_analysis(output_path: str = "outputs/hansard_analysis.json"):
    """Run the full Hansard analysis pipeline."""
    df = load_hansard()

    results = {
        "dataset": "Hansard speeches v3.1.0 (Evan Odell)",
        "period": f"{df['year'].min()}-{df['year'].max()}",
        "total_speeches": len(df),
    }

    results["frequency"] = analyze_frequency(df)
    results["collocates"] = analyze_collocates(df)
    results["contexts"] = extract_contexts(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    run_hansard_analysis()
