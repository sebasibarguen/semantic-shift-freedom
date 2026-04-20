# ABOUTME: Analyzes freedom discourse in EEBO-TCP corpus (1500-1700).
# ABOUTME: Extracts contexts, computes collocates, and tags by semantic domain.

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from normalizer import EarlyModernNormalizer
from domain_tagger import DomainTagger


# Freedom variants to search for (pre-normalization)
FREEDOM_VARIANTS = [
    'freedom', 'freedome', 'fredom', 'freedoom', 'fredome',
    'liberty', 'libertie', 'libertye', 'lybertie', 'lyberty',
    'free', 'fre',  # as in "free man", "free subject"
]

# Compiled pattern
FREEDOM_PATTERN = re.compile(
    r'\b(' + '|'.join(FREEDOM_VARIANTS) + r')\b',
    re.IGNORECASE
)


def load_bin_corpus(corpus_dir: Path, bin_label: str) -> list[dict]:
    """Load corpus for a specific time bin."""
    json_path = corpus_dir / f"eebo_{bin_label}.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def extract_contexts(text: str, window: int = 10) -> list[dict]:
    """
    Extract contexts around freedom/liberty mentions.
    Returns list of {match, left_context, right_context, full_context}.
    """
    contexts = []
    words = text.split()

    for i, word in enumerate(words):
        if FREEDOM_PATTERN.match(word):
            left = words[max(0, i - window):i]
            right = words[i + 1:i + 1 + window]

            contexts.append({
                'match': word,
                'position': i,
                'left_context': ' '.join(left),
                'right_context': ' '.join(right),
                'full_context': ' '.join(left + [word] + right),
            })

    return contexts


def compute_collocates(contexts: list[dict], normalizer: EarlyModernNormalizer) -> Counter:
    """Compute collocate frequencies from contexts."""
    collocates = Counter()

    for ctx in contexts:
        # Combine left and right context
        context_text = f"{ctx['left_context']} {ctx['right_context']}"

        # Normalize
        normalized = normalizer.normalize(context_text)

        # Tokenize and clean
        words = re.findall(r'\b[a-z]+\b', normalized.lower())

        # Filter stopwords and very short words
        stopwords = {
            'the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'it', 'for',
            'as', 'be', 'by', 'with', 'which', 'or', 'from', 'this', 'at',
            'an', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'but',
            'not', 'they', 'their', 'them', 'he', 'his', 'him', 'she', 'her',
            'we', 'our', 'us', 'you', 'your', 'who', 'whom', 'what', 'when',
            'where', 'there', 'then', 'than', 'so', 'if', 'no', 'all', 'any',
            'some', 'such', 'more', 'most', 'other', 'only', 'can', 'may',
            'must', 'should', 'would', 'could', 'shall', 'will', 'do', 'did',
            'does', 'being', 'upon', 'into', 'own', 'same', 'also', 'how',
            'those', 'these', 'one', 'two', 'first', 'after', 'before',
        }

        for word in words:
            if len(word) > 2 and word not in stopwords:
                collocates[word] += 1

    return collocates


def analyze_bin(corpus_dir: Path, bin_label: str, normalizer: EarlyModernNormalizer,
                tagger: DomainTagger) -> dict:
    """Analyze a single time bin."""
    print(f"\nAnalyzing {bin_label}...")

    texts = load_bin_corpus(corpus_dir, bin_label)
    if not texts:
        return {'error': f'No texts found for {bin_label}'}

    all_contexts = []
    doc_count = 0
    total_words = 0

    for doc in texts:
        text = doc['text']
        total_words += doc['word_count']

        # Normalize before extracting contexts
        normalized_text = normalizer.normalize(text)

        contexts = extract_contexts(normalized_text)
        if contexts:
            doc_count += 1
            for ctx in contexts:
                ctx['tcp_id'] = doc['tcp_id']
                ctx['year'] = doc['year']
                ctx['title'] = doc['title'][:100]
            all_contexts.extend(contexts)

    print(f"  Texts: {len(texts)}, with freedom mentions: {doc_count}")
    print(f"  Total contexts: {len(all_contexts)}")

    # Compute collocates
    collocates = compute_collocates(all_contexts, normalizer)

    # Get top collocates
    top_collocates = collocates.most_common(100)

    # Tag collocates by domain
    tagged_collocates = []
    for word, count in top_collocates:
        domains = tagger.tag(word)
        tagged_collocates.append({
            'word': word,
            'count': count,
            'domains': domains,
            'primary_domain': domains[0],
        })

    # Domain distribution
    domain_dist = defaultdict(int)
    for item in tagged_collocates[:50]:  # Top 50
        domain_dist[item['primary_domain']] += 1

    # Sample contexts
    sample_contexts = all_contexts[:20] if all_contexts else []

    return {
        'bin': bin_label,
        'doc_count': len(texts),
        'docs_with_freedom': doc_count,
        'total_words': total_words,
        'context_count': len(all_contexts),
        'top_collocates': tagged_collocates,
        'domain_distribution': dict(domain_dist),
        'sample_contexts': sample_contexts,
    }


def main():
    project_root = Path(__file__).parent.parent
    corpus_dir = project_root / "data" / "eebo" / "corpus"
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    normalizer = EarlyModernNormalizer()
    tagger = DomainTagger()

    # Time bins for Tier 2
    bins = ['1500-1550', '1550-1600', '1600-1650', '1650-1700']

    results = {}
    for bin_label in bins:
        results[bin_label] = analyze_bin(corpus_dir, bin_label, normalizer, tagger)

    # Save results
    output_path = output_dir / "tier2_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved analysis to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TIER 2 ANALYSIS SUMMARY")
    print("=" * 60)

    for bin_label in bins:
        r = results[bin_label]
        if 'error' in r:
            print(f"\n{bin_label}: {r['error']}")
            continue

        print(f"\n{bin_label}:")
        print(f"  Documents: {r['doc_count']} ({r['docs_with_freedom']} with freedom)")
        print(f"  Words: {r['total_words']:,}")
        print(f"  Freedom contexts: {r['context_count']}")

        print(f"  Top collocates: ", end='')
        top_words = [c['word'] for c in r['top_collocates'][:10]]
        print(', '.join(top_words))

        print(f"  Domain distribution (top 50):")
        for domain, count in sorted(r['domain_distribution'].items(), key=lambda x: -x[1]):
            pct = count / 50 * 100
            print(f"    {domain}: {count} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
