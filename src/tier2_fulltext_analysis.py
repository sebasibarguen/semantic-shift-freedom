# ABOUTME: Analyzes freedom collocates from full-text search results.
# ABOUTME: Works with output from eebo_fulltext_search.py.

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from .normalizer import EarlyModernNormalizer
from .domain_tagger import DomainTagger


def load_contexts(corpus_dir: Path, bin_label: str) -> list[dict]:
    """Load contexts for a specific time bin."""
    json_path = corpus_dir / f"freedom_contexts_{bin_label}.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def compute_collocates(texts: list[dict], normalizer: EarlyModernNormalizer) -> Counter:
    """Compute collocate frequencies from all contexts."""
    collocates = Counter()

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
        'now', 'even', 'much', 'both', 'out', 'yet', 'nor', 'hath',
        'doth', 'thee', 'thou', 'thy', 'ye', 'unto', 'wherein', 'thereof',
        'thereby', 'whereof', 'whereby', 'hereof', 'hee', 'shee', 'wee',
        'bee', 'mee', 'himselfe', 'themselves', 'selfe', 'owne', 'vpon',
    }

    for text in texts:
        for ctx in text.get('contexts', []):
            # Combine left and right context
            context_text = f"{ctx['left_context']} {ctx['right_context']}"

            # Normalize spelling
            normalized = normalizer.normalize(context_text)

            # Tokenize
            words = re.findall(r'\b[a-z]+\b', normalized.lower())

            for word in words:
                if len(word) > 2 and word not in stopwords:
                    collocates[word] += 1

    return collocates


def analyze_bin(corpus_dir: Path, bin_label: str, normalizer: EarlyModernNormalizer,
                tagger: DomainTagger) -> dict:
    """Analyze a single time bin."""
    print(f"\nAnalyzing {bin_label}...")

    texts = load_contexts(corpus_dir, bin_label)
    if not texts:
        return {'error': f'No data for {bin_label}'}

    total_contexts = sum(t.get('context_count', 0) for t in texts)
    print(f"  Texts: {len(texts)}, Contexts: {total_contexts}")

    # Compute collocates
    collocates = compute_collocates(texts, normalizer)

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

    # Domain distribution (top 50)
    domain_dist = defaultdict(int)
    for item in tagged_collocates[:50]:
        domain_dist[item['primary_domain']] += 1

    # Sample contexts
    sample_contexts = []
    for text in texts[:10]:
        for ctx in text.get('contexts', [])[:2]:
            sample_contexts.append({
                'year': text['year'],
                'title': text['title'][:60],
                'context': ctx['full_context'],
            })

    return {
        'bin': bin_label,
        'text_count': len(texts),
        'context_count': total_contexts,
        'top_collocates': tagged_collocates,
        'domain_distribution': dict(domain_dist),
        'sample_contexts': sample_contexts[:15],
    }


def main():
    project_root = Path(__file__).parent.parent
    corpus_dir = project_root / "data" / "eebo" / "fulltext_corpus"
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    normalizer = EarlyModernNormalizer()
    tagger = DomainTagger()

    # Time bins
    bins = ['1500-1550', '1550-1600', '1600-1650', '1650-1700']

    results = {}
    for bin_label in bins:
        results[bin_label] = analyze_bin(corpus_dir, bin_label, normalizer, tagger)

    # Save results
    output_path = output_dir / "tier2_fulltext_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved analysis to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("TIER 2 FULL-TEXT ANALYSIS SUMMARY")
    print("=" * 70)

    for bin_label in bins:
        r = results[bin_label]
        if 'error' in r:
            print(f"\n{bin_label}: {r['error']}")
            continue

        print(f"\n{bin_label}:")
        print(f"  Texts with freedom: {r['text_count']}")
        print(f"  Total contexts: {r['context_count']}")

        print(f"  Top 15 collocates:")
        for i, c in enumerate(r['top_collocates'][:15], 1):
            domain_str = c['primary_domain'][:12]
            print(f"    {i:2}. {c['word']:<15} ({c['count']:>5}) [{domain_str}]")

        print(f"\n  Domain distribution (top 50):")
        for domain, count in sorted(r['domain_distribution'].items(), key=lambda x: -x[1]):
            pct = count / 50 * 100
            print(f"    {domain:<25} {count:>2} ({pct:>4.0f}%)")


if __name__ == "__main__":
    main()
