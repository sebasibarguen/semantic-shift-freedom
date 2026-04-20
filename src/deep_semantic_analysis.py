# ABOUTME: Analyzes semantic domains of FROM/TO objects to test if surface syntax reflects deep meaning.
# ABOUTME: Categorizes objects into domains and computes overlap between negative and positive framings.

"""
Deep Semantic Analysis of Freedom FROM/TO Objects

Tests whether the surface distinction (FROM vs TO) reflects genuine semantic difference
or whether both framings express the same underlying concepts in different domains.

Key question: If "freedom from sin" and "freedom to worship" both map to Religious domain,
does the FROM/TO distinction capture real semantic difference?
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


# Function words to filter out (not semantically meaningful in this context)
FUNCTION_WORDS = {
    # Articles
    'the', 'a', 'an',
    # Pronouns
    'it', 'them', 'him', 'her', 'us', 'me', 'them', 'themselves', 'himself',
    'herself', 'itself', 'ourselves', 'yourself', 'yourselves', 'myself',
    # Possessives
    'his', 'her', 'their', 'our', 'my', 'your', 'its',
    # Demonstratives
    'this', 'that', 'these', 'those',
    # Quantifiers
    'all', 'any', 'some', 'every', 'each', 'such', 'many', 'much', 'few',
    # Conjunctions/prepositions commonly following freedom
    'which', 'what', 'who', 'whom', 'whose', 'where', 'when', 'how', 'why',
    # Other function words
    'be', 'have', 'had', 'having', 'been', 'being',
}

# Semantic domain definitions
DOMAIN_DEFINITIONS = {
    'Servitude': [
        'slavery', 'bondage', 'chains', 'servitude', 'captivity', 'thrall',
        'slave', 'slaves', 'bond', 'yoke', 'fetters', 'manumit', 'emancipation',
        'servile', 'subjection', 'subjugation', 'vassalage', 'thraldom',
        'enslavement', 'serfdom', 'master', 'masters',
    ],
    'Religious': [
        'sin', 'sinne', 'god', 'christ', 'soul', 'salvation', 'conscience',
        'worship', 'pray', 'preach', 'church', 'spirit', 'spiritual', 'holy',
        'heaven', 'hell', 'devil', 'satan', 'faith', 'grace', 'redemption',
        'damnation', 'condemnation', 'eternal', 'divine', 'sacred', 'profane',
        'antichristianisme', 'ceremoniall', 'gospel', 'scripture',
        'righteousness', 'godly', 'ungodly', 'wicked', 'wickedness',
        'iniquity', 'transgression', 'trespass', 'repentance', 'judgement',
        'judgment', 'prayer', 'prayers', 'flesh', 'carnal',
    ],
    'Political': [
        'tyranny', 'tyrant', 'oppression', 'king', 'parliament', 'law', 'laws',
        'government', 'vote', 'citizen', 'rights', 'constitution', 'democracy',
        'republic', 'nation', 'state', 'ruler', 'subject', 'sovereignty',
        'allegiance', 'authority', 'power', 'liberty', 'liberties',
        'prince', 'princes', 'crown', 'kingdom', 'realm', 'commonwealth',
        'magistrate', 'magistrates', 'civil', 'civill', 'dominion',
    ],
    'Economic': [
        'debt', 'money', 'trade', 'market', 'property', 'wealth', 'labor',
        'work', 'poverty', 'want', 'profit', 'income', 'financial', 'economic',
        'commerce', 'merchant', 'tax', 'taxes', 'labour', 'wages', 'goods',
        'riches', 'estate', 'inheritance',
    ],
    'Personal/Psychological': [
        'fear', 'pain', 'misery', 'trouble', 'anxiety', 'suffering', 'grief',
        'sorrow', 'distress', 'anguish', 'torment', 'passion', 'passions',
        'desire', 'will', 'choice', 'choose', 'chuse', 'autonomy', 'self',
        'mind', 'thought', 'think', 'conscience', 'care', 'cares', 'worry',
        'danger', 'dangers', 'hurt', 'harm', 'affliction', 'vexation',
    ],
    'Action/Capacity': [
        'do', 'doe', 'act', 'speak', 'speake', 'go', 'goe', 'make', 'take',
        'give', 'use', 'come', 'live', 'travel', 'marry', 'enter', 'depart',
        'exercise', 'perform', 'execute', 'serve', 'tell', 'say', 'write',
        'read', 'print', 'publish', 'move', 'walk', 'return', 'pass', 'buy',
        'sell', 'build', 'create', 'dispose', 'enjoy', 'possess', 'keep',
        'hold', 'receive', 'obtain', 'acquire', 'attain', 'follow', 'obey',
        'disobey', 'resist', 'refuse', 'deny', 'grant', 'allow', 'permit',
    ],
    'Constraint/Absence': [
        'constraint', 'coercion', 'coaction', 'necessity', 'compulsion',
        'restraint', 'restriction', 'limitation', 'impediment', 'hindrance',
        'obstacle', 'barrier', 'arrest', 'arrests', 'imprisonment',
        'punishment', 'penalty', 'persecution', 'violence', 'force',
        'burden', 'burdens', 'obligation', 'obligations', 'duty', 'duties',
        'command', 'commands', 'control', 'censure',
    ],
    'Mortality': [
        'death', 'dying', 'dead', 'die', 'dye', 'mortality', 'mortal',
        'destruction', 'perish', 'decay', 'corruption', 'grave', 'tomb',
    ],
    'Moral/Evil': [
        'evil', 'evill', 'evils', 'vice', 'vices', 'corruption', 'malice',
        'injustice', 'wrong', 'wrongs', 'injury', 'injuries', 'crime',
        'crimes', 'fault', 'faults', 'error', 'errors', 'folly',
    ],
}


def tag_domain(word: str) -> list:
    """Tag a word with its semantic domain(s). Returns list of matching domains."""
    word_lower = word.lower().strip()

    # Filter out function words
    if word_lower in FUNCTION_WORDS:
        return ['Function_Word']

    domains = []
    for domain, terms in DOMAIN_DEFINITIONS.items():
        if word_lower in [t.lower() for t in terms]:
            domains.append(domain)
    return domains if domains else ['Untagged']


def analyze_eebo_objects(data: dict) -> dict:
    """Analyze FROM/TO objects from EEBO data."""
    from_objects = data.get('from_following', {})
    to_objects = data.get('to_following', {})

    # Tag each object
    from_tagged = {}
    for word, count in from_objects.items():
        domains = tag_domain(word)
        from_tagged[word] = {'count': count, 'domains': domains}

    to_tagged = {}
    for word, count in to_objects.items():
        domains = tag_domain(word)
        to_tagged[word] = {'count': count, 'domains': domains}

    # Aggregate by domain (excluding Function_Word from semantic analysis)
    from_by_domain = defaultdict(int)
    to_by_domain = defaultdict(int)
    from_function_word_count = 0
    to_function_word_count = 0

    for word, info in from_tagged.items():
        for domain in info['domains']:
            if domain == 'Function_Word':
                from_function_word_count += info['count']
            else:
                from_by_domain[domain] += info['count']

    for word, info in to_tagged.items():
        for domain in info['domains']:
            if domain == 'Function_Word':
                to_function_word_count += info['count']
            else:
                to_by_domain[domain] += info['count']

    return {
        'from_tagged': from_tagged,
        'to_tagged': to_tagged,
        'from_by_domain': dict(from_by_domain),
        'to_by_domain': dict(to_by_domain),
        'from_function_words_filtered': from_function_word_count,
        'to_function_words_filtered': to_function_word_count,
    }


def compute_domain_overlap(from_domains: dict, to_domains: dict) -> dict:
    """Compute overlap between FROM and TO domain distributions."""
    all_domains = set(from_domains.keys()) | set(to_domains.keys())

    # Normalize to percentages
    from_total = sum(from_domains.values())
    to_total = sum(to_domains.values())

    from_pct = {d: from_domains.get(d, 0) / from_total * 100 for d in all_domains}
    to_pct = {d: to_domains.get(d, 0) / to_total * 100 for d in all_domains}

    # Compute overlap (minimum of the two percentages for each domain)
    overlap = {}
    total_overlap = 0
    for domain in all_domains:
        overlap[domain] = min(from_pct.get(domain, 0), to_pct.get(domain, 0))
        total_overlap += overlap[domain]

    return {
        'from_distribution': from_pct,
        'to_distribution': to_pct,
        'overlap_by_domain': overlap,
        'total_overlap_pct': total_overlap,
    }


def analyze_paraphrasability(from_objects: dict, to_objects: dict) -> dict:
    """
    Analyze whether FROM and TO objects are paraphrasable.
    E.g., "freedom from persecution" ↔ "freedom to worship"
    """
    # Known paraphrase pairs (FROM constraint → TO action it enables)
    paraphrase_mappings = {
        # Religious
        'sin': ['worship', 'pray', 'preach', 'serve'],
        'persecution': ['worship', 'preach', 'speak'],
        'condemnation': ['live', 'enter'],
        # Political
        'tyranny': ['speak', 'vote', 'act'],
        'oppression': ['speak', 'act', 'live'],
        # Economic
        'debt': ['work', 'live', 'choose'],
        'want': ['live', 'choose'],
        'poverty': ['live', 'work'],
        # Personal
        'fear': ['speak', 'act', 'live', 'choose'],
        'pain': ['live', 'act'],
        # Constraint
        'constraint': ['act', 'do', 'choose'],
        'coercion': ['choose', 'act'],
        'necessity': ['choose', 'act'],
    }

    # Check which FROM objects have TO paraphrases
    paraphrasable = {}
    non_paraphrasable = {}

    for from_word in from_objects:
        from_lower = from_word.lower()
        if from_lower in paraphrase_mappings:
            to_equivalents = paraphrase_mappings[from_lower]
            found_to = [t for t in to_equivalents if t in [w.lower() for w in to_objects]]
            if found_to:
                paraphrasable[from_word] = {
                    'count': from_objects[from_word],
                    'to_equivalents': found_to,
                }
            else:
                non_paraphrasable[from_word] = from_objects[from_word]
        else:
            non_paraphrasable[from_word] = from_objects[from_word]

    paraphrasable_count = sum(p['count'] for p in paraphrasable.values())
    non_paraphrasable_count = sum(non_paraphrasable.values())
    total = paraphrasable_count + non_paraphrasable_count

    pct = paraphrasable_count / total * 100 if total > 0 else 0
    if pct > 50:
        interpretation = (
            'High paraphrasability suggests FROM/TO distinction is largely surface grammar.'
        )
    elif pct > 30:
        interpretation = (
            'Moderate paraphrasability suggests partial overlap between FROM and TO meanings.'
        )
    else:
        interpretation = (
            'Low paraphrasability suggests FROM/TO captures genuine semantic difference.'
        )

    return {
        'paraphrasable': paraphrasable,
        'non_paraphrasable': non_paraphrasable,
        'paraphrasable_pct': pct,
        'interpretation': interpretation,
    }


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'outputs'

    print("=" * 70)
    print("DEEP SEMANTIC ANALYSIS: Freedom FROM/TO Objects")
    print("=" * 70)
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Load EEBO data
    eebo_path = output_dir / 'negative_positive_eebo.json'
    with open(eebo_path) as f:
        eebo_data = json.load(f)

    print("1. EEBO-TCP Analysis (1500-1700)")
    print("-" * 70)

    eebo_analysis = analyze_eebo_objects(eebo_data)

    print(f"\nFunction words filtered out:")
    print(f"  FROM: {eebo_analysis['from_function_words_filtered']:,} instances")
    print(f"  TO: {eebo_analysis['to_function_words_filtered']:,} instances")

    print("\nFROM objects by domain (content words only):")
    from_domains = eebo_analysis['from_by_domain']
    from_total = sum(from_domains.values())
    for domain, count in sorted(from_domains.items(), key=lambda x: x[1], reverse=True):
        pct = count / from_total * 100
        print(f"  {domain:<25} {count:>6} ({pct:>5.1f}%)")

    print("\nTO objects by domain (content words only):")
    to_domains = eebo_analysis['to_by_domain']
    to_total = sum(to_domains.values())
    for domain, count in sorted(to_domains.items(), key=lambda x: x[1], reverse=True):
        pct = count / to_total * 100
        print(f"  {domain:<25} {count:>6} ({pct:>5.1f}%)")

    # Compute overlap
    print("\n2. Domain Overlap Analysis")
    print("-" * 70)

    overlap = compute_domain_overlap(from_domains, to_domains)

    print("\nDomain distribution comparison:")
    print(f"{'Domain':<25} {'FROM %':>10} {'TO %':>10} {'Overlap':>10}")
    print("-" * 55)
    for domain in sorted(overlap['from_distribution'].keys()):
        from_pct = overlap['from_distribution'].get(domain, 0)
        to_pct = overlap['to_distribution'].get(domain, 0)
        ovl = overlap['overlap_by_domain'].get(domain, 0)
        print(f"{domain:<25} {from_pct:>9.1f}% {to_pct:>9.1f}% {ovl:>9.1f}%")

    print("-" * 55)
    print(f"{'TOTAL OVERLAP':<25} {'':<10} {'':<10} {overlap['total_overlap_pct']:>9.1f}%")

    # Paraphrasability analysis
    print("\n3. Paraphrasability Analysis")
    print("-" * 70)

    para = analyze_paraphrasability(
        eebo_data.get('from_following', {}),
        eebo_data.get('to_following', {})
    )

    print(f"\nParaphrasable FROM objects: {para['paraphrasable_pct']:.1f}%")
    print("\nTop paraphrasable pairs:")
    for word, info in sorted(para['paraphrasable'].items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
        print(f"  'freedom from {word}' ↔ 'freedom to {', '.join(info['to_equivalents'])}'")
        print(f"    (count: {info['count']})")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Check if Religious domain appears in both
    religious_from = overlap['from_distribution'].get('Religious', 0)
    religious_to = overlap['to_distribution'].get('Religious', 0)

    action_to = overlap['to_distribution'].get('Action/Capacity', 0)
    constraint_from = overlap['from_distribution'].get('Constraint/Absence', 0)

    print(f"""
1. DOMAIN OVERLAP: {overlap['total_overlap_pct']:.1f}%
   - If high (>50%), FROM/TO distinction is largely surface grammar
   - If low (<30%), FROM/TO captures genuine semantic difference

2. RELIGIOUS DOMAIN:
   - In FROM objects: {religious_from:.1f}%
   - In TO objects: {religious_to:.1f}%
   - Interpretation: {"SAME domain, different framing" if min(religious_from, religious_to) > 10 else "Different domains"}

3. ACTION vs CONSTRAINT:
   - TO objects in Action/Capacity domain: {action_to:.1f}%
   - FROM objects in Constraint/Absence domain: {constraint_from:.1f}%
   - This IS a real difference: TO enables action, FROM removes constraint

4. PARAPHRASABILITY: {para['paraphrasable_pct']:.1f}%
   - {para['interpretation']}

5. CONCLUSION:
""")

    if overlap['total_overlap_pct'] > 40:
        conclusion = """   The FROM/TO distinction is PARTIALLY surface grammar.
   Both framings often express the same underlying concepts (especially Religious).
   However, TO uniquely emphasizes ACTION, while FROM uniquely emphasizes CONSTRAINT.

   The 19th-century "negative peak" may reflect:
   - Abolition discourse returning to the ORIGINAL "non-slave" meaning
   - NOT a semantic shift, but a return to etymological roots

   The modern "positive" shift reflects:
   - New domains (Economic, Personal autonomy) that inherently use TO framing
   - These ARE semantically new, not just surface reframing"""
    else:
        conclusion = """   The FROM/TO distinction captures GENUINE semantic difference.
   FROM objects cluster in different domains than TO objects.
   Surface syntax reflects deep meaning."""

    print(conclusion)

    # Save results
    results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'source': 'EEBO-TCP',
            'purpose': 'Test whether FROM/TO distinction is surface or deep',
        },
        'domain_definitions': DOMAIN_DEFINITIONS,
        'eebo_analysis': {
            'from_by_domain': eebo_analysis['from_by_domain'],
            'to_by_domain': eebo_analysis['to_by_domain'],
        },
        'overlap_analysis': overlap,
        'paraphrasability': {
            'paraphrasable_pct': para['paraphrasable_pct'],
            'top_pairs': {k: v for k, v in list(para['paraphrasable'].items())[:10]},
        },
        'conclusion': {
            'total_overlap': overlap['total_overlap_pct'],
            'interpretation': 'partial_surface' if overlap['total_overlap_pct'] > 40 else 'genuine_difference',
        },
    }

    output_path = output_dir / 'deep_semantic_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
