# ABOUTME: Tags words by semantic domain to track shifts in "freedom" collocates.
# ABOUTME: Domains: servitude, political, economic, personal, religious, legal, abstract.

from typing import Optional
from pathlib import Path
import json


# Domain lexicons - curated lists of words associated with each semantic domain
# These are used to tag collocates and track domain distribution over time

DOMAIN_LEXICONS = {
    'servitude_bondage': {
        # Core slavery/freedom binary (the original meaning)
        'slavery', 'slave', 'slaves', 'enslaved', 'enslaving', 'enslavement',
        'bondage', 'bondsman', 'bondsmen', 'bond', 'bonds', 'bound',
        'servitude', 'servant', 'servants', 'serf', 'serfs', 'serfdom',
        'thrall', 'thralls', 'thralldom', 'captive', 'captives', 'captivity',
        # Unfreedom states
        'chains', 'chained', 'fetters', 'fettered', 'shackles', 'shackled',
        'yoke', 'yoked', 'subjection', 'subjugation', 'subjugated',
        'oppressed', 'oppression', 'oppressor', 'oppressors',
        'domination', 'mastery',
        # Liberation from servitude
        'emancipation', 'emancipate', 'emancipated', 'manumission', 'manumitted',
        'liberated', 'liberating', 'liberation', 'liberator',
        'freed', 'freeing', 'freedman', 'freedmen', 'freedwoman',
        # Related status terms
        'master', 'masters', 'mistress', 'owner', 'owners',
        'vassal', 'vassals', 'villeinage', 'villein',
        # Abolition movement
        'abolition', 'abolitionist', 'abolitionists', 'antislavery',
    },

    'constraint_liberation': {
        # Words describing the state/experience of freedom vs constraint
        # This captures the semantic field of freedom itself
        # Constraint/restriction
        'restraint', 'restraints', 'restrained', 'restriction', 'restrictions',
        'constraint', 'constraints', 'constrained', 'confinement', 'confined',
        'encroachment', 'encroachments', 'infringement', 'infringements',
        'curtailment', 'abridging', 'abridge', 'abridged', 'abridges',
        'deprivation', 'depriving', 'deprived', 'deprive', 'deprives', 'denial',
        'infringe', 'infringed', 'violated', 'violating',
        'restrict', 'restricting', 'hindrance', 'impede',
        # Freedom/liberation states
        'unfettered', 'unrestrained', 'unrestricted', 'unconstrained',
        'uncontrolled', 'unimpeded', 'undisturbed',
        'free', 'freely', 'freedoms', 'liberties', 'freest',
        'enjoyment', 'enjoyed', 'enjoy', 'enjoying', 'enjoys',
        'regain', 'regained', 'restore', 'maintain',
        # Protection of freedom
        'guarantee', 'guarantees', 'guaranteed', 'guaranteeing', 'guaranties',
        'protect', 'protected', 'protection', 'safeguard', 'safeguards', 'safeguarding',
        'preserve', 'preserved', 'preservation', 'secure', 'secured', 'security', 'secures',
        'vindicate', 'vindicates', 'vindication',
        # Inalienable rights language
        'inalienable', 'unalienable', 'inviolable', 'inviolability', 'inviolate',
        'birthright', 'inherent',
        # Palladium (metaphor for safeguard of liberty, common 19th c.)
        'palladium',
        # Exercise and grant freedom
        'exercise', 'granting', 'confers', 'conserve',
        # Threats to freedom
        'abuse', 'destroys', 'endangers', 'subversive', 'subvert', 'inimical',
    },

    'political': {
        # Government and state
        'government', 'state', 'nation', 'country', 'republic', 'democracy',
        'tyranny', 'despotism', 'monarchy', 'monarchical', 'parliament', 'congress', 'senate',
        'constitution', 'legislation', 'vote', 'election', 'ballot',
        'citizen', 'citizenship', 'patriot', 'patriotism', 'revolution', 'rebellion',
        'absolutism', 'centralization',
        # Rights and civil liberties
        'speech', 'press', 'assembly', 'petition', 'protest', 'dissent',
        'civil', 'political', 'democratic', 'republican', 'liberal', 'conservative',
        'censorship',
        # Power relations
        'power', 'authority', 'sovereignty', 'independence', 'independency', 'self-determination',
        'supremacy',
        # Specific political concepts
        'equality', 'justice', 'representation', 'suffrage', 'franchise', 'franchises',
        'solidarity', 'fraternity', 'pluralism',
        # Collective/national
        'unity', 'fatherland', 'motherland', 'institutions', 'social',
        'nationality', 'peoples', 'civilization',
        # Advocacy
        'struggle', 'struggling', 'champion', 'champions', 'advocacy', 'advocates', 'advocating',
        'fighters', 'fight', 'triumph', 'assert', 'asserted', 'asserting',
        # Participation in political life
        'participate', 'participating', 'participates', 'participation',
        'recognition', 'slogan',
    },

    'economic': {
        # Markets and trade
        'market', 'trade', 'commerce', 'business', 'enterprise', 'industry',
        'capitalism', 'socialism', 'communism', 'competition', 'monopoly',
        # Property and ownership
        'property', 'ownership', 'private', 'wealth', 'capital', 'investment',
        # Labor
        'labor', 'work', 'employment', 'wage', 'worker', 'union',
        'contract', 'bargaining', 'employer', 'employee',
        # Economic freedom concepts
        'choice', 'consumer', 'regulation', 'deregulation', 'laissez-faire',
        'taxation', 'tariff', 'subsidy', 'welfare',
    },

    'personal': {
        # Individual autonomy
        'individual', 'personal', 'private', 'autonomy', 'self',
        'lifestyle', 'preference', 'expression', 'identity',
        'individualism', 'individuality', 'individualistic',
        # Privacy and personal space
        'privacy', 'intimate', 'domestic', 'family', 'marriage',
        # Self-development
        'happiness', 'pursuit', 'fulfillment', 'self-realization',
        'contentment', 'prosperity', 'attainment', 'realization',
        # Movement and action
        'movement', 'travel', 'mobility', 'action', 'behavior',
        # Modern personal freedoms
        'sexuality', 'reproductive', 'bodily', 'conscience',
        # Spontaneity and self-determination
        'spontaneity', 'spontaneous',
        # Personal qualities/states
        'boldness', 'bold', 'openness', 'open', 'frankness', 'frank',
        'ease', 'easiness', 'tranquillity', 'tranquility', 'peace',
        'vigor', 'energy', 'vitality',
        'manly', 'manful', 'fearlessness',
        # Virtue words associated with freedom
        'magnanimity', 'nobleness', 'unselfishness',
        'integrity', 'morality',
        # Choice and desire
        'choose', 'choice', 'yearning', 'love', 'loving',
        # Personal qualities
        'assertiveness', 'flexibility', 'sentiment',
    },

    'religious': {
        # Religious institutions
        'church', 'religion', 'religious', 'faith', 'worship', 'prayer', 'clergy',
        'christian', 'catholic', 'protestant', 'protestantism', 'jewish', 'muslim',
        # Theological concepts
        'god', 'divine', 'soul', 'spirit', 'spiritual', 'salvation',
        'sin', 'redemption', 'grace', 'heaven', 'hell',
        'immortality', 'immortal', 'eternal', 'eternity',
        # Religious values/virtues
        'purity', 'pure', 'blessings', 'blessed', 'sacred', 'holy', 'sanctity',
        'innocence', 'righteousness', 'righteous',
        # Religious freedom
        'conscience', 'belief', 'toleration', 'tolerance', 'tolerating', 'persecution',
        'heresy', 'blasphemy', 'apostasy',
        # Reformation era
        'reformation', 'scripture', 'gospel',
        # Temperance movement (religious-adjacent)
        'temperance',
    },

    'legal': {
        # Legal system
        'law', 'legal', 'court', 'judge', 'jury', 'trial',
        'statute', 'ordinance', 'regulation', 'code',
        # Rights and protections
        'right', 'rights', 'privilege', 'privileges', 'immunity', 'immunities',
        'due', 'process', 'habeas', 'corpus',
        # Legal status (not covered by servitude_bondage)
        'citizen', 'subject', 'alien', 'enfranchisement',
        # Constitutional law
        'constitutional', 'constitutionally', 'unconstitutional',
        'amendment', 'amendments', 'ratified',
        # Legal procedures
        'arrest', 'imprisonment', 'detention', 'bail', 'pardon',
        'incrimination', 'self-incrimination', 'expatriation',
    },

    'abstract_philosophical': {
        # Philosophical concepts
        'liberty', 'autonomy', 'self-determination', 'agency',
        'will', 'free-will', 'determinism', 'necessity',
        # Abstract values
        'virtue', 'dignity', 'humanity', 'nature', 'natural',
        'reason', 'rational', 'rationality', 'enlightenment',
        'ideals', 'ideal', 'concept',
        # Philosophical schools
        'liberal', 'liberalism', 'libertarian', 'utilitarian',
        # Contrast concepts
        'constraint', 'coercion', 'compulsion', 'necessity',
        'positive', 'negative', 'interference', 'interfering', 'interfere', 'interferes', 'interfered',
        # Compatibility/logical relations
        'compatible', 'incompatible', 'consistent', 'inconsistent',
        # Abstract qualities
        'permanence', 'permanent', 'permanency', 'unlimited', 'infinite',
        'originality', 'original', 'principle', 'principles',
        'moderation', 'moderate',
        'absolute', 'absoluteness', 'universality',
        'inestimable', 'indivisibility',
        # Consciousness/awareness
        'consciousness', 'objectivity',
        # Commitment/dedication
        'commitment', 'reliance', 'fervor',
    },
}

# Flattened lookup for quick domain assignment
_WORD_TO_DOMAINS = {}
for domain, words in DOMAIN_LEXICONS.items():
    for word in words:
        if word not in _WORD_TO_DOMAINS:
            _WORD_TO_DOMAINS[word] = []
        _WORD_TO_DOMAINS[word].append(domain)


class DomainTagger:
    """Tags words with semantic domains."""

    def __init__(self, custom_lexicons: Optional[dict] = None):
        self.lexicons = dict(DOMAIN_LEXICONS)
        if custom_lexicons:
            for domain, words in custom_lexicons.items():
                if domain in self.lexicons:
                    self.lexicons[domain].update(words)
                else:
                    self.lexicons[domain] = set(words)

        # Rebuild lookup
        self._word_to_domains = {}
        for domain, words in self.lexicons.items():
            for word in words:
                if word not in self._word_to_domains:
                    self._word_to_domains[word] = []
                self._word_to_domains[word].append(domain)

    def tag(self, word: str) -> list[str]:
        """Return list of domains a word belongs to."""
        word_lower = word.lower()
        return self._word_to_domains.get(word_lower, ['untagged'])

    def tag_primary(self, word: str) -> str:
        """Return the primary (first) domain for a word."""
        domains = self.tag(word)
        return domains[0] if domains else 'untagged'

    def tag_list(self, words: list[str]) -> dict[str, list[str]]:
        """Tag a list of words, returning dict mapping word -> domains."""
        return {word: self.tag(word) for word in words}

    def get_domain_distribution(self, words: list[str]) -> dict[str, int]:
        """
        Calculate distribution of domains across a word list.
        Useful for analyzing collocate domain distribution over time.
        """
        distribution = {domain: 0 for domain in self.lexicons.keys()}
        distribution['untagged'] = 0

        for word in words:
            domains = self.tag(word)
            for domain in domains:
                distribution[domain] = distribution.get(domain, 0) + 1

        return distribution

    def get_domain_words(self, domain: str) -> set[str]:
        """Get all words in a domain."""
        return self.lexicons.get(domain, set())


def analyze_collocate_domains(collocates: list[tuple[str, float]], tagger: DomainTagger) -> dict:
    """
    Analyze domain distribution of collocates.

    Args:
        collocates: List of (word, score) tuples from nearest neighbor analysis
        tagger: DomainTagger instance

    Returns:
        Dict with domain distribution and tagged collocates
    """
    words = [w for w, _ in collocates]

    distribution = tagger.get_domain_distribution(words)

    tagged_collocates = []
    for word, score in collocates:
        domains = tagger.tag(word)
        tagged_collocates.append({
            'word': word,
            'score': score,
            'domains': domains,
            'primary_domain': domains[0]
        })

    # Calculate percentages
    total = len(words)
    percentages = {
        domain: count / total * 100
        for domain, count in distribution.items()
    }

    return {
        'total_collocates': total,
        'distribution': distribution,
        'percentages': percentages,
        'tagged_collocates': tagged_collocates
    }


def demo():
    """Demonstrate domain tagging."""
    tagger = DomainTagger()

    print("Domain Tagger Demo")
    print("=" * 50)

    # Test words
    test_words = [
        'liberty', 'democracy', 'market', 'soul', 'privacy',
        'constitution', 'capitalism', 'conscience', 'happiness',
        'slavery', 'emancipation', 'autonomy', 'regulation'
    ]

    print("\nWord -> Domain mappings:")
    print("-" * 50)
    for word in test_words:
        domains = tagger.tag(word)
        print(f"  {word}: {', '.join(domains)}")

    # Simulate collocate analysis
    print("\n\nSimulated collocate domain distribution:")
    print("-" * 50)

    # Mock collocates from different eras
    collocates_1800 = [
        ('liberty', 0.65), ('independence', 0.45), ('slavery', 0.40),
        ('emancipation', 0.38), ('soul', 0.35), ('conscience', 0.32),
        ('rights', 0.30), ('licentiousness', 0.28)
    ]

    collocates_1990 = [
        ('liberty', 0.51), ('democracy', 0.36), ('rights', 0.34),
        ('autonomy', 0.38), ('choice', 0.33), ('individual', 0.31),
        ('market', 0.28), ('expression', 0.27)
    ]

    print("\n1800s collocates:")
    result_1800 = analyze_collocate_domains(collocates_1800, tagger)
    for domain, pct in sorted(result_1800['percentages'].items(), key=lambda x: -x[1]):
        if pct > 0:
            print(f"  {domain}: {pct:.1f}%")

    print("\n1990s collocates:")
    result_1990 = analyze_collocate_domains(collocates_1990, tagger)
    for domain, pct in sorted(result_1990['percentages'].items(), key=lambda x: -x[1]):
        if pct > 0:
            print(f"  {domain}: {pct:.1f}%")


if __name__ == "__main__":
    demo()
