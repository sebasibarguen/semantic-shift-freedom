# ABOUTME: Normalizes Early Modern English spelling to modern forms.
# ABOUTME: Handles long-s, u/v, vv/w, i/j, and common spelling variants.

import re
from typing import Callable


# Common spelling variants for freedom-related vocabulary
FREEDOM_VARIANTS = {
    # Freedom variants
    'freedome': 'freedom',
    'fredom': 'freedom',
    'freedoom': 'freedom',
    'fredome': 'freedom',
    'fredeome': 'freedom',
    'freedam': 'freedom',
    # Liberty variants
    'libertie': 'liberty',
    'libertye': 'liberty',
    'lybertie': 'liberty',
    'lyberty': 'liberty',
    'liberte': 'liberty',
    # Tyranny variants
    'tyrannie': 'tyranny',
    'tyrannye': 'tyranny',
    'tiranny': 'tyranny',
    'tirannie': 'tyranny',
    # Slavery variants
    'slaverie': 'slavery',
    'slauerie': 'slavery',
    'slauery': 'slavery',
    # Bondage variants
    'bondadge': 'bondage',
    # Conscience variants
    'conscienc': 'conscience',
    'conſcience': 'conscience',
    # Common word variants
    'haue': 'have',
    'giue': 'give',
    'giuen': 'given',
    'giuing': 'giving',
    'liue': 'live',
    'loue': 'love',
    'aboue': 'above',
    'mooue': 'move',
    'prooue': 'prove',
    'remoue': 'remove',
    'approoue': 'approve',
    'improoue': 'improve',
    'beleeue': 'believe',
    'receiue': 'receive',
    'deceiue': 'deceive',
    'conceiue': 'conceive',
    'perceiue': 'perceive',
    'atchieue': 'achieve',
    'relieue': 'relieve',
    'grieue': 'grieve',
    # -tion/-ation variants
    'nacion': 'nation',
    'condicion': 'condition',
    'declaracion': 'declaration',
    'preseruation': 'preservation',
    'obseruation': 'observation',
    'saluation': 'salvation',
    'damnation': 'damnation',
    'reuelation': 'revelation',
    'congregacion': 'congregation',
    'obligacion': 'obligation',
    # People/names
    'iesus': 'jesus',
    'iesvs': 'jesus',
    'iohn': 'john',
    'iames': 'james',
    # Common words
    'vnto': 'unto',
    'vpon': 'upon',
    'vp': 'up',
    'vs': 'us',
    'vse': 'use',
    'vsed': 'used',
    'vsuall': 'usual',
    'vtter': 'utter',
    'vtmost': 'utmost',
    'evill': 'evil',
    'euill': 'evil',
    'ciuill': 'civil',
    'diuine': 'divine',
    'heauenly': 'heavenly',
    'heauen': 'heaven',
    'leaue': 'leave',
    'seuen': 'seven',
    'eleuen': 'eleven',
    'euer': 'ever',
    'neuer': 'never',
    'whatsoeuer': 'whatsoever',
    'howeuer': 'however',
    'euery': 'every',
    'ouer': 'over',
    'reuerend': 'reverend',
    'gouernment': 'government',
    'gouernour': 'governor',
    'gouerne': 'govern',
    # Misc
    'onely': 'only',
    'vvhich': 'which',
    'vvhat': 'what',
    'vvhere': 'where',
    'vvhen': 'when',
    'vvhy': 'why',
    'vvith': 'with',
    'vvell': 'well',
    'vvork': 'work',
    'vvorke': 'work',
    'vvord': 'word',
    'vvorld': 'world',
    'vvas': 'was',
    'vvere': 'were',
    'vvill': 'will',
    'vvould': 'would',
    'beene': 'been',
    'seene': 'seen',
    'knowne': 'known',
    'shewn': 'shown',
    'owne': 'own',
    'iust': 'just',
    'iudge': 'judge',
    'iudgement': 'judgment',
    'iustice': 'justice',
    'iniustice': 'injustice',
    'iniury': 'injury',
    'subiect': 'subject',
    'subiects': 'subjects',
    'obiect': 'object',
    'proiect': 'project',
    'maiesty': 'majesty',
    'maiestie': 'majesty',
    'maiestye': 'majesty',
}


class EarlyModernNormalizer:
    """Normalizes Early Modern English spelling."""

    def __init__(self, custom_mappings: dict = None):
        self.mappings = dict(FREEDOM_VARIANTS)
        if custom_mappings:
            self.mappings.update(custom_mappings)

        # Build regex for known mappings (case-insensitive)
        self._mapping_pattern = None
        self._build_mapping_pattern()

    def _build_mapping_pattern(self):
        """Build regex pattern for dictionary lookups."""
        if self.mappings:
            # Sort by length (longest first) to avoid partial matches
            sorted_keys = sorted(self.mappings.keys(), key=len, reverse=True)
            escaped = [re.escape(k) for k in sorted_keys]
            self._mapping_pattern = re.compile(
                r'\b(' + '|'.join(escaped) + r')\b',
                re.IGNORECASE
            )

    def _apply_mappings(self, text: str) -> str:
        """Apply dictionary-based mappings."""
        if not self._mapping_pattern:
            return text

        def replace(match):
            word = match.group(0)
            lower = word.lower()
            if lower in self.mappings:
                replacement = self.mappings[lower]
                # Preserve case pattern
                if word.isupper():
                    return replacement.upper()
                elif word[0].isupper():
                    return replacement.capitalize()
                return replacement
            return word

        return self._mapping_pattern.sub(replace, text)

    def normalize_long_s(self, text: str) -> str:
        """Replace long-s (ſ) with regular s."""
        return text.replace('ſ', 's')

    def normalize_vv_to_w(self, text: str) -> str:
        """Replace vv with w (common in early printing)."""
        # Handle both lowercase and uppercase
        text = re.sub(r'vv', 'w', text)
        text = re.sub(r'VV', 'W', text)
        text = re.sub(r'Vv', 'W', text)
        return text

    def normalize_u_v(self, text: str) -> str:
        """
        Normalize u/v interchange.
        In Early Modern English:
        - Initial 'v' often = 'u' (vnto → unto)
        - Medial 'u' often = 'v' (haue → have)

        This is a heuristic - not perfect but catches common cases.
        """
        words = text.split()
        normalized = []

        for word in words:
            if not word:
                normalized.append(word)
                continue

            # Handle initial v → u (before consonant)
            if len(word) > 1 and word[0].lower() == 'v':
                next_char = word[1].lower()
                # If v is followed by a consonant (not a, e, i, o, u)
                if next_char not in 'aeiou':
                    if word[0] == 'V':
                        word = 'U' + word[1:]
                    else:
                        word = 'u' + word[1:]

            normalized.append(word)

        return ' '.join(normalized)

    def normalize_i_j(self, text: str) -> str:
        """
        Normalize i/j interchange.
        In Early Modern English, 'i' was often used for both i and j sounds.
        """
        # Common patterns: initial i before vowel often = j
        # This is handled mainly by the dictionary mappings
        return text

    def normalize_final_e(self, text: str) -> str:
        """
        Normalize final -e patterns.
        Many Early Modern words had silent final -e that modern spelling dropped.
        E.g., 'worke' → 'work'
        """
        # Common patterns
        patterns = [
            (r'\b(\w+)ome\b', r'\1om'),  # freedome → freedom (but not 'some')
            (r'\b(\w+)ke\b', r'\1k'),  # worke → work (but careful with 'like')
        ]

        # Only apply to specific known words to avoid over-correction
        return text

    def normalize_ie_y(self, text: str) -> str:
        """Normalize -ie/-ye endings to -y."""
        # libertie → liberty, etc.
        # Be careful not to affect words like 'die', 'lie', 'pie'
        text = re.sub(r'\b(\w{4,})ie\b', r'\1y', text)
        text = re.sub(r'\b(\w{4,})ye\b', r'\1y', text)
        return text

    def normalize(self, text: str) -> str:
        """Apply all normalizations."""
        # Order matters!

        # 1. Long-s first (simple substitution)
        text = self.normalize_long_s(text)

        # 2. vv → w (before other v/u work)
        text = self.normalize_vv_to_w(text)

        # 3. Dictionary-based mappings (catches most u/v and i/j)
        text = self._apply_mappings(text)

        # 4. u/v normalization for remaining cases
        text = self.normalize_u_v(text)

        # 5. -ie/-ye → -y
        text = self.normalize_ie_y(text)

        return text

    def normalize_word(self, word: str) -> str:
        """Normalize a single word."""
        return self.normalize(word)


def demo():
    """Demonstrate normalization."""
    normalizer = EarlyModernNormalizer()

    test_texts = [
        "The freedome of the ſubiect is the libertie of conſcience",
        "Vnto vvhich vve haue giuen our conſent",
        "The tyrannie of kings and the bondage of the people",
        "Iustice and iudgement belong vnto God",
        "For the preſeruation of their ancient liberties",
        "The euill of slauerie and the blessing of freedome",
        "Euery man ought to haue libertie to worſhip as his conſcience dictates",
    ]

    print("Early Modern English Normalization Demo")
    print("=" * 60)

    for text in test_texts:
        normalized = normalizer.normalize(text)
        print(f"\nOriginal:   {text}")
        print(f"Normalized: {normalized}")


def normalize_corpus_file(input_path: str, output_path: str):
    """Normalize an entire corpus file."""
    from pathlib import Path

    normalizer = EarlyModernNormalizer()

    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                normalized = normalizer.normalize(line)
                f_out.write(normalized)


if __name__ == "__main__":
    demo()
