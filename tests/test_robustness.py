# ABOUTME: Tests the legal→personal gap lexicon-sensitivity wrapper.
# ABOUTME: Uses a deterministic fake embedding so the trend actually computes.

import unittest

import numpy as np

from src.robustness import SLAVERY_TERMS, gap_lexicon_sensitivity


class FakeEmbeddings:
    """Deterministic vectors: 'freedom' drifts toward the personal cluster
    over time, so the personal-minus-legal gap declines — a real trend the
    wrapper can measure without loading HistWords."""

    def __init__(self, legal, personal):
        self.legal = set(legal)
        self.personal = set(personal)

    def get_vector(self, word, decade):
        t = (decade - 1800) / 190.0  # 0.0 at 1800 → 1.0 at 1990
        if word == "freedom":
            # Starts near the legal axis, rotates toward the personal axis.
            return np.array([1.0 - t, t, 0.1])
        if word in self.legal:
            return np.array([1.0, 0.0, 0.0])
        if word in self.personal:
            return np.array([0.0, 1.0, 0.0])
        return None


class GapLexiconSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.legal = ["slavery", "bondage", "emancipation", "rights", "law",
                      "citizen", "slave", "servitude"]
        self.personal = ["choice", "autonomy", "independence", "self",
                         "ability", "power", "individual", "personal"]
        self.decades = list(range(1800, 2000, 10))
        self.emb = FakeEmbeddings(self.legal, self.personal)

    def _run(self):
        return gap_lexicon_sensitivity(
            self.emb, "freedom", self.legal, self.personal, self.decades,
            n_permutations=50, rng=np.random.default_rng(0),
        )

    def test_leave_one_out_drops_exactly_one_word_each(self):
        res = self._run()
        self.assertEqual(set(res["leave_one_out"]), set(self.legal))
        for dropped, entry in res["leave_one_out"].items():
            self.assertNotIn(dropped, entry["legal_words"])
            self.assertEqual(len(entry["legal_words"]), len(self.legal) - 1)

    def test_no_slavery_variant_removes_exactly_slavery_terms(self):
        res = self._run()
        kept = res["no_slavery_terms"]["legal_words"]
        self.assertEqual(set(kept), set(self.legal) - set(SLAVERY_TERMS))
        for term in SLAVERY_TERMS:
            self.assertNotIn(term, kept)

    def test_trend_is_negative_and_computed_for_all_variants(self):
        # Freedom rotates toward the personal cluster → gap declines everywhere.
        res = self._run()
        self.assertLess(res["full_cluster"]["slope_per_century"], 0)
        self.assertLess(res["no_slavery_terms"]["slope_per_century"], 0)
        for entry in res["leave_one_out"].values():
            self.assertIsNotNone(entry)


if __name__ == "__main__":
    unittest.main()
