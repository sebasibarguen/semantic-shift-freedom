import unittest

from src.sample_annotation_set import build_validation_sample
from src.score_annotations import (
    cohen_kappa,
    fleiss_kappa,
    human_consensus,
    score,
)


def corpus_record(sid, year, label):
    return {
        "id": sid,
        "year": year,
        "date": f"{year}-01-01",
        "word": "freedom",
        "speaker": "Someone",
        "party": "",
        "sentence": f"A sentence about freedom ({sid}).",
        "methods": {"llm": {"label": label, "rationale": "r"}},
    }


class KappaTests(unittest.TestCase):
    def test_cohen_kappa_perfect_agreement(self):
        pairs = [("a", "a"), ("b", "b"), ("a", "a")]
        self.assertEqual(cohen_kappa(pairs), 1.0)

    def test_cohen_kappa_known_value(self):
        # po = 0.75, pe = 0.5 → kappa = 0.5
        pairs = [("a", "a"), ("a", "a"), ("a", "b"), ("b", "b")]
        self.assertAlmostEqual(cohen_kappa(pairs), 0.5, places=6)

    def test_cohen_kappa_empty_is_none(self):
        self.assertIsNone(cohen_kappa([]))

    def test_fleiss_kappa_perfect_within_item(self):
        # each item unanimous → kappa = 1.0
        items = [{"a": 3}, {"b": 3}]
        self.assertAlmostEqual(fleiss_kappa(items), 1.0, places=6)

    def test_fleiss_kappa_known_negative(self):
        # two items, 3 raters each split 2/1 → kappa = -0.5
        items = [{"a": 2, "b": 1}, {"a": 2, "b": 1}]
        self.assertAlmostEqual(fleiss_kappa(items), -0.5, places=6)

    def test_fleiss_kappa_uneven_raters_is_none(self):
        self.assertIsNone(fleiss_kappa([{"a": 3}, {"b": 2}]))


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.corpus = {f"s{i}": corpus_record(f"s{i}", 1900 + i, "negative_liberty")
                       for i in range(20)}
        self.council = (
            [{"id": "s0", "tier": "disputed", "gold_label": None}]
            + [{"id": f"s{i}", "tier": "silver", "gold_label": "negative_liberty"} for i in (1, 2)]
            + [{"id": f"s{i}", "tier": "gold", "gold_label": "negative_liberty"} for i in (3, 4, 5)]
        )

    def test_deterministic(self):
        a, ka = build_validation_sample(self.corpus, self.council, n_random=3, n_gold=2,
                                        n_silver=2, n_disputed=None, seed=42)
        b, kb = build_validation_sample(self.corpus, self.council, n_random=3, n_gold=2,
                                        n_silver=2, n_disputed=None, seed=42)
        self.assertEqual([r["id"] for r in a], [r["id"] for r in b])
        self.assertEqual(ka["keys"], kb["keys"])

    def test_all_disputed_taken(self):
        _, key = build_validation_sample(self.corpus, self.council, n_random=0, n_gold=0,
                                         n_silver=0, n_disputed=None, seed=1)
        disputed = [sid for sid, v in key["keys"].items() if v["sample_reason"] == "council_disputed"]
        self.assertEqual(disputed, ["s0"])

    def test_annotator_records_strip_model_labels(self):
        records, _ = build_validation_sample(self.corpus, self.council, n_random=5, n_gold=2,
                                             n_silver=2, n_disputed=None, seed=7)
        for r in records:
            self.assertEqual(r["methods"], {})
            self.assertNotIn("haiku_label", r)
            self.assertNotIn("council_gold", r)

    def test_answer_key_carries_model_and_council_labels(self):
        _, key = build_validation_sample(self.corpus, self.council, n_random=0, n_gold=0,
                                         n_silver=2, n_disputed=0, seed=3)
        for sid, info in key["keys"].items():
            self.assertEqual(info["haiku_label"], "negative_liberty")  # from corpus methods.llm
            self.assertEqual(info["council_tier"], "silver")
            self.assertEqual(info["council_gold"], "negative_liberty")

    def test_no_duplicate_ids_across_buckets(self):
        records, _ = build_validation_sample(self.corpus, self.council, n_random=10, n_gold=3,
                                             n_silver=2, n_disputed=None, seed=9)
        ids = [r["id"] for r in records]
        self.assertEqual(len(ids), len(set(ids)))


class ScoreTests(unittest.TestCase):
    def _answer_key(self):
        keys = {
            "a": {"year": 1900, "sample_reason": "random", "haiku_label": "positive_liberty",
                  "council_tier": "gold", "council_gold": "positive_liberty"},
            "b": {"year": 1910, "sample_reason": "random", "haiku_label": "negative_liberty",
                  "council_tier": "gold", "council_gold": "negative_liberty"},
            "c": {"year": 1920, "sample_reason": "council_silver", "haiku_label": "ambiguous",
                  "council_tier": "silver", "council_gold": "negative_liberty"},
        }
        return {"meta": {}, "keys": keys}

    def test_score_basic(self):
        annotators = {
            "alice": {"a": "positive_liberty", "b": "negative_liberty", "c": "negative_liberty"},
            "bob": {"a": "positive_liberty", "b": "negative_liberty", "c": "ambiguous"},
        }
        result = score(self._answer_key(), annotators)
        self.assertEqual(result["n_validation_set"], 3)
        self.assertEqual(result["coverage"]["alice"]["labeled"], 3)
        # alice & bob agree on a,b; disagree on c → c dropped from consensus (tie)
        self.assertEqual(result["human_consensus"]["agreed"], 2)
        self.assertEqual(result["human_consensus"]["tie"], 1)
        # Haiku matches consensus on a and b
        self.assertEqual(result["haiku_vs_human"]["n"], 2)
        self.assertEqual(result["haiku_vs_human"]["agreement"], 1.0)

    def test_score_ignores_ids_outside_validation_set(self):
        annotators = {"alice": {"a": "positive_liberty", "zzz": "other"}}
        result = score(self._answer_key(), annotators)
        self.assertEqual(result["coverage"]["alice"]["labeled"], 1)

    def test_consensus_single_annotator(self):
        consensus, stats = human_consensus({"solo": {"a": "other", "b": "positive_liberty"}})
        self.assertEqual(stats["single"], 2)
        self.assertEqual(consensus, {"a": "other", "b": "positive_liberty"})


if __name__ == "__main__":
    unittest.main()
