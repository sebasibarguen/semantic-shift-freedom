# ABOUTME: Tests the Method 1 robustness diagnostics (composition, handoff, error).
# ABOUTME: Verifies the standardization and weighted-OLS math on hand-built cases.

import json
import tempfile
import unittest
from pathlib import Path

from src.trend_robustness import (
    _invert,
    _weighted_ols,
    classifier_error_over_time,
    composition_adjusted_trend,
    handoff_controlled_trend,
    per_word_trends,
    primary_domain,
    source_segment,
)


def rec(year, word, label, source="", domains=None):
    return {
        "year": year,
        "word": word,
        "source_file": source,
        "methods": {"llm": {"label": label}, "domains": domains or {}},
    }


class SegmentAndDomainTests(unittest.TestCase):
    def test_source_segment_splits_on_source_file(self):
        self.assertEqual(source_segment({"source_file": "S3V0117P0"}), "historic_hansard")
        self.assertEqual(source_segment({"source_file": ""}), "parlparse")
        self.assertEqual(source_segment({}), "parlparse")

    def test_primary_domain_breaks_ties_alphabetically(self):
        self.assertEqual(
            primary_domain(rec(1850, "freedom", "x", domains={"legal": 5, "economic": 3})),
            "legal",
        )
        self.assertEqual(
            primary_domain(rec(1850, "freedom", "x", domains={"legal": 2, "economic": 2})),
            "economic",
        )
        self.assertEqual(primary_domain(rec(1850, "freedom", "x")), "untagged")


class LinearAlgebraTests(unittest.TestCase):
    def test_weighted_ols_recovers_known_plane(self):
        # y = 2 + 3*x1 - 1*x2 exactly.
        pts = [
            ([1.0, 0.0, 0.0], 2.0, 1.0),
            ([1.0, 1.0, 0.0], 5.0, 1.0),
            ([1.0, 0.0, 1.0], 1.0, 1.0),
            ([1.0, 2.0, 1.0], 7.0, 1.0),
            ([1.0, 1.0, 1.0], 4.0, 1.0),
        ]
        beta = _weighted_ols(pts)["beta"]
        self.assertAlmostEqual(beta[0], 2.0, places=9)
        self.assertAlmostEqual(beta[1], 3.0, places=9)
        self.assertAlmostEqual(beta[2], -1.0, places=9)

    def test_invert_roundtrips(self):
        a = [[4.0, 3.0], [6.0, 3.0]]
        inv = _invert(a)
        prod = [[sum(a[i][k] * inv[k][j] for k in range(2)) for j in range(2)]
                for i in range(2)]
        self.assertAlmostEqual(prod[0][0], 1.0)
        self.assertAlmostEqual(prod[1][1], 1.0)
        self.assertAlmostEqual(prod[0][1], 0.0, places=9)


class CompositionTests(unittest.TestCase):
    def test_adjustment_neutralizes_pure_mix_shift(self):
        # Both topics present every decade with CONSTANT within-topic shares
        # (legal=0.1, personal=0.5); only their relative volume shifts toward
        # the high-positive 'personal' topic. Raw trend rises; standardized
        # trend (topic mix held fixed) stays flat.
        cells = {
            (1800, "legal"): (10, 90), (1800, "personal"): (10, 10),
            (1850, "legal"): (5, 45), (1850, "personal"): (25, 25),
            (1900, "legal"): (1, 9), (1900, "personal"): (45, 45),
            (1950, "legal"): (1, 9), (1950, "personal"): (45, 45),
        }
        records = []
        for (decade, dom), (pos, neg) in cells.items():
            records += [rec(decade, "freedom", "positive_liberty", domains={dom: 1})
                        for _ in range(pos)]
            records += [rec(decade, "freedom", "negative_liberty", domains={dom: 1})
                        for _ in range(neg)]

        res = composition_adjusted_trend(records, min_domain_denominator=10,
                                         min_denominator=10)
        self.assertGreater(res["raw_trend"]["slope_per_century"], 0.05)
        self.assertLess(abs(res["composition_adjusted_trend"]["slope_per_century"]), 0.02)


class PerWordTests(unittest.TestCase):
    def test_split_reports_both_words(self):
        records = [
            rec(1800, "freedom", "positive_liberty"),
            rec(1800, "freedom", "negative_liberty"),
            rec(1900, "freedom", "positive_liberty"),
            rec(1800, "liberty", "negative_liberty"),
            rec(1900, "liberty", "positive_liberty"),
            rec(1900, "liberty", "negative_liberty"),
        ]
        res = per_word_trends(records, min_denominator=1)
        self.assertEqual(set(res["by_word"]), {"freedom", "liberty"})
        self.assertIn("1800", res["frequency_mix"])


class HandoffTests(unittest.TestCase):
    def test_segments_split_by_source(self):
        def block(decade, pos, neg, source):
            return (
                [rec(decade, "freedom", "positive_liberty", source=source) for _ in range(pos)]
                + [rec(decade, "freedom", "negative_liberty", source=source) for _ in range(neg)]
            )

        # Each segment internally flat (share 0.2 historic, 0.4 parlparse);
        # the only movement is the level jump at the handoff.
        records = (
            block(1850, 20, 80, "S3V01") + block(1860, 20, 80, "S3V01")
            + block(1870, 20, 80, "S3V01")
            + block(1950, 40, 60, "") + block(1960, 40, 60, "")
            + block(1970, 40, 60, "")
        )
        res = handoff_controlled_trend(records, min_denominator=10)
        self.assertIsNotNone(res["by_source"]["historic_hansard"])
        self.assertIsNotNone(res["by_source"]["parlparse"])
        # Both segments flat internally; the jump is the level shift.
        self.assertGreater(res["source_dummy_model"]["handoff_level_shift"], 0.1)


class ClassifierErrorTests(unittest.TestCase):
    def test_directional_bias_by_era(self):
        rows = [
            {"year": 1850, "opus": "negative_liberty", "haiku_v2": "negative_liberty"},
            {"year": 1860, "opus": "positive_liberty", "haiku_v2": "positive_liberty"},
            {"year": 1950, "opus": "negative_liberty", "haiku_v2": "positive_liberty"},
            {"year": 1960, "opus": "positive_liberty", "haiku_v2": "positive_liberty"},
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "eval.json"
            p.write_text(json.dumps(rows))
            res = classifier_error_over_time(p, "opus", "haiku_v2")
        self.assertEqual(res["by_era"]["pre_1909_historic"]["agreement"], 1.0)
        self.assertGreater(res["by_era"]["post_1909_parlparse"]["directional_bias"], 0)


if __name__ == "__main__":
    unittest.main()
