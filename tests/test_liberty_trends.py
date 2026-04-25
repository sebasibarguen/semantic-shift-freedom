import json
import tempfile
import unittest
from pathlib import Path

from src.corpus_manifest import build_manifest
from src.liberty_trends import aggregate_by_decade, run_analysis, weighted_linear_trend


def record(year, label, method_key="llm"):
    methods = {method_key: {"label": label}} if method_key else {}
    return {
        "id": f"{year}-{label}",
        "year": year,
        "sentence": "Test sentence.",
        "methods": methods,
    }


class LibertyTrendTests(unittest.TestCase):
    def test_aggregate_positive_share_uses_positive_negative_denominator(self):
        records = [
            record(1901, "positive_liberty"),
            record(1902, "negative_liberty"),
            record(1903, "negative_liberty"),
            record(1904, "ambiguous"),
            record(1905, "other"),
        ]

        by_decade = aggregate_by_decade(records)
        row = by_decade["1900"]

        self.assertEqual(row["denominators"]["positive_plus_negative"], 3)
        self.assertEqual(row["denominators"]["substantive"], 4)
        self.assertAlmostEqual(row["positive_share_of_positive_negative"]["point"], 1 / 3, places=6)
        self.assertAlmostEqual(row["positive_share_of_substantive"]["point"], 1 / 4, places=6)

    def test_weighted_trend_detects_increasing_proportion(self):
        points = [
            (1900, 0.10, 100),
            (1910, 0.20, 100),
            (1920, 0.30, 100),
            (1930, 0.40, 100),
        ]

        trend = weighted_linear_trend(points)

        self.assertIsNotNone(trend)
        self.assertGreater(trend["slope_per_century"], 0)
        self.assertEqual(trend["endpoint_change"], 0.3)

    def test_run_analysis_writes_primary_trend_from_sentence_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            rows = []
            rows.extend(record(1900, "positive_liberty") for _ in range(10))
            rows.extend(record(1900, "negative_liberty") for _ in range(90))
            rows.extend(record(1910, "positive_liberty") for _ in range(30))
            rows.extend(record(1910, "negative_liberty") for _ in range(70))
            rows.extend(record(1920, "positive_liberty") for _ in range(50))
            rows.extend(record(1920, "negative_liberty") for _ in range(50))
            (data_dir / "sentences_1900s.json").write_text(json.dumps(rows))

            results = run_analysis(data_dir, min_denominator=1)
            trend = results["trend_tests"]["positive_share_of_positive_negative"]

        self.assertGreater(trend["slope_per_century"], 0)
        self.assertEqual(trend["first_proportion"], 0.1)
        self.assertEqual(trend["last_proportion"], 0.5)


class CorpusManifestTests(unittest.TestCase):
    def test_manifest_detects_stale_from_to_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            data = [record(2000, "positive_liberty")]
            data[0]["methods"]["from_to"] = {"type": "to", "object": "act"}
            (data_dir / "sentences_2000s.json").write_text(json.dumps(data))
            (data_dir / "index.json").write_text(json.dumps({"total_sentences": 1}))

            manifest = build_manifest(data_dir)

        self.assertFalse(manifest["checks"]["from_to_removed"])
        self.assertEqual(manifest["scan"]["stale_method_keys"], ["from_to"])


if __name__ == "__main__":
    unittest.main()
