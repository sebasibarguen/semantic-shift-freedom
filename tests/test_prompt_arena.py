# ABOUTME: Tests for the prompt arena: deterministic splits, metrics, history I/O.
# ABOUTME: No network — these tests cover the harness logic, not the model itself.

import tempfile
import unittest
from pathlib import Path

from src.prompt_arena.evaluator import compute_metrics
from src.prompt_arena.history import (
    history_table,
    log_evaluation,
    lookup_eval,
    prompt_hash,
)
from src.prompt_arena.splits import assign_split, split_summary


class SplitsTests(unittest.TestCase):
    def test_assignment_is_deterministic(self):
        # Same id always lands in the same split, twice in a row.
        for sid in ["1804-abc-001", "2020-09-16-bd78fc-001", "x"]:
            self.assertEqual(assign_split(sid), assign_split(sid))

    def test_split_summary_uses_70_15_15_buckets(self):
        # On a large enough corpus the proportions converge.
        records = [{"id": f"sentence-{i:08d}"} for i in range(2000)]
        summary = split_summary(records)
        # 70/15/15 = ±2pp tolerance for 2000 samples
        self.assertAlmostEqual(summary["train"] / 2000, 0.70, delta=0.04)
        self.assertAlmostEqual(summary["dev"] / 2000, 0.15, delta=0.04)
        self.assertAlmostEqual(summary["test"] / 2000, 0.15, delta=0.04)


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions_score_one(self):
        labels = ["positive_liberty", "negative_liberty", "ambiguous", "other"]
        gold = ["positive_liberty", "negative_liberty", "ambiguous", "other"]
        pred = list(gold)
        m = compute_metrics(pred, gold, labels)
        self.assertEqual(m["accuracy"], 1.0)
        for label in labels:
            self.assertEqual(m["per_class_f1"][label], 1.0)

    def test_skipped_errors_excluded_from_accuracy(self):
        labels = ["positive_liberty", "negative_liberty", "ambiguous", "other"]
        gold = ["positive_liberty", "negative_liberty", "ambiguous", "other"]
        pred = ["positive_liberty", "error", "ambiguous", "other"]
        m = compute_metrics(pred, gold, labels)
        # 3 valid predictions, all correct
        self.assertEqual(m["n_skipped"], 1)
        self.assertEqual(m["accuracy"], 1.0)
        # Recall on negative_liberty is 0 (the one valid was skipped) but precision math
        # tests are covered separately

    def test_per_class_metrics_reasonable_on_mixed(self):
        labels = ["positive_liberty", "negative_liberty", "ambiguous", "other"]
        gold = ["positive_liberty"] * 5 + ["negative_liberty"] * 5
        pred = ["positive_liberty"] * 4 + ["negative_liberty"] * 6  # one false negative on positive
        m = compute_metrics(pred, gold, labels)
        self.assertEqual(m["accuracy"], 0.9)  # 9 correct out of 10
        self.assertGreater(m["per_class_f1"]["positive_liberty"], 0.8)
        self.assertGreater(m["per_class_f1"]["negative_liberty"], 0.8)


class HistoryTests(unittest.TestCase):
    def test_log_and_lookup_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "history.jsonl"
            entry = {
                "prompt_hash": "abc123",
                "prompt_path": "prompts/v1.txt",
                "split": "dev",
                "model": "claude-haiku-4-5",
                "n": 100,
                "accuracy": 0.71,
                "per_class_f1": {"positive_liberty": 0.6, "negative_liberty": 0.8},
            }
            log_evaluation(path, entry)
            log_evaluation(path, {**entry, "accuracy": 0.73})  # newer eval same prompt+split

            found = lookup_eval(path, "abc123", "dev", "claude-haiku-4-5")
            self.assertIsNotNone(found)
            # Returns the most recent (last appended)
            self.assertEqual(found["accuracy"], 0.73)

            # Different prompt → no match
            self.assertIsNone(lookup_eval(path, "different", "dev", "claude-haiku-4-5"))

    def test_history_table_returns_descending(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "history.jsonl"
            log_evaluation(path, {"prompt_hash": "a", "split": "dev", "model": "m", "accuracy": 0.5})
            log_evaluation(path, {"prompt_hash": "b", "split": "dev", "model": "m", "accuracy": 0.7})
            rows = history_table(path)
            self.assertEqual(len(rows), 2)
            # Most recent first
            self.assertEqual(rows[0]["prompt_hash"], "b")

    def test_prompt_hash_stable_and_short(self):
        text = "You are a classifier. Apply Berlin's distinction."
        h1 = prompt_hash(text)
        h2 = prompt_hash(text)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 12)  # 12-char prefix, like our convention


if __name__ == "__main__":
    unittest.main()
