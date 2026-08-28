# ABOUTME: Tests for the shared trend/significance helpers in src/stats.py.
# ABOUTME: Pins the OLS contract the robustness and embedding analyses report to JSON.

import unittest

from src.stats import linear_trend, normal_two_sided_p


class TestNormalTwoSidedP(unittest.TestCase):
    """The reported p-values are what turn a slope into a claim, so the
    z→p mapping must stay pinned to the standard normal, not drift."""

    def test_known_quantiles(self):
        self.assertAlmostEqual(normal_two_sided_p(0.0), 1.0, places=10)
        self.assertAlmostEqual(normal_two_sided_p(1.0), 0.3173105079, places=10)
        self.assertAlmostEqual(normal_two_sided_p(1.96), 0.0499957903, places=10)
        self.assertAlmostEqual(normal_two_sided_p(5.0), 5.733e-07, places=10)

    def test_sign_of_z_does_not_matter(self):
        self.assertEqual(normal_two_sided_p(-2.5), normal_two_sided_p(2.5))


class TestLinearTrend(unittest.TestCase):
    """Slope is reported per century because every consumer compares
    centuries; an unscaled slope would silently shrink every effect 100x."""

    def setUp(self):
        self.decades = [1800, 1850, 1900, 1950, 2000]
        self.values = [0.10, 0.18, 0.31, 0.44, 0.55]

    def test_slope_is_per_century_not_per_year(self):
        result = linear_trend(self.decades, self.values)
        self.assertAlmostEqual(result["slope_per_century"], 0.232, places=6)

    def test_reports_error_and_significance(self):
        result = linear_trend(self.decades, self.values)
        self.assertAlmostEqual(result["std_error"], 0.010066, places=6)
        self.assertAlmostEqual(result["z"], 23.047, places=3)
        self.assertAlmostEqual(result["intercept_at_mean_decade"], 0.316, places=6)
        self.assertEqual(result["p_value_approx"], 0.0)

    def test_perfectly_flat_series_has_zero_slope(self):
        result = linear_trend(self.decades, [0.3] * 5)
        self.assertEqual(result["slope_per_century"], 0.0)
        self.assertEqual(result["std_error"], 0.0)
        self.assertEqual(result["z"], 0.0)

    def test_returns_none_when_a_trend_would_be_meaningless(self):
        # Fewer than three points cannot support a standard error.
        self.assertIsNone(linear_trend([1800, 1900], [0.1, 0.2]))
        # No spread on x: slope is undefined, not zero.
        self.assertIsNone(linear_trend([1900] * 4, [0.1, 0.2, 0.3, 0.4]))
        # Misaligned inputs are a caller bug, never a silent partial fit.
        self.assertIsNone(linear_trend(self.decades, self.values[:3]))


if __name__ == "__main__":
    unittest.main()
