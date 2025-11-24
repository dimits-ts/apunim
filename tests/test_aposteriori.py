import unittest
import numpy as np
from apunim import aposteriori_unimodality, ApunimResult


class TestAposterioriUnimodality(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_basic_output_structure(self):
        # Use polarized (bimodal) annotations so comments are valid.
        # A-group: low cluster; B-group: high cluster.
        annotations = [1, 1, 1, 1, 1, 5, 5, 5, 5, 5]
        factor_group = ["A"] * 5 + ["B"] * 5
        comment_group = ["c1"] * 5 + ["c2"] * 5

        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, num_bins=5
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"A", "B"})

        for k, v in result.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, ApunimResult)
            self.assertIsInstance(v.apunim, float)
            self.assertIsInstance(v.pvalue, float)

    def test_empty_inputs(self):
        with self.assertRaises(ValueError):
            aposteriori_unimodality([], [], [], num_bins=5)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            aposteriori_unimodality([1, 2], ["A"], ["c1", "c2"], num_bins=5)

    def test_single_factor_group(self):
        # Implementation requires ≥2 groups
        annotations = [1, 2, 3]
        factor_group = ["solo"] * 3
        comment_group = ["c1", "c2", "c3"]
        with self.assertRaises(ValueError):
            aposteriori_unimodality(
                annotations, factor_group, comment_group, num_bins=5
            )

    def test_bimodal_partition_low_pvals(self):
        # Strong separation should create high apunim and low p-values
        annotations = [1] * 50 + [5] * 50
        factor_group = ["L"] * 50 + ["R"] * 50
        comment_group = ["c1", "c2"] * 50

        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, num_bins=5
        )

        for res in result.values():
            self.assertLess(res.pvalue, 0.05)

    def test_random_noise_high_pvals(self):
        # Random noise → apunim close to 0 → high p-values
        annotations = self.rng.normal(0, 1, 200).tolist()
        factor_group = ["A"] * 100 + ["B"] * 100
        comment_group = ["c1", "c2"] * 100

        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, num_bins=5
        )

        for res in result.values():
            self.assertGreater(res.pvalue, 0.05)

    def test_multiple_comments_aggregation(self):
        # Some comments polarized, some not → allow NaNs
        annotations = [1, 5, 1, 5, 1, 5, 2, 4, 2, 4]
        factor_group = ["A", "B"] * 5
        comment_group = [
            "c1",
            "c1",
            "c2",
            "c2",
            "c3",
            "c3",
            "c4",
            "c4",
            "c5",
            "c5",
        ]

        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, num_bins=5
        )

        self.assertEqual(set(result.keys()), {"A", "B"})
        for res in result.values():
            self.assertIsInstance(res, ApunimResult)
            self.assertTrue(np.isnan(res.pvalue) or 0 <= res.pvalue <= 1)

    def test_nan_annotations_handling(self):
        # Function should not crash. Result values may be nan.
        annotations = [1, 2, np.nan, 3, 4, 5, 1, 2, 3, 4]
        factor_group = ["A"] * 5 + ["B"] * 5
        comment_group = ["c1"] * 5 + ["c2"] * 5

        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, num_bins=5
        )

        self.assertEqual(set(result.keys()), {"A", "B"})
        for v in result.values():
            self.assertIsInstance(v, ApunimResult)

    def test_non_numeric_annotations_raise(self):
        annotations = ["a", "b", "c"]
        factor_group = ["A"] * 3
        comment_group = ["c1", "c2", "c3"]

        with self.assertRaises(Exception):
            aposteriori_unimodality(
                annotations, factor_group, comment_group, num_bins=5
            )


if __name__ == "__main__":
    unittest.main()
