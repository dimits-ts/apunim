# Apunim: Quantifying and attributing polarization to annotator groups.
# Copyright (C) 2026 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr

import unittest
import numpy as np
from apunim import aposteriori_unimodality, ApunimResult


class TestAposterioriUnimodality(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_basic_output_structure(self):
        # Define bimodal annotations per comment and factor group
        # c1: factor A=[1,1,1], B=[5,5,5] → strongly polarized
        # c2: factor A=[2,2,2], B=[4,4,4] → polarized
        annotations = [
            1,
            1,
            1,
            5,
            5,
            5,  # c1: A vs B
            2,
            2,
            2,
            4,
            4,
            4,  # c2: A vs B
        ]
        factor_group = ["A", "A", "A", "B", "B", "B"] * 2
        comment_group = ["c1"] * 6 + ["c2"] * 6

        # Shuffle annotations within each comment to avoid ordered sequences
        for c in set(comment_group):
            mask = [i for i, x in enumerate(comment_group) if x == c]
            subset = [annotations[i] for i in mask]
            self.rng.shuffle(subset)
            for idx, val in zip(mask, subset):
                annotations[idx] = val

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
        n_per_group = 25

        annotations = []
        factor_group = []
        comment_group = []

        for c in ["c1", "c2"]:
            for f in ["A", "B"]:
                if c == "c1" and f == "A":
                    vals = self.rng.choice([1, 5], size=n_per_group)  # bimodal
                elif c == "c1" and f == "B":
                    vals = self.rng.choice(
                        [2, 4], size=n_per_group
                    )  # shifted bimodal
                elif c == "c2" and f == "A":
                    vals = self.rng.choice([2, 4], size=n_per_group)
                else:  # c2, B
                    vals = self.rng.choice([1, 5], size=n_per_group)

                annotations.extend(vals.tolist())
                factor_group.extend([f] * n_per_group)
                comment_group.extend([c] * n_per_group)

        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, num_bins=5
        )

        for res in result.values():
            self.assertGreater(res.pvalue, 0.05)

    def test_multiple_comments_aggregation(self):
        # Some comments polarized, some not → allow NaNs
        # Each group (A/B) has 3 annotations per comment to satisfy the >= 3 threshold
        annotations = [
            1,
            1,
            1,
            5,
            5,
            5,  # c1: A=[1,1,1], B=[5,5,5] → polarized
            1,
            1,
            1,
            5,
            5,
            5,  # c2: A=[1,1,1], B=[5,5,5] → polarized
            2,
            2,
            2,
            4,
            4,
            4,  # c3: A=[2,2,2], B=[4,4,4] → mildly polarized
            3,
            3,
            3,
            3,
            3,
            3,  # c4: A=[3,3,3], B=[3,3,3] → not polarized
            2,
            2,
            2,
            3,
            3,
            3,  # c5: A=[2,2,2], B=[3,3,3] → not polarized
        ]
        factor_group = ["A", "A", "A", "B", "B", "B"] * 5
        comment_group = (
            ["c1"] * 6 + ["c2"] * 6 + ["c3"] * 6 + ["c4"] * 6 + ["c5"] * 6
        )
        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, num_bins=5
        )
        self.assertEqual(set(result.keys()), {"A", "B"})
        for res in result.values():
            self.assertIsInstance(res, ApunimResult)
            self.assertTrue(np.isnan(res.pvalue) or 0 <= res.pvalue <= 1)

    def test_nan_annotations_handling(self):
        # Function should not crash. Result values may be nan.
        # Each group has 3+ annotations per comment; one NaN is inserted to test filtering.
        annotations = [
            1,
            1,
            1,
            5,
            5,
            5,  # c1: A=[1,1,1], B=[5,5,5] → bimodal
            1,
            1,
            1,
            5,
            5,
            5,  # c2: A=[1,1,1], B=[5,5,5] → bimodal
        ]
        factor_group = ["A", "A", "A", "B", "B", "B"] * 2
        comment_group = ["c1"] * 6 + ["c2"] * 6
        # insert a NaN to test filtering; group A in c1 still has 2 valid annotations
        # but c2 remains fully valid to keep the test meaningful
        annotations[1] = np.nan
        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, num_bins=5
        )
        self.assertEqual(set(result.keys()), {"A", "B"})
        for v in result.values():
            self.assertIsInstance(v, ApunimResult)
            self.assertIsInstance(v.apunim, float)
            self.assertIsInstance(v.pvalue, float)

    def test_non_numeric_annotations_raise(self):
        annotations = ["a", "b", "c"]
        factor_group = ["A"] * 3
        comment_group = ["c1", "c2", "c3"]

        with self.assertRaises(Exception):
            aposteriori_unimodality(
                annotations, factor_group, comment_group, num_bins=5
            )

    def test_none_in_factor_group_single_comment(self):
        """
        Regression test: None in factor_group caused IndexError in _comment_is_valid
        due to groups array being shorter than annotations array after None filtering.
        """
        annotations = [
            1,
            1,
            1,
            5,
            5,  # c1: A=[1,1,1], B=[5,5]
            1,
            1,
            1,
            5,
            5,
            5,
        ]  # c2: A=[1,1,1], B=[5,5,5]
        factor_group = [
            "A",
            "A",
            "A",
            "B",
            None,  # c1: one None in factor_group
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
        ]
        comment_group = ["c1"] * 5 + ["c2"] * 6

        # Should not raise IndexError
        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, num_bins=5
        )
        self.assertIsInstance(result, dict)
        for v in result.values():
            self.assertIsInstance(v, ApunimResult)

    def test_nan_in_factor_group_single_comment(self):
        """
        Regression test: NaN float in factor_group caused IndexError in _comment_is_valid.
        NaN factor labels are treated the same as None by _is_not_none.
        """
        annotations = [
            1,
            1,
            1,
            5,
            5,  # c1: one NaN factor label
            1,
            1,
            1,
            5,
            5,
            5,
        ]  # c2: clean
        factor_group = [
            "A",
            "A",
            "A",
            "B",
            float("nan"),
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
        ]
        comment_group = ["c1"] * 5 + ["c2"] * 6

        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, num_bins=5
        )
        self.assertIsInstance(result, dict)
        for v in result.values():
            self.assertIsInstance(v, ApunimResult)

    def test_multiple_nones_in_factor_group(self):
        """
        Multiple None factor labels across different comments; each None widens
        the size gap between groups and annotations, making the crash more likely.
        """
        annotations = [1, 1, 2, 2, 1, 1, 5, 4, 4, 5, 5, 5]  # c1  # c2
        factor_group = [
            None,
            "A",
            "A",
            "B",
            None,
            "B",  # c1: 2 Nones
            "A",
            "A",
            "A",
            "A",
            "B",
            None,
        ]  # c2: 1 None
        comment_group = ["c1"] * 6 + ["c2"] * 6

        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, num_bins=5
        )
        self.assertIsInstance(result, dict)

    def test_none_in_factor_group_preserves_valid_results(self):
        """
        None labels should be silently dropped; the remaining annotations
        should still produce meaningful (non-NaN) apunim values when the
        surviving groups are strongly polarized and meet the >= 3 threshold.
        """
        # c1/c2: after dropping the None, A has 3 and B has 3 annotations → valid
        annotations = [
            1,
            1,
            1,
            5,
            5,
            5,
            float("nan"),  # c1: 7 entries, last is NaN annotation
            1,
            1,
            1,
            5,
            5,
            5,
            2,
        ]  # c2: 7 entries, last has None factor
        factor_group = [
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            None,
        ]
        comment_group = ["c1"] * 7 + ["c2"] * 7

        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, num_bins=5
        )
        self.assertEqual(set(result.keys()), {"A", "B"})
        for v in result.values():
            self.assertFalse(
                np.isnan(v.apunim),
                "Expected valid apunim after dropping None/NaN entries",
            )

    def test_none_only_comment_does_not_crash(self):
        """
        A comment where every factor label is None should be skipped entirely,
        not cause a crash. The other valid comment keeps the test from raising
        'No polarized comments found'.
        """
        annotations = [
            1,
            1,
            1,
            5,
            5,
            5,  # c1: all None factors
            1,
            1,
            1,
            5,
            5,
            5,
        ]  # c2: normal
        factor_group = [
            None,
            None,
            None,
            None,
            None,
            None,
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
        ]
        comment_group = ["c1"] * 6 + ["c2"] * 6

        result = aposteriori_unimodality(
            annotations, factor_group, comment_group, num_bins=5
        )
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
