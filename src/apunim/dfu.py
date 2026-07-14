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

from collections.abc import Collection

import numpy as np
from numpy.typing import NDArray


# code adapted from John Pavlopoulos
# https://github.com/ipavlopoulos/ndfu/blob/main/src/__init__.py
def dfu(x: Collection[float], bins: int, normalized: bool = True) -> float:
    """
    Compute the Distance From Unimodality (DFU) for a sequence of annotations.

    DFU measures how much a distribution deviates from being unimodal. The
    normalized DFU (nDFU) rescales the value to the range [0, 1].

    - DFU/nDFU = 0 indicates a unimodal or flat distribution.
    - Higher DFU/nDFU values indicate stronger multimodality or polarization.
    - nDFU = 1 indicates the maximum possible polarization.

    :param x: Sequence of annotation values (e.g., ratings, scores). Values
        need not be discrete, but discrete annotations should use a number
        of bins equal to the number of distinct values.
    :type x: Collection[float]
    :param bins: Number of bins to use for histogramming. For discrete data,
        it is recommended to use the number of distinct annotation levels.
    :type bins: int
    :param normalized: If True, returns the normalized DFU (nDFU). If False,
        returns the raw DFU.
    :type normalized: bool
    :raises ValueError: If `x` is empty or`bins` < 2.
    :return: DFU or normalized DFU (nDFU) statistic for the sequence.
    :rtype: float

    .. note::
        DFU is computed based on the maximum difference between the histogram
        peak and its neighbors. For details on the methodology and usage, see
        the original paper:
        `Pavlopoulos and Likas 2024
        <https://aclanthology.org/2024.eacl-long.117/>`_.

    .. seealso::
        - :func:`aposteriori_unimodality` for testing group-level polarization
          using DFU/nDFU.

    .. rubric:: Credits
        Original code and concept adapted from John Pavlopoulos:
        https://github.com/ipavlopoulos/ndfu
    """
    if bins <= 1:
        raise ValueError("Number of bins must be at least two.")

    hist = _to_hist(x, bins=bins)

    max_value = np.max(hist)
    pos_max = np.argmax(hist)

    # right search
    right_diffs = hist[pos_max + 1:] - hist[pos_max:-1]
    max_rdiff = right_diffs.max(initial=0)

    # left search
    if pos_max > 0:
        left_diffs = hist[0:pos_max] - hist[1: pos_max + 1]
        max_ldiff = left_diffs[left_diffs > 0].max(initial=0)
    else:
        max_ldiff = 0

    max_diff = max(max_rdiff, max_ldiff)
    dfu_stat = max_diff / max_value if normalized else max_diff
    return float(dfu_stat)


def _to_hist(scores: Collection[float], bins: int) -> NDArray:
    """
    Creates a normalised histogram. Used for DFU calculation.
    :param: scores: the ratings (not necessarily discrete)
    :param: num_bins: the number of bins to create
    :param: normed: whether to normalise the counts or not, by default true
    :return: the histogram
    """
    scores_array = np.array(scores)
    if len(scores_array) == 0:
        raise ValueError("Annotation list can not be empty.")

    counts, _ = np.histogram(a=scores_array, bins=bins, density=True)
    return counts
