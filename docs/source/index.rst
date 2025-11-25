Apunim: Attributing polarization to sociodemographic groups
===========================================================


Introduction
============

Annotators disagree with each other all the time. The reasons may be anything 
from their understanding of the task and the guidelines, to socio-demographic
factors, ideology, personal opinions and experiences, to different expertise.
Disagreements can either occur in the details (e.g., a difference 
between a 4-star and 5-star rating), or as fundamental disagreements 
(polarization).

Disregarding polarization in annotations is scientifically and ethically unsound.
In tasks such as toxicity/hate speech detection it is outright 
counterproductive, since disregarding minority opinions makes systems biased
and fundamentally misconfigured. However, it is often difficult to understand
whether disagreement is caused by the (random) factors mentioned above, by 
mismatches in minority vs majority group opinions, or by ideology.

The Apunim (Aposteriori Unimodality) tool solves this problem. Using this 
python library, we can attribute polarization to each individual annotator
characteristic.


Installation
============
This library is available in PyPi::

   pip install apunim

For other installation options, consult the project's 
`Github repository <https://github.com/dimits-ts/apunim>`_.


Usage Examples
===============

Here's a small python snippet that estimates the polarization (nDFU) values
for three distinct annotation distributions:

.. code-block:: python

   import numpy as np
   from apunim import dfu

   rng = np.random.default_rng(42)
   data = rng.normal(loc=0, scale=1, size=1000)
   score = dfu(data, bins=10, normalized=True)
   print(f"Normal distribution polarization: {score:.4f}")


   mode1 = rng.normal(loc=-2, scale=0.3, size=500)
   mode2 = rng.normal(loc=2, scale=0.3, size=500)
   data = np.hstack([mode1, mode2])
   score = dfu(data, bins=10, normalized=True)
   print(f"Bimodal distribution polarization: {score:.4f}")


.. code-block:: text

   # Output:
   Normal distribution polarization: 0.0000
   Bimodal distribution polarization: 0.6024


And a snippet that investigates whether the polarization of a discussion of 
two comments is caused by gender:

.. code-block:: python

   import numpy as np
   from apunim import aposteriori_unimodality

   # Example discussion with two comments (c1, c2),
   # each annotated by two gender groups (A, B).

   # c1: polarized (A gives low scores, B gives high)
   c1_A = [1, 1, 2]
   c1_B = [5, 5, 4]

   # c2: also polarized but differently distributed
   c2_A = [2, 2, 3]
   c2_B = [4, 5, 5]

   annotations = c1_A + c1_B + c2_A + c2_B
   factor_group = ["A"] * len(c1_A) + ["B"] * len(c1_B) + \
                  ["A"] * len(c2_A) + ["B"] * len(c2_B)
   comment_group = ["c1"] * (len(c1_A) + len(c1_B)) + \
                     ["c2"] * (len(c2_A) + len(c2_B))

   result = aposteriori_unimodality(
         annotations,
         factor_group,
         comment_group,
         num_bins=5,
   )

   for group, res in result.items():
         print(f"Group: {group}")
         print(f"  Apunim: {res.apunim:.4f}")
         print(f"  p-value: {res.pvalue:.4f}")

.. code-block:: text

   # Output:
   Group: A
      Apunim: -1.3529
      p-value: 0.1114
   Group: B
      Apunim: -1.7778
      p-value: 0.1252



.. code-block:: text

   # Output:
   Normal distribution polarization: 0.0000
   Bimodal distribution polarization: 0.6024


Module Documentation
====================

This library offers two main functions: `dfu` which estimates polarization,
and `aposteriori_unimodality`, which attributes polarization to annotator 
groups.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: apunim
   :members:
   :undoc-members:
   :show-inheritance: