Apunim: Attributing polarization to sociodemographic groups
===========================================================

.. image:: apunim_logo.svg

Annotators disagree with each other all the time. The reasons may be:

- Their understanding of the task and the guidelines
- Socio-demographic factors
- Ideology / personal opinions and experiences
- Different expertise

These disagreements can either occur in the details (e.g., a difference 
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


Examples and API doc 
============

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   usage
   api