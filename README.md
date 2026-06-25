# Apunim: Attributing polarization to sociodemographic groups

Repository housing the implementation of the Aposteriori Unimodality ("Apunim") metric, which attributes polarization in annotation to specific annotator groups. See the accompanying paper for details:  [Quantifying and Attributing Polarization to Annotator Groups]([link pending](https://arxiv.org/abs/2602.06055)).

Includes two functions: `dfu`, which calculates the Distance From Unimodality (and also implements the normalized variant - nDFU), and `aposteriori-unimodality`, which calculates the apunim metric and associated pvalue.

See the [online documentation](https://dimits-ts.github.io/apunim/) for high-level notes, usage examples, and module documentation.

Installation is possible through PyPi: `pip install apunim`.