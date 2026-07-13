"""Corner count distributions and corner-market probabilities."""

from __future__ import annotations

import numpy
from scipy import stats


def count_pmf_vector(mu, max_count, dispersion=None):
    """Per-side count PMFs using Poisson or NB2 variance."""
    mu = numpy.asarray(mu, dtype=float).reshape(-1, 1)
    counts = numpy.arange(max_count + 1).reshape(1, -1)
    if dispersion is None or dispersion <= 1e-9:
        pmf = stats.poisson.pmf(counts, mu)
    else:
        size_parameter = 1.0 / dispersion
        success_probability = size_parameter / (size_parameter + mu)
        pmf = stats.nbinom.pmf(counts, size_parameter, success_probability)
    return pmf / pmf.sum(axis=1, keepdims=True)


def convolve_pmfs(pmf_a, pmf_b):
    """Row-wise convolution for total-count probabilities."""
    pmf_a = numpy.asarray(pmf_a, dtype=float)
    pmf_b = numpy.asarray(pmf_b, dtype=float)
    return numpy.stack([
        numpy.convolve(pmf_a[row_index], pmf_b[row_index])
        for row_index in range(pmf_a.shape[0])
    ])


def pmf_over(pmf, line):
    """Probability that a count is above a half-unit line."""
    minimum_count = int(numpy.floor(line)) + 1
    return numpy.asarray(pmf, dtype=float)[:, minimum_count:].sum(axis=1)

