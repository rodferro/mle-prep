"""
Reference solution — Problem 13 (PSI).

- Edges are quantiles of the expected sample, so each expected bin holds roughly
  1/n_bins of the data. Deduplicate edges for heavily tied features.
- bisect_right(edges, v) = number of edges <= v = bin index. O(log b) per value.
- The eps floor avoids log(0) for empty bins. With very small samples, eps choice
  affects the value noticeably, which is one reason PSI needs enough data per bin.

Complexity: O(n log n) to sort expected, O((n + m) log b) to bin.

Reading PSI: it is symmetric-ish (a sum of KL(a||e) + KL(e||a) terms), has no
p-value, and the 0.1 / 0.25 thresholds are conventions. Track it over time and
alert on trends, not single values.
"""

import math
from bisect import bisect_right
from typing import List, Sequence


def _edges(expected: Sequence[float], n_bins: int) -> List[float]:
    s = sorted(expected)
    n = len(s)
    edges: List[float] = []
    for i in range(1, n_bins):
        e = s[(i * n) // n_bins]
        if not edges or e != edges[-1]:
            edges.append(e)
    return edges


def _fractions(values: Sequence[float], edges: List[float], eps: float) -> List[float]:
    counts = [0] * (len(edges) + 1)
    for v in values:
        counts[bisect_right(edges, v)] += 1
    total = len(values)
    return [max(c / total, eps) for c in counts]


def psi(expected: Sequence[float], actual: Sequence[float], n_bins: int = 10, eps: float = 1e-4) -> float:
    if len(expected) == 0 or len(actual) == 0:
        raise ValueError("samples must be non-empty")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    edges = _edges(expected, n_bins)
    e = _fractions(expected, edges, eps)
    a = _fractions(actual, edges, eps)
    return sum((ai - ei) * math.log(ai / ei) for ai, ei in zip(a, e))


def drift_level(value: float) -> str:
    if value < 0.1:
        return "none"
    if value < 0.25:
        return "moderate"
    return "significant"
