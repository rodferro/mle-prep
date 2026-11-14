"""
Reference solution — Problem 21 (best split).

Sweep
- Sort (x, y) pairs by x once: O(n log n).
- Start with every sample on the right. Move samples left one at a time, updating class
  counts in O(1). A threshold is only valid between two DIFFERENT x values, so score
  only where x[i] != x[i+1].
- Gini from counts: 1 - Σ c^2 / m^2. Keeping Σ c^2 updated incrementally
  (c -> c+1 adds 2c+1) makes each step O(1).
- Entropy needs O(k) per valid threshold (k = number of classes), which is fine for
  small k.

Total O(n log n + n * k). This is the same idea trees use per feature at every node;
LightGBM/XGBoost speed it up further by bucketing x into histograms.
"""

import math
from collections import Counter
from typing import Hashable, List, Optional, Sequence, Tuple

CRITERIA = ("gini", "entropy")


def _from_counts(counts: List[int], total: int, criterion: str) -> float:
    if total == 0:
        return 0.0
    if criterion == "gini":
        return 1.0 - sum(c * c for c in counts) / (total * total)
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


def impurity(y: Sequence[Hashable], criterion: str = "gini") -> float:
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}")
    if len(y) == 0:
        raise ValueError("empty input")
    return _from_counts(list(Counter(y).values()), len(y), criterion)


def best_split(
    x: Sequence[float],
    y: Sequence[Hashable],
    criterion: str = "gini",
) -> Tuple[Optional[float], float]:
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}")
    if len(x) == 0 or len(x) != len(y):
        raise ValueError("x and y must be non-empty and the same length")

    n = len(x)
    classes = {c: i for i, c in enumerate(dict.fromkeys(y))}
    order = sorted(range(n), key=lambda i: x[i])
    xs = [x[i] for i in order]
    ys = [classes[y[i]] for i in order]

    right = [0] * len(classes)
    for c in ys:
        right[c] += 1
    left = [0] * len(classes)
    parent = _from_counts(right, n, criterion)

    left_sq, right_sq = 0, sum(c * c for c in right)
    best_t, best_score = None, math.inf
    for i in range(n - 1):
        c = ys[i]
        left_sq += 2 * left[c] + 1
        right_sq -= 2 * right[c] - 1
        left[c] += 1
        right[c] -= 1
        if xs[i] == xs[i + 1]:
            continue
        nl, nr = i + 1, n - i - 1
        if criterion == "gini":
            score = (nl - left_sq / nl) / n + (nr - right_sq / nr) / n  # = nl/n*gini_l + nr/n*gini_r
        else:
            score = nl / n * _from_counts(left, nl, criterion) + nr / n * _from_counts(right, nr, criterion)
        if score < best_score - 1e-12:
            best_t, best_score = (xs[i] + xs[i + 1]) / 2, score

    if best_t is None:
        return None, parent
    return best_t, max(best_score, 0.0)
