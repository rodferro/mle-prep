"""
Problem 21 — Best split for a decision tree node

The heart of decision trees, random forests and gradient boosting: for one numeric
feature, find the threshold that best separates the labels.

Implement `best_split(x, y, criterion="gini") -> (threshold, impurity)`
    x: numeric feature values; y: class labels (any hashable). Same length, not sorted.

    - Candidate thresholds: midpoints (a + b) / 2 between consecutive DISTINCT sorted
      values of x. A split sends x <= threshold left and x > threshold right.
    - Score of a split = weighted impurity of the children:
          (n_left / n) * I(left) + (n_right / n) * I(right)
    - criterion "gini":    I = 1 - Σ_c p_c^2
      criterion "entropy": I = -Σ_c p_c log2(p_c)    (0 log 0 = 0)
    - Return the threshold with the LOWEST score and that score. If several tie (within
      1e-12), return the smallest threshold.
    - If x has only one distinct value there is no valid split: return
      (None, impurity of the whole node).

Also implement `impurity(y, criterion="gini") -> float` for a single node.

Validation: raise ValueError on empty input, length mismatch, or an unknown criterion.

Performance: O(n log n). Sort once, then sweep left to right updating class counts.
Re-counting both sides for every threshold is O(n^2) and too slow: a test uses 100,000
samples.

Follow-ups to discuss: information gain, why trees don't need feature scaling,
categorical features, histogram-based splits in LightGBM/XGBoost, and min_samples_leaf
and depth limits against overfitting.
"""

from typing import Hashable, Optional, Sequence, Tuple


def impurity(y: Sequence[Hashable], criterion: str = "gini") -> float:
    raise NotImplementedError


def best_split(
    x: Sequence[float],
    y: Sequence[Hashable],
    criterion: str = "gini",
) -> Tuple[Optional[float], float]:
    raise NotImplementedError
