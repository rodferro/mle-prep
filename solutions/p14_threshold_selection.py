"""
Reference solution — Problem 14 (threshold selection).

Both parts: sort by score descending, then sweep the threshold downward one run of
tied scores at a time. Lowering the threshold past a run adds every row in it, so
the running total (value, or alert count) is updated in O(run length).
Total O(n log n) for the sort.

Part A: start with (inf, 0.0) = review nothing; replace only on a STRICTLY better
value. Since the sweep goes from high to low thresholds, ties keep the higher one.

Part B: counts only grow as the threshold drops, so stop at the first run that would
exceed the budget.

Talking points
- With calibrated probabilities you don't need labels at decision time: review when
  p * amount > review_cost (expected value per transaction).
- Capacity is usually the binding constraint in fraud ops; set the threshold from
  recent score distributions (expected daily volume above t) and monitor it daily.
"""

import math
from typing import Sequence, Tuple


def _check_lengths(*seqs: Sequence) -> None:
    n = len(seqs[0])
    if any(len(s) != n for s in seqs):
        raise ValueError("inputs must have the same length")
    if n == 0:
        raise ValueError("empty input")


def best_threshold_by_value(
    scores: Sequence[float],
    labels: Sequence[int],
    amounts: Sequence[float],
    review_cost: float,
) -> Tuple[float, float]:
    _check_lengths(scores, labels, amounts)
    if any(y not in (0, 1) for y in labels):
        raise ValueError("labels must be 0 or 1")
    if review_cost < 0:
        raise ValueError("review_cost must be >= 0")

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    best_t, best_v = math.inf, 0.0
    value = 0.0
    i = 0
    while i < len(order):
        t = scores[order[i]]
        while i < len(order) and scores[order[i]] == t:
            k = order[i]
            value += labels[k] * amounts[k] - review_cost
            i += 1
        if value > best_v:
            best_t, best_v = t, value
    return best_t, best_v


def threshold_for_budget(scores: Sequence[float], max_alerts: int) -> Tuple[float, int]:
    _check_lengths(scores)
    if max_alerts < 0:
        raise ValueError("max_alerts must be >= 0")

    ordered = sorted(scores, reverse=True)
    best_t, best_n = math.inf, 0
    i = 0
    while i < len(ordered):
        t = ordered[i]
        while i < len(ordered) and ordered[i] == t:
            i += 1
        if i > max_alerts:
            break
        best_t, best_n = t, i
    return best_t, best_n
