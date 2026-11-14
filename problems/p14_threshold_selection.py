"""
Problem 14 — Choosing a decision threshold under costs and capacity

A fraud model outputs a score per transaction. Transactions with score >= threshold
are sent to manual review. Pick the threshold.

Part A — `best_threshold_by_value(scores, labels, amounts, review_cost)`
    Reviewing a transaction costs `review_cost`. Reviewing a fraud (label 1) prevents
    losing its `amount`. Not reviewing has value 0. So the value of a threshold t is:

        value(t) = Σ_{i: score_i >= t} (labels_i * amounts_i - review_cost)

    Candidate thresholds: every distinct score, plus float("inf") (review nothing,
    value 0.0). Return (best_threshold, best_value). If several thresholds tie on
    value, return the HIGHEST one (fewest reviews).
    Must be O(n log n): a test uses 200,000 transactions.

Part B — `threshold_for_budget(scores, max_alerts)`
    The review team can handle at most `max_alerts` transactions. Return
    (threshold, n_alerts) for the LOWEST threshold among the distinct scores such that
    #{score >= threshold} <= max_alerts. If even the highest score has more than
    `max_alerts` rows tied on it, return (float("inf"), 0).

Validation (both parts): raise ValueError on length mismatch, empty input, labels not
0/1, or negative review_cost / max_alerts.

Pure Python is fine.

Follow-ups to discuss: why the best threshold drifts when fraud rates or amounts
change, using expected value with calibrated probabilities instead of labels
(review if p * amount > cost), and how review capacity, customer friction and false
declines change the objective.
"""

from typing import Sequence, Tuple


def best_threshold_by_value(
    scores: Sequence[float],
    labels: Sequence[int],
    amounts: Sequence[float],
    review_cost: float,
) -> Tuple[float, float]:
    raise NotImplementedError


def threshold_for_budget(scores: Sequence[float], max_alerts: int) -> Tuple[float, int]:
    raise NotImplementedError
