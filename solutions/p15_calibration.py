"""
Reference solution — Problem 15 (calibration).

Reliability bins / ECE: one pass to accumulate count, sum of probs and sum of labels
per bin. ECE weights each bin's |accuracy - confidence| gap by its share of samples.

Platt scaling: logistic regression on a single feature (the raw score). The log-loss
gradient for a sigmoid output is simply (p - y) times the input, so batch gradient
descent is a few lines. The loss is convex, so GD reaches the global optimum with a
small enough learning rate.

Numerically stable sigmoid: for z >= 0 use 1 / (1 + e^-z); for z < 0 use
e^z / (1 + e^z). Neither branch ever computes e^(large positive), so nothing overflows.
"""

import math
from typing import Dict, List, Sequence, Tuple


def _validate(xs: Sequence[float], labels: Sequence[int], probs: bool) -> None:
    if len(xs) != len(labels):
        raise ValueError("inputs must have the same length")
    if len(xs) == 0:
        raise ValueError("empty input")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("labels must be 0 or 1")
    if probs and any(not 0.0 <= p <= 1.0 for p in xs):
        raise ValueError("probabilities must be in [0, 1]")


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def reliability_bins(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> List[Dict]:
    _validate(probs, labels, probs=True)
    count = [0] * n_bins
    sum_p = [0.0] * n_bins
    sum_y = [0] * n_bins
    for p, y in zip(probs, labels):
        b = min(int(p * n_bins), n_bins - 1)
        count[b] += 1
        sum_p[b] += p
        sum_y[b] += y
    return [
        {"bin": b, "count": count[b], "mean_pred": sum_p[b] / count[b], "frac_pos": sum_y[b] / count[b]}
        for b in range(n_bins)
        if count[b]
    ]


def expected_calibration_error(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    bins = reliability_bins(probs, labels, n_bins)
    n = len(probs)
    return sum(b["count"] / n * abs(b["frac_pos"] - b["mean_pred"]) for b in bins)


def fit_platt(
    scores: Sequence[float],
    labels: Sequence[int],
    learning_rate: float = 0.1,
    n_iters: int = 2000,
) -> Tuple[float, float]:
    _validate(scores, labels, probs=False)
    n = len(scores)
    a, b = 1.0, 0.0
    for _ in range(n_iters):
        grad_a = grad_b = 0.0
        for s, y in zip(scores, labels):
            err = _sigmoid(a * s + b) - y
            grad_a += err * s
            grad_b += err
        a -= learning_rate * grad_a / n
        b -= learning_rate * grad_b / n
    return a, b


def apply_platt(scores: Sequence[float], a: float, b: float) -> List[float]:
    return [_sigmoid(a * s + b) for s in scores]
