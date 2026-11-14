"""
Problem 15 — Probability calibration: reliability bins, ECE, Platt scaling

A model is calibrated if, among cases where it predicts 0.8, about 80% are positive.
That matters whenever probabilities feed a decision (expected-loss thresholds,
pricing, combining models).

Part A — `reliability_bins(probs, labels, n_bins=10) -> list[dict]`
    Equal-width bins over [0, 1]: bin i covers [i/n_bins, (i+1)/n_bins), except the
    last bin, which also includes 1.0. Bin index = min(int(p * n_bins), n_bins - 1).
    Return one dict per NON-EMPTY bin, in bin order:
        {"bin": i, "count": c, "mean_pred": average prob, "frac_pos": fraction of labels == 1}

Part B — `expected_calibration_error(probs, labels, n_bins=10) -> float`
    ECE = Σ_bins (count / N) * |frac_pos - mean_pred|

Part C — `fit_platt(scores, labels, learning_rate=0.1, n_iters=2000) -> (a, b)`
    Platt scaling maps a raw score s to p = sigmoid(a * s + b). Fit a and b by
    minimizing mean log loss with batch gradient descent, starting from a = 1, b = 0.
        dL/da = mean((p - y) * s),   dL/db = mean(p - y)
    Use a numerically stable sigmoid (no overflow for large |a*s + b|).

    `apply_platt(scores, a, b) -> list[float]` returns the calibrated probabilities.

Validation: raise ValueError on length mismatch, empty input, labels not 0/1, or (for
Parts A/B) probabilities outside [0, 1].

Pure Python is fine (math module).

Follow-ups to discuss: isotonic regression vs Platt (and overfitting with little
data), calibrating on a held-out set (never the training set), class re-weighting or
downsampling breaking calibration (and how to correct it), and that calibration
doesn't change ranking metrics like AUC.
"""

from typing import Dict, List, Sequence, Tuple


def reliability_bins(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> List[Dict]:
    raise NotImplementedError


def expected_calibration_error(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    raise NotImplementedError


def fit_platt(
    scores: Sequence[float],
    labels: Sequence[int],
    learning_rate: float = 0.1,
    n_iters: int = 2000,
) -> Tuple[float, float]:
    raise NotImplementedError


def apply_platt(scores: Sequence[float], a: float, b: float) -> List[float]:
    raise NotImplementedError
