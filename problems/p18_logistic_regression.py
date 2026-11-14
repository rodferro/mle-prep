"""
Problem 18 — Logistic regression with L2 regularization (numpy)

Implement `LogisticRegression(learning_rate=0.1, n_iters=1000, l2=0.0)` for binary
labels 0/1, trained with batch gradient descent.

    Objective (minimize):
        J(w, b) = mean_i [ -y_i log p_i - (1 - y_i) log(1 - p_i) ] + (l2 / 2) * ||w||^2
        p_i = sigmoid(x_i · w + b)          (the bias b is NOT regularized)

    Gradients:
        dJ/dw = X^T (p - y) / n + l2 * w
        dJ/db = mean(p - y)

    fit(X, y)              Start from w = 0, b = 0 and take exactly n_iters steps.
                           Store coef_ (shape (d,)) and intercept_ (float). Return self.
    predict_proba(X)       1-D array of P(y = 1 | x), shape (n,).
    predict(X, threshold=0.5)   1-D int array: 1 where proba >= threshold, else 0.
    objective(X, y)        J(w, b) for the current parameters (float).

Requirements
    - Vectorized with numpy (no Python loop over samples).
    - Numerically stable: predict_proba must not emit overflow warnings or return NaN
      for huge |x · w + b| (a test turns warnings into errors). The naive
      1 / (1 + exp(-z)) overflows for z around -750.
    - objective must be finite even when some p is exactly 0 or 1 (clip, or use
      log(1 + exp(z)) written stably).
    - Raise ValueError if labels are not 0/1, len(X) != len(y), or X is empty.

Follow-ups to discuss: why the weights grow without bound on separable data when
l2 = 0, L1 vs L2, feature scaling and the learning rate, and Newton's method / L-BFGS
(what sklearn actually uses).
"""

import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, n_iters: int = 1000, l2: float = 0.0):
        raise NotImplementedError

    def fit(self, X, y) -> "LogisticRegression":
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        raise NotImplementedError

    def objective(self, X, y) -> float:
        raise NotImplementedError
