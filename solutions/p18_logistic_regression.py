"""
Reference solution — Problem 18 (logistic regression with L2).

Stability
- sigmoid(z) = 0.5 * (1 + tanh(z / 2)) is mathematically identical and never overflows:
  tanh saturates at ±1 instead of exp blowing up.
- Log loss per sample, written in terms of z = x·w + b:
      -y log p - (1-y) log(1-p) = log(1 + e^z) - y z = logaddexp(0, z) - y z
  np.logaddexp is stable for any z, so the objective is finite even when p rounds
  to exactly 0 or 1.

Training: batch gradient descent, fully vectorized. Each step is O(n * d).
With l2 > 0 the objective is strongly convex, so GD converges to the unique optimum
for a small enough learning rate. With l2 = 0 on separable data, ||w|| grows forever.
"""

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, n_iters: int = 1000, l2: float = 0.0):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.l2 = l2

    @staticmethod
    def _check(X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(X) == 0 or len(X) != len(y):
            raise ValueError("X and y must be non-empty and the same length")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("labels must be 0 or 1")
        return X.reshape(len(X), -1), y

    def fit(self, X, y) -> "LogisticRegression":
        X, y = self._check(X, y)
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        for _ in range(self.n_iters):
            err = _sigmoid(X @ w + b) - y
            grad_w = X.T @ err / n + self.l2 * w
            grad_b = err.mean()
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
        self.coef_, self.intercept_ = w, float(b)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        return _sigmoid(X @ self.coef_ + self.intercept_)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def objective(self, X, y) -> float:
        X, y = self._check(X, y)
        z = X @ self.coef_ + self.intercept_
        loss = np.mean(np.logaddexp(0.0, z) - y * z)
        return float(loss + 0.5 * self.l2 * self.coef_ @ self.coef_)
