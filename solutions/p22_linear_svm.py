"""
Reference solution — Problem 22 (linear SVM, batch subgradient descent).

The hinge loss max(0, 1 - m) isn't differentiable at m = 1, so we use a subgradient:
-y x for points with margin m < 1 ("active": misclassified or inside the margin),
0 otherwise. Vectorized, one step costs O(n * d).

Points that end up active at the optimum are the support vectors: only they determine
w. The rest could be removed without changing the solution.

lam trades margin width (small ||w||) against violations; sklearn's C corresponds to
1 / (n * lam). A fixed learning rate means subgradient descent hovers near the optimum
instead of converging exactly; Pegasos uses a decaying 1 / (lam * t) step for that.
"""

import numpy as np


class LinearSVM:
    def __init__(self, lam: float = 0.01, learning_rate: float = 0.1, n_iters: int = 1000):
        self.lam = lam
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    @staticmethod
    def _check(X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(X) == 0 or len(X) != len(y):
            raise ValueError("X and y must be non-empty and the same length")
        if not np.isin(y, (-1.0, 1.0)).all():
            raise ValueError("labels must be -1 or +1")
        return X.reshape(len(X), -1), y

    def fit(self, X, y) -> "LinearSVM":
        X, y = self._check(X, y)
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        for _ in range(self.n_iters):
            active = y * (X @ w + b) < 1
            ya = y[active]
            grad_w = self.lam * w - X[active].T @ ya / n
            grad_b = -ya.sum() / n
            w = w - self.learning_rate * grad_w
            b = b - self.learning_rate * grad_b
        self.coef_, self.intercept_ = w, float(b)
        return self

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        return X @ self.coef_ + self.intercept_

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1).astype(int)

    def objective(self, X, y) -> float:
        X, y = self._check(X, y)
        hinge = np.maximum(0.0, 1 - y * (X @ self.coef_ + self.intercept_))
        return float(0.5 * self.lam * self.coef_ @ self.coef_ + hinge.mean())
