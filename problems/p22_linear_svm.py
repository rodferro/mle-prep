"""
Problem 22 — Linear SVM with hinge loss (numpy)

Implement `LinearSVM(lam=0.01, learning_rate=0.1, n_iters=1000)`, a soft-margin linear
SVM trained with BATCH subgradient descent. Labels are -1 / +1.

    Objective (minimize):
        J(w, b) = (lam / 2) * ||w||^2 + mean_i max(0, 1 - y_i (x_i · w + b))

    Subgradient, with "active" points = margin y_i (x_i · w + b) < 1 (strictly):
        dJ/dw = lam * w - Σ_{active} y_i x_i / n
        dJ/db =         - Σ_{active} y_i     / n

    fit(X, y)              Start from w = 0, b = 0; take exactly n_iters steps of
                           w -= learning_rate * dJ/dw ;  b -= learning_rate * dJ/db
                           (both gradients computed from the same w, b). Store coef_
                           (shape (d,)) and intercept_ (float). Return self.
    decision_function(X)   X @ coef_ + intercept_
    predict(X)             +1 where decision_function >= 0, else -1 (int array)
    objective(X, y)        J for the current parameters

Requirements
    - Vectorized with numpy: no Python loop over samples inside an iteration.
    - Raise ValueError if labels aren't all in {-1, +1}, len(X) != len(y), or X is empty.

Note: some tutorials write the decision function as w·x - b and update per sample
(SGD). This problem fixes one version so the result is deterministic.

Follow-ups to discuss: what lam controls (margin width vs violations; C = 1/(n*lam)),
support vectors (which points end up active), hinge vs logistic loss, the kernel trick
and why it needs the dual problem, and Pegasos (SGD with a 1/(lam*t) step size).
"""

import numpy as np


class LinearSVM:
    def __init__(self, lam: float = 0.01, learning_rate: float = 0.1, n_iters: int = 1000):
        raise NotImplementedError

    def fit(self, X, y) -> "LinearSVM":
        raise NotImplementedError

    def decision_function(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def objective(self, X, y) -> float:
        raise NotImplementedError
