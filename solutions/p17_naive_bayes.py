"""
Reference solution — Problem 17 (Gaussian Naive Bayes).

- Fit is closed form: per-class priors, means and variances. O(n * d).
- Joint log likelihood for all classes at once with broadcasting:
      jll[i, c] = log prior_c - 0.5 Σ_j log(2π var_cj) - 0.5 Σ_j (x_ij - mean_cj)^2 / var_cj
  Shape (n, n_classes); no Python loop over samples.
- Posteriors: subtract each row's max before exp (log-sum-exp trick), so the largest
  term is exp(0) = 1 and nothing overflows or underflows to all zeros.
- Variance smoothing: adding epsilon to every variance keeps a zero-variance feature
  from dividing by zero. sklearn uses the same rule (var_smoothing * max variance).
"""

import numpy as np


class GaussianNB:
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y) -> "GaussianNB":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if len(X) == 0 or len(X) != len(y):
            raise ValueError("X and y must be non-empty and the same length")
        X = X.reshape(len(X), -1)

        self.classes_ = np.unique(y)
        max_var = X.var(axis=0).max()
        eps = self.var_smoothing * max_var if max_var > 0 else self.var_smoothing

        k, d = len(self.classes_), X.shape[1]
        self.theta_ = np.zeros((k, d))
        self.var_ = np.zeros((k, d))
        self.class_prior_ = np.zeros(k)
        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[i] = Xc.mean(axis=0)
            self.var_[i] = Xc.var(axis=0) + eps
            self.class_prior_[i] = len(Xc) / len(X)
        return self

    def _joint_log_likelihood(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        if X.shape[1] != self.theta_.shape[1]:
            raise ValueError("number of features differs from training data")
        log_norm = -0.5 * np.log(2 * np.pi * self.var_).sum(axis=1)          # (k,)
        sq = ((X[:, None, :] - self.theta_[None, :, :]) ** 2 / self.var_[None, :, :]).sum(axis=2)  # (n, k)
        return np.log(self.class_prior_)[None, :] + log_norm[None, :] - 0.5 * sq

    def predict_proba(self, X) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        jll -= jll.max(axis=1, keepdims=True)
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]
