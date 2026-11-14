"""
Reference solution — Problem 19 (PCA).

Via SVD of the centered data: Xc = U S V^T.
- Rows of V^T are the principal directions, already orthonormal and ordered by
  singular value (descending).
- Variance along component i = S_i^2 / (n - 1).
- Total variance = Σ of all S^2 / (n - 1) = sum of per-feature variances (ddof=1).

Why SVD rather than eig(cov)
- Never forms the d x d covariance matrix (matters when d is large).
- Better numerical accuracy: forming X^T X squares the condition number.
- np.linalg.eig (general matrices) can return complex values and unsorted output;
  np.linalg.eigh (symmetric) is fine too if you do use the covariance.

Complexity: O(n * d * min(n, d)) for the SVD.
"""

import numpy as np


class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X) -> "PCA":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be 2-D with at least 2 rows")
        n, d = X.shape
        if not 1 <= self.n_components <= min(n, d):
            raise ValueError("n_components must be between 1 and min(n_samples, n_features)")

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)

        comps = vt[: self.n_components].copy()
        # Sign convention: largest-|value| entry of each component is positive.
        idx = np.argmax(np.abs(comps), axis=1)
        signs = np.sign(comps[np.arange(len(comps)), idx])
        signs[signs == 0] = 1.0
        comps *= signs[:, None]

        var = s**2 / (n - 1)
        self.components_ = comps
        self.explained_variance_ = var[: self.n_components]
        total = var.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total if total > 0 else np.zeros(self.n_components)
        return self

    def transform(self, X) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z) -> np.ndarray:
        return np.asarray(Z, dtype=float) @ self.components_ + self.mean_

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)
