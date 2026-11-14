"""
Problem 19 — Principal Component Analysis (numpy)

Implement `PCA(n_components)`.

    fit(X)               X: (n, d). Learn:
                           mean_                       shape (d,)
                           components_                 shape (k, d): orthonormal rows,
                                                       the top-k principal directions,
                                                       ordered by variance, descending
                           explained_variance_         shape (k,): variance along each
                                                       component (divide by n - 1)
                           explained_variance_ratio_   shape (k,): explained_variance_ /
                                                       total variance of X
                         Return self.
    transform(X)         (X - mean_) @ components_.T  -> shape (n, k)
    inverse_transform(Z) Z @ components_ + mean_      -> shape (n, d)
    fit_transform(X)     fit, then transform.

Sign convention
    A principal direction and its negative are equally valid. To make results unique:
    flip each component so that its entry with the largest absolute value is positive.

Requirements
    - Raise ValueError if n_components < 1 or n_components > min(n, d), or if X has
      fewer than 2 rows.
    - Results must be real numbers. Careful: np.linalg.eig on a covariance matrix can
      return complex values and unsorted eigenvalues. Use np.linalg.eigh (for symmetric
      matrices) or the SVD of the centered data.

Follow-ups to discuss: why SVD of the centered data is preferred over the covariance
matrix (numerical stability, no d x d matrix when d is huge), choosing k (explained
variance curve), PCA on unscaled features, and PCA leaking if fit on train + test.
"""

import numpy as np


class PCA:
    def __init__(self, n_components: int):
        raise NotImplementedError

    def fit(self, X) -> "PCA":
        raise NotImplementedError

    def transform(self, X) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, Z) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X) -> np.ndarray:
        raise NotImplementedError
