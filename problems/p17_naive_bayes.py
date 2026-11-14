"""
Problem 17 — Gaussian Naive Bayes (numpy)

Implement `GaussianNB(var_smoothing=1e-9)`.

    fit(X, y)          Learn, for each class c (sorted unique labels -> self.classes_):
                         prior_c  = fraction of samples in class c
                         mean_c   = per-feature mean of class c
                         var_c    = per-feature variance of class c (divide by n_c)
                                    + epsilon
                       epsilon = var_smoothing * (largest per-feature variance of the
                       whole X); if that is 0, use epsilon = var_smoothing.
                       Return self.
    predict_proba(X)   Array (n, n_classes): posterior P(class | x), columns in the
                       order of self.classes_. Each row sums to 1.
    predict(X)         Array of labels: the class with the highest posterior.
                       (Ties: the first class in self.classes_.)

The model ("naive" = features independent given the class):
    log P(c | x) = log prior_c + Σ_j log N(x_j; mean_cj, var_cj) - log Z
    log N(x; m, v) = -0.5 * log(2π v) - (x - m)^2 / (2 v)

Requirements
    - Work in log space. Multiplying many small densities underflows to 0.
    - predict_proba must not return NaN/inf even for points far from all classes.
      Hint: normalize with log-sum-exp (subtract the row max before exponentiating).
    - A feature that is constant within a class must not crash (that's what the
      epsilon is for).
    - Labels can be any sortable type (ints, strings).
    - Raise ValueError if len(X) != len(y) or X is empty; and in predict / predict_proba
      if the number of features differs from training.

Follow-ups to discuss: why NB's probabilities are badly calibrated but its ranking is
often fine, multinomial/Bernoulli NB for text, and handling a category never seen
with a class (Laplace smoothing).
"""

import numpy as np


class GaussianNB:
    def __init__(self, var_smoothing: float = 1e-9):
        raise NotImplementedError

    def fit(self, X, y) -> "GaussianNB":
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError
