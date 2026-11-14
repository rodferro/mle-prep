"""
Problem 16 — K-nearest neighbors classifier (numpy)

Implement `KNNClassifier(k=3, weighted=False)` with a scikit-learn style API.

    fit(X, y)       X: array-like (n_train, d); y: n_train labels (ints or strings).
                    Store the training data. Return self.
    predict(X)      X: array-like (n_test, d). Return a numpy array of n_test labels.

Rules (the tests check all of them)
    - Distance: Euclidean.
    - Neighbors: the k training points with the smallest distance. If distances tie,
      the point with the LOWER training index comes first.
    - weighted=False: each neighbor gets 1 vote.
      weighted=True:  each neighbor votes with weight 1 / distance. If any neighbor is
      at distance 0, only the distance-0 neighbors vote (1 vote each).
    - Vote ties: among the tied labels, pick the one whose closest neighbor comes first
      in the neighbor order above.

Validation
    - fit: raise ValueError if k < 1, k > n_train, or len(X) != len(y).
    - predict: raise ValueError if the number of features differs from training.

Performance: compute distances with numpy, not a Python loop over training points.
A test predicts 1,000 points against 3,000 training points with 16 features.
Hint: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b gives the whole distance matrix in
one matrix product.

Follow-ups to discuss: feature scaling (why KNN without it is mostly measuring the
largest-unit feature), choosing k, the curse of dimensionality, and approximate
nearest neighbors (KD-trees, HNSW, FAISS) for large datasets.
"""

import numpy as np


class KNNClassifier:
    def __init__(self, k: int = 3, weighted: bool = False):
        raise NotImplementedError

    def fit(self, X, y) -> "KNNClassifier":
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError
