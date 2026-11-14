"""
Reference solution — Problem 16 (KNN).

Distances
- The expansion ||a||^2 + ||b||^2 - 2 a·b gives the whole matrix with one matmul and is
  the fastest, but it adds rounding error, so two points at the same true distance can
  come out slightly different, which breaks exact tie rules.
- Here: exact (a - b)^2 sums, computed in chunks of test rows so memory stays bounded
  (chunk x n_train x d floats at a time). Still fully vectorized inside a chunk.

Neighbor order: a stable argsort along each row keeps lower training indices first
among equal distances.

Voting: walk neighbors in order, adding weights to a dict. Python dicts keep insertion
order, so on a tie the label seen first (closest) wins when we only replace the best
on a STRICTLY larger weight.

Complexity: O(n_test * n_train * d) for distances + O(n_test * n_train log n_train) for
the sorts (np.argpartition would make selection O(n_train), but then ties need care).
"""

from typing import Dict

import numpy as np


class KNNClassifier:
    def __init__(self, k: int = 3, weighted: bool = False):
        self.k = k
        self.weighted = weighted

    def fit(self, X, y) -> "KNNClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if self.k < 1 or self.k > len(X):
            raise ValueError("k must be between 1 and the number of training samples")
        self.X_ = X.reshape(len(X), -1)
        self.y_ = y
        return self

    def _neighbors(self, X: np.ndarray, chunk: int = 128):
        idx_out = np.empty((len(X), self.k), dtype=int)
        dist_out = np.empty((len(X), self.k), dtype=float)
        for start in range(0, len(X), chunk):
            block = X[start : start + chunk]
            d2 = ((block[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=2)
            order = np.argsort(d2, axis=1, kind="stable")[:, : self.k]
            idx_out[start : start + chunk] = order
            dist_out[start : start + chunk] = np.sqrt(np.take_along_axis(d2, order, axis=1))
        return idx_out, dist_out

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features differs from training data")
        idx, dist = self._neighbors(X)
        out = []
        for neigh, d in zip(idx, dist):
            if self.weighted:
                if (d == 0).any():
                    weights = (d == 0).astype(float)
                else:
                    weights = 1.0 / d
            else:
                weights = np.ones(len(neigh))
            votes: Dict = {}
            for j, w in zip(neigh, weights):
                if w > 0:
                    label = self.y_[j]
                    votes[label] = votes.get(label, 0.0) + w
            best_label, best_w = None, -1.0
            for label, w in votes.items():
                if w > best_w:
                    best_label, best_w = label, w
            out.append(best_label)
        return np.array(out, dtype=self.y_.dtype)
