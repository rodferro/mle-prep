"""
Reference solution — Problem 20 (K-means).

Assignment: squared-distance matrix (n, k) with broadcasting; argmax/argmin return
the FIRST minimum, which gives the lowest-index tie rule for free.

Update: np.add.at / bincount-style sums per cluster, divided by counts; clusters
with count 0 keep their old centroid (one common policy; others re-seed them at the
point farthest from its centroid).

k-means++: D^2 sampling spreads the initial centroids out. It gives an O(log k)
approximation guarantee in expectation and usually converges faster than uniform init.

Complexity: O(n * k * d) per iteration.
"""

import numpy as np


def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-6, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _kmeans_pp(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        centers = [X[rng.integers(len(X))]]
        d2 = _sq_dists(X, np.array(centers))[:, 0]
        for _ in range(1, self.n_clusters):
            total = d2.sum()
            if total == 0:  # all remaining points coincide with chosen centers
                nxt = X[rng.integers(len(X))]
            else:
                nxt = X[rng.choice(len(X), p=d2 / total)]
            centers.append(nxt)
            d2 = np.minimum(d2, ((X - nxt) ** 2).sum(axis=1))
        return np.array(centers, dtype=float)

    def fit(self, X, init=None) -> "KMeans":
        X = np.asarray(X, dtype=float)
        n = len(X)
        if self.n_clusters < 1 or self.n_clusters > n:
            raise ValueError("n_clusters must be between 1 and n_samples")
        if init is not None:
            C = np.array(init, dtype=float)
            if C.shape != (self.n_clusters, X.shape[1]):
                raise ValueError("init must have shape (n_clusters, n_features)")
        else:
            C = self._kmeans_pp(X)

        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            labels = np.argmin(_sq_dists(X, C), axis=1)
            new_C = C.copy()
            counts = np.bincount(labels, minlength=self.n_clusters)
            sums = np.zeros_like(C)
            np.add.at(sums, labels, X)
            nonempty = counts > 0
            new_C[nonempty] = sums[nonempty] / counts[nonempty, None]
            shift = np.sqrt(((new_C - C) ** 2).sum(axis=1)).max()
            C = new_C
            if shift <= self.tol:
                break

        d2 = _sq_dists(X, C)
        self.cluster_centers_ = C
        self.labels_ = np.argmin(d2, axis=1)
        self.inertia_ = float(d2[np.arange(n), self.labels_].sum())
        self.n_iter_ = n_iter
        return self

    def predict(self, X) -> np.ndarray:
        return np.argmin(_sq_dists(np.asarray(X, dtype=float), self.cluster_centers_), axis=1)
