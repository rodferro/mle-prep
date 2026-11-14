"""
Problem 20 — K-means clustering (numpy)

Implement `KMeans(n_clusters, max_iter=300, tol=1e-6, random_state=None)`.

    fit(X, init=None)    X: (n, d).
                         Starting centroids:
                           - if `init` is given (array of shape (n_clusters, d)), use it;
                           - otherwise k-means++ with rng = np.random.default_rng(random_state):
                             pick the first centroid uniformly at random from X, then each
                             next one from X with probability proportional to D(x)^2, the
                             squared distance to the nearest centroid chosen so far.
                         Then repeat (Lloyd's algorithm), at most max_iter times:
                           1. assign each point to its nearest centroid
                              (ties: the lowest centroid index);
                           2. move each centroid to the mean of its points. An EMPTY
                              cluster keeps its previous centroid;
                           3. stop if no centroid moved more than `tol` (Euclidean).
                         Store:
                           cluster_centers_   (n_clusters, d), the final centroids
                           labels_            (n,), each point's nearest final centroid
                           inertia_           float, sum of squared distances to it
                           n_iter_            number of iterations run
                         Return self.
    predict(X)           index of the nearest centroid for each row (same tie rule).

Validation: raise ValueError if n_clusters < 1, n_clusters > n, or init has the wrong
shape.

Vectorize the assignment step (distance matrix with numpy), no Python loop over points.

Follow-ups to discuss: why k-means++ beats random init, choosing k (elbow, silhouette),
k-means assumptions (spherical, similar-size clusters; scale matters), local minima
and n_init restarts, and mini-batch k-means for large data.
"""

import numpy as np


class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-6, random_state=None):
        raise NotImplementedError

    def fit(self, X, init=None) -> "KMeans":
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError
