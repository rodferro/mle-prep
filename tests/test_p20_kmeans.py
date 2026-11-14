import math

import numpy as np
import pytest

from conftest import load

m = load("p20_kmeans")


def brute_lloyd(X, C, max_iter, tol):
    """Plain-Python Lloyd's with the same rules (reference)."""
    X = [list(r) for r in X]
    C = [list(c) for c in C]
    k = len(C)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        labels = [min(range(k), key=lambda j: (sum((a - b) ** 2 for a, b in zip(x, C[j])), j)) for x in X]
        new_C = []
        for j in range(k):
            pts = [x for x, lab in zip(X, labels) if lab == j]
            new_C.append([sum(col) / len(pts) for col in zip(*pts)] if pts else C[j])
        shift = max(math.dist(a, b) for a, b in zip(C, new_C))
        C = new_C
        if shift <= tol:
            break
    labels = [min(range(k), key=lambda j: (sum((a - b) ** 2 for a, b in zip(x, C[j])), j)) for x in X]
    inertia = sum(sum((a - b) ** 2 for a, b in zip(x, C[lab])) for x, lab in zip(X, labels))
    return C, labels, inertia, n_iter


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [8, 0], [0, 8], [8, 8]], dtype=float)
    X = np.vstack([rng.normal(c, 0.8, size=(50, 2)) for c in centers])
    return X, np.repeat(np.arange(4), 50)


def test_given_init_matches_reference(blobs):
    X, _ = blobs
    init = X[[0, 1, 2, 3]]          # all from the first blob: needs several iterations
    km = m.KMeans(4, max_iter=100, tol=1e-6)
    assert km.fit(X, init=init) is km
    C, labels, inertia, n_iter = brute_lloyd(X, init, 100, 1e-6)
    assert np.allclose(km.cluster_centers_, C)
    assert km.labels_.tolist() == labels
    assert km.inertia_ == pytest.approx(inertia)
    assert km.n_iter_ == n_iter


def test_max_iter_respected(blobs):
    X, _ = blobs
    init = X[[0, 1, 2, 3]]
    km = m.KMeans(4, max_iter=2).fit(X, init=init)
    C, labels, inertia, n_iter = brute_lloyd(X, init, 2, 1e-6)
    assert km.n_iter_ == 2
    assert np.allclose(km.cluster_centers_, C)


def test_empty_cluster_keeps_its_centroid():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    init = np.array([[0.5, 0.5], [100.0, 100.0]])
    km = m.KMeans(2).fit(X, init=init)
    assert np.allclose(km.cluster_centers_[1], [100.0, 100.0])
    assert np.allclose(km.cluster_centers_[0], [0.5, 0.5])
    assert km.labels_.tolist() == [0, 0, 0, 0]


def test_tie_goes_to_lowest_index():
    X = np.array([[0.0], [2.0], [1.0]])
    km = m.KMeans(2, max_iter=1).fit(X, init=[[0.0], [2.0]])
    # after one update centroids stay at 0 and 2 ; point 1.0 is equidistant -> cluster 0
    assert km.predict([[1.0]]).tolist() == [0]


def test_kmeans_pp_recovers_blobs(blobs):
    X, truth = blobs
    km = m.KMeans(4, random_state=0).fit(X)
    assert km.cluster_centers_.shape == (4, 2)
    # each true blob maps to exactly one cluster
    mapping = {}
    for t, lab in zip(truth, km.labels_):
        mapping.setdefault(t, set()).add(int(lab))
    assert all(len(v) == 1 for v in mapping.values())
    assert len({next(iter(v)) for v in mapping.values()}) == 4
    assert km.predict(X).tolist() == km.labels_.tolist()


def test_kmeans_pp_is_reproducible(blobs):
    X, _ = blobs
    a = m.KMeans(4, random_state=7).fit(X)
    b = m.KMeans(4, random_state=7).fit(X)
    assert np.allclose(a.cluster_centers_, b.cluster_centers_)


def test_invalid(blobs):
    X, _ = blobs
    with pytest.raises(ValueError):
        m.KMeans(0).fit(X)
    with pytest.raises(ValueError):
        m.KMeans(500).fit(X)
    with pytest.raises(ValueError):
        m.KMeans(2).fit(X, init=[[0.0, 0.0]])
