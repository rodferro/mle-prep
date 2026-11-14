import math
import time

import numpy as np
import pytest

from conftest import load

m = load("p16_knn")


def brute_predict(Xtr, ytr, Xte, k, weighted):
    out = []
    for x in Xte:
        d = [math.dist(x, t) for t in Xtr]
        order = sorted(range(len(Xtr)), key=lambda i: (d[i], i))[:k]
        if weighted:
            zeros = [i for i in order if d[i] == 0]
            weights = {i: 1.0 for i in zeros} if zeros else {i: 1 / d[i] for i in order}
        else:
            weights = {i: 1.0 for i in order}
        votes, first = {}, {}
        for pos, i in enumerate(order):
            if i in weights:
                votes[ytr[i]] = votes.get(ytr[i], 0.0) + weights[i]
                first.setdefault(ytr[i], pos)
        best = max(votes.values())
        out.append(min((lab for lab, v in votes.items() if v == best), key=lambda lab: first[lab]))
    return out


def test_simple_majority():
    X = [[0, 0], [1, 0], [0, 1], [10, 10], [11, 10]]
    y = ["a", "a", "a", "b", "b"]
    clf = m.KNNClassifier(k=3).fit(X, y)
    assert clf.predict([[0.5, 0.5], [10, 9]]).tolist() == ["a", "b"]
    assert isinstance(clf.predict([[0, 0]]), np.ndarray)


def test_fit_returns_self():
    clf = m.KNNClassifier(k=1)
    assert clf.fit([[0.0]], [1]) is clf


def test_distance_tie_prefers_lower_index():
    # query at 0: points at -1 (index 1) and +1 (index 0) are equidistant; k=1 -> index 0
    X = [[1.0], [-1.0], [5.0]]
    y = [7, 8, 9]
    assert m.KNNClassifier(k=1).fit(X, y).predict([[0.0]]).tolist() == [7]


def test_vote_tie_goes_to_closest_label():
    X = [[1.0], [2.0], [3.0], [4.0]]
    y = ["far", "near", "far", "near"]
    # query 2.1: neighbors in order 2 (near), 3 (far), 1 (far), 4 (near) -> with k=2: 1-1 tie -> "near"
    assert m.KNNClassifier(k=2).fit(X, y).predict([[2.1]]).tolist() == ["near"]
    # k=4: 2-2 tie, "near" is closest -> "near"
    assert m.KNNClassifier(k=4).fit(X, y).predict([[2.1]]).tolist() == ["near"]


def test_weighted_voting():
    X = [[0.0], [3.0], [3.5]]
    y = [1, 2, 2]
    # query 0.5: unweighted k=3 -> label 2 (2 votes) ; weighted: 1/0.5=2 vs 1/2.5+1/3=0.73 -> label 1
    assert m.KNNClassifier(k=3).fit(X, y).predict([[0.5]]).tolist() == [2]
    assert m.KNNClassifier(k=3, weighted=True).fit(X, y).predict([[0.5]]).tolist() == [1]


def test_weighted_exact_match_wins():
    X = [[0.0], [0.0], [1.0], [1.0], [1.0]]
    y = [1, 2, 3, 3, 3]
    # query exactly at 0: only the two distance-0 points vote (tie 1 vs 2 -> lower index first: label 1)
    assert m.KNNClassifier(k=5, weighted=True).fit(X, y).predict([[0.0]]).tolist() == [1]


@pytest.mark.parametrize("weighted", [False, True])
def test_matches_brute_force(weighted):
    rng = np.random.default_rng(0)
    Xtr = rng.integers(0, 6, size=(120, 3)).astype(float)   # integer grid -> many exact ties
    ytr = rng.integers(0, 4, size=120)
    Xte = rng.integers(0, 6, size=(60, 3)).astype(float)
    for k in (1, 4, 7):
        got = m.KNNClassifier(k=k, weighted=weighted).fit(Xtr, ytr).predict(Xte).tolist()
        assert got == brute_predict(Xtr.tolist(), ytr.tolist(), Xte.tolist(), k, weighted)


def test_invalid():
    with pytest.raises(ValueError):
        m.KNNClassifier(k=0).fit([[0.0]], [1])
    with pytest.raises(ValueError):
        m.KNNClassifier(k=3).fit([[0.0], [1.0]], [1, 2])
    with pytest.raises(ValueError):
        m.KNNClassifier(k=1).fit([[0.0], [1.0]], [1])
    clf = m.KNNClassifier(k=1).fit([[0.0, 1.0]], [1])
    with pytest.raises(ValueError):
        clf.predict([[0.0, 1.0, 2.0]])


def test_performance():
    rng = np.random.default_rng(1)
    Xtr = rng.normal(size=(3000, 16))
    ytr = rng.integers(0, 5, 3000)
    Xte = rng.normal(size=(1000, 16))
    start = time.perf_counter()
    pred = m.KNNClassifier(k=5).fit(Xtr, ytr).predict(Xte)
    elapsed = time.perf_counter() - start
    assert len(pred) == 1000
    assert elapsed < 10.0, f"too slow: {elapsed:.1f}s (vectorize the distances)"
