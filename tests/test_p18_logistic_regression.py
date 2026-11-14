import warnings

import numpy as np
import pytest

from conftest import load

m = load("p18_logistic_regression")


def newton(X, y, l2, iters=50):
    """Exact minimizer of the regularized objective (reference)."""
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    theta = np.zeros(d + 1)
    reg = np.diag([l2] * d + [0.0])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(A @ theta)))
        g = A.T @ (p - y) / n + reg @ theta
        H = (A * (p * (1 - p))[:, None]).T @ A / n + reg
        theta -= np.linalg.solve(H, g)
    return theta[:d], theta[d]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    logits = X @ np.array([1.5, -2.0, 0.5]) + 0.3
    y = (rng.random(400) < 1 / (1 + np.exp(-logits))).astype(int)
    return X, y


def test_fit_returns_self_and_shapes(data):
    X, y = data
    clf = m.LogisticRegression(n_iters=10)
    assert clf.fit(X, y) is clf
    assert clf.coef_.shape == (3,)
    assert isinstance(clf.intercept_, float)


def test_converges_to_regularized_optimum(data):
    X, y = data
    clf = m.LogisticRegression(learning_rate=0.5, n_iters=3000, l2=0.1).fit(X, y)
    w, b = newton(X, y, l2=0.1)
    assert np.allclose(clf.coef_, w, atol=1e-3)
    assert clf.intercept_ == pytest.approx(b, abs=1e-3)


def test_unregularized_recovers_true_direction(data):
    X, y = data
    clf = m.LogisticRegression(learning_rate=0.5, n_iters=3000).fit(X, y)
    w, b = newton(X, y, l2=0.0)
    assert np.allclose(clf.coef_, w, atol=1e-2)


def test_regularization_shrinks_weights(data):
    X, y = data
    small = m.LogisticRegression(learning_rate=0.5, n_iters=2000, l2=0.0).fit(X, y)
    big = m.LogisticRegression(learning_rate=0.5, n_iters=2000, l2=1.0).fit(X, y)
    assert np.linalg.norm(big.coef_) < np.linalg.norm(small.coef_)


def test_objective_decreases(data):
    X, y = data
    clf = m.LogisticRegression(n_iters=0).fit(X, y)
    start = clf.objective(X, y)
    assert start == pytest.approx(np.log(2))
    clf = m.LogisticRegression(learning_rate=0.5, n_iters=200, l2=0.01).fit(X, y)
    assert clf.objective(X, y) < start


def test_predict_and_threshold(data):
    X, y = data
    clf = m.LogisticRegression(learning_rate=0.5, n_iters=500).fit(X, y)
    p = clf.predict_proba(X)
    assert p.shape == (400,) and ((p >= 0) & (p <= 1)).all()
    assert clf.predict(X).tolist() == (p >= 0.5).astype(int).tolist()
    assert clf.predict(X, threshold=0.9).sum() <= clf.predict(X).sum()
    assert (clf.predict(X) == y).mean() > 0.75


def test_numerically_stable_for_huge_inputs(data):
    X, y = data
    clf = m.LogisticRegression(learning_rate=0.5, n_iters=200).fit(X, y)
    huge = np.array([[1e6, -1e6, 1e6], [-1e6, 1e6, -1e6]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = clf.predict_proba(huge)
        obj = clf.objective(huge, [0, 1])          # confidently wrong on both -> huge but finite
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert np.isfinite(obj) and obj > 1000


def test_invalid():
    with pytest.raises(ValueError):
        m.LogisticRegression().fit([[1.0], [2.0]], [0, 2])
    with pytest.raises(ValueError):
        m.LogisticRegression().fit([[1.0], [2.0]], [0])
    with pytest.raises(ValueError):
        m.LogisticRegression().fit(np.empty((0, 1)), [])
