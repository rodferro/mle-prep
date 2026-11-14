import numpy as np
import pytest

from conftest import load

m = load("p22_linear_svm")


def reference_fit(X, y, lam, lr, n_iters):
    """Plain-Python version of the specified update rule."""
    n, d = len(X), len(X[0])
    w, b = [0.0] * d, 0.0
    for _ in range(n_iters):
        gw = [lam * wj for wj in w]
        gb = 0.0
        for xi, yi in zip(X, y):
            if yi * (sum(a * c for a, c in zip(xi, w)) + b) < 1:
                for j in range(d):
                    gw[j] -= yi * xi[j] / n
                gb -= yi / n
        w = [wj - lr * g for wj, g in zip(w, gw)]
        b -= lr * gb
    return w, b


@pytest.fixture(scope="module")
def separable():
    rng = np.random.default_rng(0)
    Xp = rng.normal([2.5, 2.5], 0.6, size=(60, 2))
    Xn = rng.normal([-2.5, -2.5], 0.6, size=(60, 2))
    X = np.vstack([Xp, Xn])
    y = np.array([1] * 60 + [-1] * 60)
    return X, y


def test_matches_specified_updates():
    X = [[1.0, 2.0], [2.0, 0.5], [-1.0, -1.5], [-2.0, 0.5], [0.5, -0.5]]
    y = [1, 1, -1, -1, 1]
    for n_iters in (1, 3, 50):
        clf = m.LinearSVM(lam=0.1, learning_rate=0.2, n_iters=n_iters).fit(X, y)
        w, b = reference_fit(X, y, 0.1, 0.2, n_iters)
        assert np.allclose(clf.coef_, w)
        assert clf.intercept_ == pytest.approx(b)


def test_fit_returns_self_and_shapes(separable):
    X, y = separable
    clf = m.LinearSVM(n_iters=5)
    assert clf.fit(X, y) is clf
    assert clf.coef_.shape == (2,)
    assert isinstance(clf.intercept_, float)


def test_separable_data(separable):
    X, y = separable
    clf = m.LinearSVM(lam=0.01, learning_rate=0.1, n_iters=1000).fit(X, y)
    assert (clf.predict(X) == y).all()
    pred = clf.predict(X)
    assert pred.dtype.kind == "i" and set(pred.tolist()) <= {-1, 1}
    assert np.allclose(clf.decision_function(X), X @ clf.coef_ + clf.intercept_)


def test_objective_decreases_and_lambda_shrinks_w(separable):
    X, y = separable
    start = m.LinearSVM(n_iters=0).fit(X, y)
    assert start.objective(X, y) == pytest.approx(1.0)      # w = 0: every hinge is 1
    trained = m.LinearSVM(lam=0.01, n_iters=500).fit(X, y)
    assert trained.objective(X, y) < 0.2
    strong = m.LinearSVM(lam=1.0, n_iters=500).fit(X, y)
    assert np.linalg.norm(strong.coef_) < np.linalg.norm(trained.coef_)


def test_predict_zero_is_positive():
    clf = m.LinearSVM(n_iters=0).fit([[1.0], [-1.0]], [1, -1])
    assert clf.predict([[3.0]]).tolist() == [1]      # decision 0 -> +1


def test_invalid():
    with pytest.raises(ValueError):
        m.LinearSVM().fit([[1.0], [2.0]], [0, 1])
    with pytest.raises(ValueError):
        m.LinearSVM().fit([[1.0], [2.0]], [1])
    with pytest.raises(ValueError):
        m.LinearSVM().fit(np.empty((0, 2)), [])
