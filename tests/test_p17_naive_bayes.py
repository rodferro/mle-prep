import math

import numpy as np
import pytest

from conftest import load

m = load("p17_naive_bayes")


def reference_proba(Xtr, ytr, Xte, var_smoothing=1e-9):
    """Plain-Python reference with the same definitions."""
    classes = sorted(set(ytr))
    d = len(Xtr[0])
    col_var = []
    for j in range(d):
        col = [r[j] for r in Xtr]
        mu = sum(col) / len(col)
        col_var.append(sum((v - mu) ** 2 for v in col) / len(col))
    eps = var_smoothing * max(col_var) if max(col_var) > 0 else var_smoothing
    params = {}
    for c in classes:
        rows = [r for r, lab in zip(Xtr, ytr) if lab == c]
        mean = [sum(r[j] for r in rows) / len(rows) for j in range(d)]
        var = [sum((r[j] - mean[j]) ** 2 for r in rows) / len(rows) + eps for j in range(d)]
        params[c] = (len(rows) / len(Xtr), mean, var)
    out = []
    for x in Xte:
        logs = []
        for c in classes:
            prior, mean, var = params[c]
            ll = math.log(prior)
            for j in range(d):
                ll += -0.5 * math.log(2 * math.pi * var[j]) - (x[j] - mean[j]) ** 2 / (2 * var[j])
            logs.append(ll)
        mx = max(logs)
        ex = [math.exp(v - mx) for v in logs]
        out.append([e / sum(ex) for e in ex])
    return classes, out


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([0, 0, 0], [1, 2, 0.5], (60, 3)), rng.normal([3, -1, 1], [0.5, 1, 1], (40, 3))])
    y = np.array(["neg"] * 60 + ["pos"] * 40)
    return X, y


def test_classes_and_fit_returns_self(blobs):
    X, y = blobs
    nb = m.GaussianNB()
    assert nb.fit(X, y) is nb
    assert list(nb.classes_) == ["neg", "pos"]


def test_proba_matches_reference(blobs):
    X, y = blobs
    Xte = np.array([[0, 0, 0], [3, -1, 1], [1.5, -0.5, 0.5], [10, 10, 10]], dtype=float)
    nb = m.GaussianNB().fit(X, y)
    classes, ref = reference_proba(X.tolist(), y.tolist(), Xte.tolist())
    proba = nb.predict_proba(Xte)
    assert proba.shape == (4, 2)
    assert np.allclose(proba, ref, atol=1e-9)
    assert nb.predict(Xte).tolist() == [classes[int(np.argmax(r))] for r in ref]


def test_rows_sum_to_one_and_no_nan_far_away(blobs):
    X, y = blobs
    nb = m.GaussianNB().fit(X, y)
    far = np.array([[1e4, -1e4, 1e4], [-500.0, 300.0, 1e3]])
    p = nb.predict_proba(far)
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1.0)


def test_zero_variance_feature_in_a_class():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [7.0, 1.0], [8.0, 2.0], [9.0, 3.0]])
    y = [0, 0, 0, 1, 1, 1]
    nb = m.GaussianNB().fit(X, y)
    p = nb.predict_proba([[2.0, 5.0], [8.0, 2.0]])
    assert np.isfinite(p).all()
    assert nb.predict([[2.0, 5.0], [8.0, 2.0]]).tolist() == [0, 1]


def test_all_features_constant_uses_var_smoothing():
    X = np.ones((4, 2))
    nb = m.GaussianNB(var_smoothing=1e-3).fit(X, [0, 0, 1, 1])
    p = nb.predict_proba([[1.0, 1.0]])
    assert np.allclose(p, [[0.5, 0.5]])


def test_priors_matter():
    X = np.array([[0.0], [0.0], [0.0], [1.0], [0.0], [1.0]])
    y = [0, 0, 0, 0, 1, 1]
    # 3 of 4 class-0 points are at 0; class 1 is split; prior 4:2 favours 0
    assert m.GaussianNB().fit(X, y).predict([[0.0]]).tolist() == [0]


def test_invalid():
    with pytest.raises(ValueError):
        m.GaussianNB().fit([[1.0], [2.0]], [0])
    with pytest.raises(ValueError):
        m.GaussianNB().fit(np.empty((0, 2)), [])
    nb = m.GaussianNB().fit([[1.0, 2.0], [2.0, 1.0]], [0, 1])
    with pytest.raises(ValueError):
        nb.predict([[1.0]])
