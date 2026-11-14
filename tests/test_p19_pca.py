import numpy as np
import pytest

from conftest import load

m = load("p19_pca")


def reference(X, k):
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / (len(X) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order].T
    for i in range(len(vecs)):
        j = np.argmax(np.abs(vecs[i]))
        if vecs[i, j] < 0:
            vecs[i] = -vecs[i]
    return vecs[:k], vals[:k], vals[:k] / vals.sum()


@pytest.fixture(scope="module")
def X():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 4)) * np.array([5.0, 2.0, 1.0, 0.3])
    rot, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    return base @ rot + np.array([10.0, -3.0, 0.0, 7.0])


def test_matches_reference(X):
    pca = m.PCA(3)
    assert pca.fit(X) is pca
    comps, var, ratio = reference(X, 3)
    assert pca.components_.shape == (3, 4)
    assert np.allclose(pca.components_, comps, atol=1e-8)
    assert np.allclose(pca.explained_variance_, var)
    assert np.allclose(pca.explained_variance_ratio_, ratio)
    assert np.allclose(pca.mean_, X.mean(axis=0))


def test_orthonormal_and_sorted(X):
    pca = m.PCA(4).fit(X)
    C = pca.components_
    assert np.allclose(C @ C.T, np.eye(4), atol=1e-10)
    assert (np.diff(pca.explained_variance_) <= 1e-12).all()
    assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0)
    assert np.isrealobj(C)


def test_sign_convention(X):
    C = m.PCA(4).fit(X).components_
    for row in C:
        assert row[np.argmax(np.abs(row))] > 0


def test_transform_and_inverse(X):
    pca = m.PCA(4).fit(X)
    Z = pca.transform(X)
    assert Z.shape == (200, 4)
    assert np.allclose(pca.inverse_transform(Z), X)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Z.var(axis=0, ddof=1), pca.explained_variance_)
    assert np.allclose(m.PCA(2).fit_transform(X), m.PCA(2).fit(X).transform(X))


def test_reconstruction_error_shrinks_with_k(X):
    errs = []
    for k in (1, 2, 3, 4):
        pca = m.PCA(k).fit(X)
        errs.append(((pca.inverse_transform(pca.transform(X)) - X) ** 2).sum())
    assert errs[0] > errs[1] > errs[2] > errs[3] >= 0
    assert errs[3] == pytest.approx(0.0, abs=1e-8)


def test_simple_line():
    t = np.linspace(-1, 1, 21)
    X = np.c_[t, t] + 5.0
    pca = m.PCA(1).fit(X)
    assert np.allclose(pca.components_[0], [1 / np.sqrt(2), 1 / np.sqrt(2)])
    assert pca.explained_variance_ratio_[0] == pytest.approx(1.0)


def test_invalid(X):
    with pytest.raises(ValueError):
        m.PCA(0).fit(X)
    with pytest.raises(ValueError):
        m.PCA(5).fit(X)
    with pytest.raises(ValueError):
        m.PCA(1).fit(X[:1])
