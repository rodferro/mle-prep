import math
import random
import time

import pytest

from conftest import load

m = load("p21_best_split")


def gini(ys):
    n = len(ys)
    return 1 - sum((ys.count(c) / n) ** 2 for c in set(ys))


def entropy(ys):
    n = len(ys)
    return -sum((ys.count(c) / n) * math.log2(ys.count(c) / n) for c in set(ys))


def brute(x, y, criterion):
    f = gini if criterion == "gini" else entropy
    vals = sorted(set(x))
    best_t, best = None, math.inf
    for a, b in zip(vals, vals[1:]):
        t = (a + b) / 2
        left = [yy for xx, yy in zip(x, y) if xx <= t]
        right = [yy for xx, yy in zip(x, y) if xx > t]
        s = len(left) / len(y) * f(left) + len(right) / len(y) * f(right)
        if s < best - 1e-12:
            best_t, best = t, s
    return (best_t, best) if best_t is not None else (None, f(list(y)))


def test_impurity():
    assert m.impurity([1, 1, 1]) == pytest.approx(0.0)
    assert m.impurity([0, 1]) == pytest.approx(0.5)
    assert m.impurity(["a", "b", "c", "d"]) == pytest.approx(0.75)
    assert m.impurity([0, 1], "entropy") == pytest.approx(1.0)
    assert m.impurity([0, 0, 1, 1, 2, 2, 3, 3], "entropy") == pytest.approx(2.0)


def test_perfect_split():
    t, s = m.best_split([1, 2, 3, 10, 11, 12], [0, 0, 0, 1, 1, 1])
    assert t == pytest.approx(6.5)
    assert s == pytest.approx(0.0)


def test_unsorted_input_and_strings():
    t, s = m.best_split([10, 1, 12, 3, 2, 11], ["b", "a", "b", "a", "a", "b"], criterion="entropy")
    assert t == pytest.approx(6.5) and s == pytest.approx(0.0)


def test_duplicate_values_are_not_split():
    # x=2 appears with both labels: candidates are only 1.5 and 2.5
    x = [1, 2, 2, 3]
    y = [0, 0, 1, 1]
    t, s = m.best_split(x, y)
    assert t in (1.5, 2.5)
    assert s == pytest.approx(brute(x, y, "gini")[1])
    assert t == 1.5        # 1.5 and 2.5 tie (both 1/3) -> smallest threshold


def test_no_valid_split():
    t, s = m.best_split([5, 5, 5], [0, 1, 1])
    assert t is None
    assert s == pytest.approx(gini([0, 1, 1]))


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_matches_brute_force(criterion):
    rng = random.Random(0)
    for _ in range(40):
        n = rng.randrange(1, 40)
        x = [rng.randrange(0, 10) for _ in range(n)]
        y = [rng.choice("abc") for _ in range(n)]
        t, s = m.best_split(x, y, criterion)
        bt, bs = brute(x, y, criterion)
        assert t == bt
        assert s == pytest.approx(bs, abs=1e-9)


def test_invalid():
    with pytest.raises(ValueError):
        m.best_split([], [])
    with pytest.raises(ValueError):
        m.best_split([1, 2], [0])
    with pytest.raises(ValueError):
        m.best_split([1, 2], [0, 1], criterion="mse")
    with pytest.raises(ValueError):
        m.impurity([])


def test_performance():
    rng = random.Random(1)
    n = 100_000
    x = [rng.random() for _ in range(n)]
    y = [int(v > 0.6) if rng.random() < 0.9 else rng.randrange(3) for v in x]
    start = time.perf_counter()
    t, _ = m.best_split(x, y)
    m.best_split(x, y, "entropy")
    elapsed = time.perf_counter() - start
    assert 0.5 < t < 0.7
    assert elapsed < 10.0, f"too slow: {elapsed:.1f}s (needs O(n log n))"
