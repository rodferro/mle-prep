import random
import time

import pytest

from conftest import load

m = load("p11_roc_auc")


def brute_auc(y, s):
    pos = [si for yi, si in zip(y, s) if yi == 1]
    neg = [si for yi, si in zip(y, s) if yi == 0]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def brute_ap(y, s):
    n_pos = sum(y)
    ap, prev_r = 0.0, 0.0
    for t in sorted(set(s), reverse=True):
        pred = [yi for yi, si in zip(y, s) if si >= t]
        tp = sum(pred)
        r, p = tp / n_pos, tp / len(pred)
        ap += (r - prev_r) * p
        prev_r = r
    return ap


def test_perfect_and_inverted():
    assert m.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert m.roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)


def test_all_tied_is_half():
    assert m.roc_auc([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)


def test_small_known_value():
    # pairs: (0.8 vs 0.1) win, (0.8 vs 0.4) win, (0.35 vs 0.1) win, (0.35 vs 0.4) loss -> 3/4
    assert m.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_auc_with_ties_matches_brute_force():
    rng = random.Random(1)
    for _ in range(20):
        n = rng.randrange(5, 60)
        y = [rng.randrange(2) for _ in range(n)]
        if len(set(y)) < 2:
            y[0], y[1] = 0, 1
        s = [rng.randrange(0, 8) / 8 for _ in range(n)]  # lots of ties
        assert m.roc_auc(y, s) == pytest.approx(brute_auc(y, s))


def test_auc_is_scale_invariant():
    y = [0, 1, 0, 1, 1, 0]
    s = [0.2, 0.7, 0.4, 0.9, 0.3, 0.1]
    assert m.roc_auc(y, s) == pytest.approx(m.roc_auc(y, [x * 100 - 7 for x in s]))


def test_ap_known_values():
    assert m.average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))
    assert m.average_precision([1, 1, 1], [0.2, 0.5, 0.9]) == pytest.approx(1.0)
    # all tied: one threshold, precision = positive rate
    assert m.average_precision([1, 0, 0, 1, 0], [0.3] * 5) == pytest.approx(0.4)


def test_ap_matches_brute_force():
    rng = random.Random(2)
    for _ in range(20):
        n = rng.randrange(5, 60)
        y = [rng.randrange(2) for _ in range(n)]
        if sum(y) == 0:
            y[0] = 1
        s = [rng.randrange(0, 10) / 10 for _ in range(n)]
        assert m.average_precision(y, s) == pytest.approx(brute_ap(y, s))


@pytest.mark.parametrize(
    "y, s",
    [([], []), ([0, 1], [0.1]), ([0, 2], [0.1, 0.2]), ([1, 1], [0.1, 0.2]), ([0, 0], [0.1, 0.2])],
)
def test_auc_invalid(y, s):
    with pytest.raises(ValueError):
        m.roc_auc(y, s)


def test_ap_invalid():
    with pytest.raises(ValueError):
        m.average_precision([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError):
        m.average_precision([0, 1], [0.1])


def test_performance():
    rng = random.Random(3)
    n = 100_000
    y = [rng.randrange(2) for _ in range(n)]
    s = [rng.random() + 0.3 * yi for yi in y]
    start = time.perf_counter()
    auc = m.roc_auc(y, s)
    ap = m.average_precision(y, s)
    elapsed = time.perf_counter() - start
    assert 0.6 < auc < 0.9 and 0.5 < ap < 1.0
    assert elapsed < 10.0, f"too slow: {elapsed:.1f}s (needs O(n log n))"
