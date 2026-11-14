import math
import random
import time

import pytest

from conftest import load

m = load("p14_threshold_selection")


def brute_value(scores, labels, amounts, cost):
    best_t, best_v = math.inf, 0.0
    for t in sorted(set(scores), reverse=True):
        v = sum(y * a - cost for s, y, a in zip(scores, labels, amounts) if s >= t)
        if v > best_v + 1e-9:
            best_t, best_v = t, v
    return best_t, best_v


def brute_budget(scores, k):
    best = (math.inf, 0)
    for t in sorted(set(scores), reverse=True):
        n = sum(1 for s in scores if s >= t)
        if n <= k:
            best = (t, n)
    return best


def test_value_small_example():
    scores = [0.9, 0.8, 0.7, 0.4, 0.2]
    labels = [1, 0, 1, 0, 1]
    amounts = [100, 50, 30, 80, 5]
    # t=0.9: 90 ; t=0.8: 80 ; t=0.7: 100 ; t=0.4: 90 ; t=0.2: 85
    t, v = m.best_threshold_by_value(scores, labels, amounts, review_cost=10)
    assert t == 0.7 and v == pytest.approx(100)


def test_value_review_nothing_when_unprofitable():
    t, v = m.best_threshold_by_value([0.9, 0.5], [0, 1], [100, 1], review_cost=5)
    assert t == math.inf and v == 0.0


def test_value_tie_prefers_higher_threshold():
    # t=0.9: 10 ; t=0.5: 10 (the extra legit+fraud cancel out) -> keep 0.9
    t, v = m.best_threshold_by_value([0.9, 0.5, 0.5], [1, 1, 0], [20, 10, 0], review_cost=10)
    assert t == 0.9 and v == pytest.approx(10)


def test_value_tied_scores_enter_together():
    # both 0.8 rows must be reviewed together: value 40 - 10 - 10 = 20 > 0
    t, v = m.best_threshold_by_value([0.8, 0.8], [1, 0], [40, 999], review_cost=10)
    assert t == 0.8 and v == pytest.approx(20)


def test_value_matches_brute_force():
    rng = random.Random(7)
    for _ in range(25):
        n = rng.randrange(1, 80)
        scores = [rng.randrange(0, 20) / 20 for _ in range(n)]
        labels = [int(rng.random() < s * 0.6) for s in scores]
        amounts = [rng.randrange(1, 200) for _ in range(n)]
        cost = rng.choice([0, 5, 15, 40])
        t, v = m.best_threshold_by_value(scores, labels, amounts, cost)
        bt, bv = brute_value(scores, labels, amounts, cost)
        assert t == bt and v == pytest.approx(bv)


def test_budget_examples():
    scores = [0.9, 0.8, 0.8, 0.5, 0.1]
    assert m.threshold_for_budget(scores, 1) == (0.9, 1)
    assert m.threshold_for_budget(scores, 2) == (0.9, 1)    # 0.8 would make 3
    assert m.threshold_for_budget(scores, 3) == (0.8, 3)
    assert m.threshold_for_budget(scores, 10) == (0.1, 5)
    assert m.threshold_for_budget(scores, 0) == (math.inf, 0)
    assert m.threshold_for_budget([0.7, 0.7, 0.7], 2) == (math.inf, 0)


def test_budget_matches_brute_force():
    rng = random.Random(8)
    for _ in range(25):
        scores = [rng.randrange(0, 15) / 15 for _ in range(rng.randrange(1, 60))]
        k = rng.randrange(0, 70)
        assert m.threshold_for_budget(scores, k) == brute_budget(scores, k)


def test_invalid():
    with pytest.raises(ValueError):
        m.best_threshold_by_value([0.1], [1, 0], [1, 1], 1)
    with pytest.raises(ValueError):
        m.best_threshold_by_value([], [], [], 1)
    with pytest.raises(ValueError):
        m.best_threshold_by_value([0.1], [2], [1], 1)
    with pytest.raises(ValueError):
        m.best_threshold_by_value([0.1], [1], [1], -1)
    with pytest.raises(ValueError):
        m.threshold_for_budget([], 3)
    with pytest.raises(ValueError):
        m.threshold_for_budget([0.5], -1)


def test_performance():
    rng = random.Random(9)
    n = 200_000
    scores = [rng.random() for _ in range(n)]
    labels = [int(rng.random() < s * 0.05) for s in scores]
    amounts = [rng.uniform(1, 500) for _ in range(n)]
    start = time.perf_counter()
    m.best_threshold_by_value(scores, labels, amounts, review_cost=3)
    m.threshold_for_budget(scores, 1000)
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"too slow: {elapsed:.1f}s (needs O(n log n))"
