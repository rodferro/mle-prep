import random
import time

import pytest

from conftest import load

m = load("p04_rolling_metrics")


def brute(pairs):
    tp = sum(1 for t, p in pairs if t == 1 and p == 1)
    fp = sum(1 for t, p in pairs if t == 0 and p == 1)
    fn = sum(1 for t, p in pairs if t == 1 and p == 0)
    tn = sum(1 for t, p in pairs if t == 0 and p == 0)
    div = lambda a, b: a / b if b else 0.0
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "accuracy": div(tp + tn, len(pairs)),
        "precision": prec,
        "recall": rec,
        "f1": div(2 * prec * rec, prec + rec),
    }


def metrics(rm):
    return {
        "accuracy": rm.accuracy(),
        "precision": rm.precision(),
        "recall": rm.recall(),
        "f1": rm.f1(),
    }


def test_empty_window_returns_zeros():
    rm = m.RollingMetrics(5)
    assert rm.count() == 0
    assert metrics(rm) == {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_small_example():
    rm = m.RollingMetrics(4)
    for t, p in [(1, 1), (0, 1), (1, 0), (0, 0)]:
        rm.update(t, p)
    assert rm.count() == 4
    assert rm.accuracy() == pytest.approx(0.5)
    assert rm.precision() == pytest.approx(0.5)
    assert rm.recall() == pytest.approx(0.5)
    assert rm.f1() == pytest.approx(0.5)


def test_window_evicts_oldest():
    rm = m.RollingMetrics(3)
    for t, p in [(1, 0), (1, 0), (1, 0), (1, 1), (1, 1), (1, 1)]:
        rm.update(t, p)
    assert rm.count() == 3
    assert rm.recall() == pytest.approx(1.0)
    assert rm.accuracy() == pytest.approx(1.0)


def test_matches_brute_force_on_random_stream():
    rng = random.Random(42)
    window = 50
    rm = m.RollingMetrics(window)
    history = []
    for _ in range(2_000):
        t = int(rng.random() < 0.3)
        p = t if rng.random() < 0.8 else 1 - t
        rm.update(t, p)
        history.append((t, p))
        exp = brute(history[-window:])
        got = metrics(rm)
        for k in exp:
            assert got[k] == pytest.approx(exp[k]), k
        assert rm.count() == min(len(history), window)


def test_window_size_one():
    rm = m.RollingMetrics(1)
    rm.update(1, 1)
    assert rm.f1() == 1.0
    rm.update(0, 1)
    assert rm.precision() == 0.0
    assert rm.count() == 1


@pytest.mark.parametrize("bad", [0, -3])
def test_invalid_window(bad):
    with pytest.raises(ValueError):
        m.RollingMetrics(bad)


@pytest.mark.parametrize("pair", [(2, 1), (1, -1), (0, 5)])
def test_invalid_labels(pair):
    rm = m.RollingMetrics(3)
    with pytest.raises(ValueError):
        rm.update(*pair)


def test_performance_is_constant_time():
    rng = random.Random(7)
    rm = m.RollingMetrics(5_000)
    start = time.perf_counter()
    for _ in range(200_000):
        t = rng.getrandbits(1)
        p = rng.getrandbits(1)
        rm.update(t, p)
        rm.accuracy(); rm.precision(); rm.recall(); rm.f1()
    elapsed = time.perf_counter() - start
    # O(1) solutions take well under 1s; recomputing over the window takes minutes.
    assert elapsed < 5.0, f"too slow: {elapsed:.2f}s"
