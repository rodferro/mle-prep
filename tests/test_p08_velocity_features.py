import random
import time

import numpy as np
import pandas as pd
import pytest

from conftest import load

m = load("p08_velocity_features")

T = pd.Timestamp("2026-03-10 12:00:00")


def at(minutes):
    return T + pd.Timedelta(minutes=minutes)


def make(cards, minutes, amounts, index=None):
    return pd.DataFrame(
        {
            "txn_id": [f"t{i}" for i in range(len(cards))],
            "card_id": cards,
            "ts": [at(x) for x in minutes],
            "amount": [float(a) for a in amounts],
        },
        index=index,
    )


def test_basic_counts_and_sums():
    tx = make(["c"] * 4, [0, 30, 59, 61], [10, 20, 30, 40])
    out = m.add_velocity_features(tx, windows=("1h",))
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["txn_id", "card_id", "ts", "amount", "count_1h", "sum_1h"]
    assert out["count_1h"].tolist() == [0, 1, 2, 2]
    assert out["sum_1h"].tolist() == [0.0, 10.0, 30.0, 50.0]  # at 61: 30 and 59 (0 is > 1h ago)
    assert pd.api.types.is_integer_dtype(out["count_1h"])
    assert pd.api.types.is_float_dtype(out["sum_1h"])


def test_exactly_window_ago_excluded():
    tx = make(["c", "c"], [0, 60], [5, 7])
    out = m.add_velocity_features(tx, windows=("1h",))
    assert out["count_1h"].tolist() == [0, 0]


def test_same_timestamp_excluded():
    tx = make(["c", "c", "c"], [0, 10, 10], [1, 2, 4])
    out = m.add_velocity_features(tx, windows=("1h",))
    assert out["count_1h"].tolist() == [0, 1, 1]
    assert out["sum_1h"].tolist() == [0.0, 1.0, 1.0]


def test_cards_are_separate_and_order_preserved():
    tx = make(["a", "b", "a", "b", "a"], [50, 5, 0, 10, 20], [1, 2, 3, 4, 5], index=[9, 3, 7, 3, 1])
    out = m.add_velocity_features(tx, windows=("1h", "15min"))
    assert out.index.tolist() == [9, 3, 7, 3, 1]          # duplicate labels kept as-is
    assert out["txn_id"].tolist() == ["t0", "t1", "t2", "t3", "t4"]
    assert out["count_1h"].tolist() == [2, 0, 0, 1, 1]
    assert out["sum_1h"].tolist() == [8.0, 0.0, 0.0, 2.0, 3.0]
    assert out["count_15min"].tolist() == [0, 0, 0, 1, 0]
    assert list(out.columns)[-4:] == ["count_1h", "sum_1h", "count_15min", "sum_15min"]


def test_input_not_modified():
    tx = make(["a", "a"], [0, 1], [1, 2])
    before = tx.copy()
    m.add_velocity_features(tx)
    pd.testing.assert_frame_equal(tx, before)


def brute(tx, minutes_window):
    w = pd.Timedelta(minutes=minutes_window)
    counts, sums = [], []
    by_card = {}
    for card, ts, amt in zip(tx["card_id"], tx["ts"], tx["amount"]):
        by_card.setdefault(card, []).append((ts, amt))
    for card, ts in zip(tx["card_id"], tx["ts"]):
        prev = [a for t, a in by_card[card] if ts - w < t < ts]
        counts.append(len(prev))
        sums.append(float(sum(prev)))
    return counts, sums


def test_matches_brute_force():
    rng = random.Random(5)
    n = 1500
    tx = make(
        [f"c{rng.randrange(25)}" for _ in range(n)],
        [rng.randrange(0, 3 * 24 * 60) for _ in range(n)],
        [rng.randrange(1, 500) for _ in range(n)],
        index=rng.sample(range(100_000), n),
    )
    out = m.add_velocity_features(tx, windows=("1h", "24h"))
    for w, minutes in (("1h", 60), ("24h", 24 * 60)):
        c, s = brute(tx, minutes)
        assert out[f"count_{w}"].tolist() == c
        assert np.allclose(out[f"sum_{w}"].to_numpy(), s)


def test_performance():
    rng = np.random.default_rng(0)
    n = 50_000
    tx = pd.DataFrame(
        {
            "txn_id": np.arange(n),
            "card_id": rng.integers(0, 5, n),
            "ts": T + pd.to_timedelta(np.sort(rng.integers(0, 30 * 24 * 3600, n)), unit="s"),
            "amount": rng.random(n) * 100,
        }
    )
    start = time.perf_counter()
    m.add_velocity_features(tx)
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"too slow: {elapsed:.1f}s (vectorize it)"


def test_missing_column():
    with pytest.raises(ValueError):
        m.add_velocity_features(pd.DataFrame({"card_id": ["a"], "ts": [T]}))
