import math
import random

import pandas as pd
import pytest

from conftest import load

m = load("p07_point_in_time_join")

D = pd.Timestamp("2026-02-01")


def day(n, hours=0):
    return D + pd.Timedelta(days=n, hours=hours)


@pytest.fixture
def data():
    txns = pd.DataFrame(
        {
            "txn_id": ["t1", "t2", "t3", "t4", "t5"],
            "customer_id": ["A", "B", "A", "A", "C"],
            "ts": [day(5), day(2), day(1), day(3), day(4)],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        },
        index=[100, 101, 102, 103, 104],
    )
    feats = pd.DataFrame(
        {
            "customer_id": ["A", "A", "A", "B"],
            "feature_ts": [day(3), day(0), day(2), day(2)],
            "avg_amount_30d": [3.0, 1.0, 2.0, 9.0],
            "n_txn_30d": [30, 10, 20, 90],
        }
    )
    return txns, feats


def test_basic_strictly_before(data):
    txns, feats = data
    out = m.point_in_time_join(txns, feats)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["txn_id", "customer_id", "ts", "amount", "avg_amount_30d", "n_txn_30d"]
    assert out.index.tolist() == [100, 101, 102, 103, 104]
    assert out["txn_id"].tolist() == ["t1", "t2", "t3", "t4", "t5"]
    got = out["avg_amount_30d"].tolist()
    # t1 (A, day5) -> day3 snapshot; t2 (B, day2) -> B's snapshot is AT day2 -> not allowed
    # t3 (A, day1) -> day0; t4 (A, day3) -> day2 (day3 is not strictly earlier); t5 (C) -> none
    assert got[0] == 3.0 and got[2] == 1.0 and got[3] == 2.0
    assert math.isnan(got[1]) and math.isnan(got[4])
    assert out["n_txn_30d"].iloc[0] == 30


def test_lag(data):
    txns, feats = data
    out = m.point_in_time_join(txns, feats, lag=pd.Timedelta(days=1))
    got = out["avg_amount_30d"].tolist()
    # cutoffs: t1 day4 -> day3 ; t3 day0 -> none (day0 not < day0) ; t4 day2 -> day0
    assert got[0] == 3.0
    assert math.isnan(got[2])
    assert got[3] == 1.0


def test_max_staleness(data):
    txns, feats = data
    out = m.point_in_time_join(txns, feats, max_staleness=pd.Timedelta(days=1))
    got = out["avg_amount_30d"].tolist()
    # t1: day3 snapshot is 2 days old -> stale ; t3: day0 is 1 day old -> ok ; t4: day2 -> ok
    assert math.isnan(got[0])
    assert got[2] == 1.0
    assert got[3] == 2.0


def test_inputs_not_modified(data):
    txns, feats = data
    t0, f0 = txns.copy(), feats.copy()
    m.point_in_time_join(txns, feats, lag=pd.Timedelta(hours=1), max_staleness=pd.Timedelta(days=3))
    pd.testing.assert_frame_equal(txns, t0)
    pd.testing.assert_frame_equal(feats, f0)


def brute(txns, feats, lag, max_staleness):
    out = []
    for _, t in txns.iterrows():
        cutoff = t["ts"] - lag
        cand = feats[(feats["customer_id"] == t["customer_id"]) & (feats["feature_ts"] < cutoff)]
        if len(cand):
            best = cand.sort_values("feature_ts").iloc[-1]
            if max_staleness is None or t["ts"] - best["feature_ts"] <= max_staleness:
                out.append(best["score"])
                continue
        out.append(float("nan"))
    return out


@pytest.mark.parametrize(
    "lag, stale",
    [(pd.Timedelta(0), None), (pd.Timedelta(hours=6), None), (pd.Timedelta(hours=2), pd.Timedelta(days=2))],
)
def test_matches_brute_force(lag, stale):
    rng = random.Random(11)
    n_t, n_f = 300, 200
    txns = pd.DataFrame(
        {
            "txn_id": [f"t{i}" for i in range(n_t)],
            "customer_id": [f"c{rng.randrange(15)}" for _ in range(n_t)],
            "ts": [day(0, rng.randrange(0, 24 * 10)) for _ in range(n_t)],
        },
        index=rng.sample(range(10_000), n_t),
    )
    feats = pd.DataFrame(
        {
            "customer_id": [f"c{rng.randrange(15)}" for _ in range(n_f)],
            "feature_ts": [day(0, rng.randrange(0, 24 * 10)) for _ in range(n_f)],
            "score": [round(rng.random(), 4) for _ in range(n_f)],
        }
    )
    # make feature timestamps unique per customer so "latest" is unambiguous
    feats = feats.drop_duplicates(["customer_id", "feature_ts"])
    out = m.point_in_time_join(txns, feats, lag=lag, max_staleness=stale)
    exp = brute(txns, feats, lag, stale)
    got = out["score"].tolist()
    assert out.index.tolist() == txns.index.tolist()
    for g, e in zip(got, exp):
        assert (math.isnan(g) and math.isnan(e)) or g == e


def test_missing_columns(data):
    txns, feats = data
    with pytest.raises(ValueError):
        m.point_in_time_join(txns.drop(columns="ts"), feats)
    with pytest.raises(ValueError):
        m.point_in_time_join(txns, feats.drop(columns="feature_ts"))
