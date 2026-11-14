import random

import pandas as pd
import pytest

from conftest import load

m = load("p06_sessionization")

T = pd.Timestamp("2026-01-05 09:00:00")


def mins(*offsets):
    return [T + pd.Timedelta(minutes=o) for o in offsets]


def make(users, minutes, **extra):
    return pd.DataFrame({"user_id": users, "timestamp": mins(*minutes), **extra})


def test_single_user_gap_rule():
    ev = make(["u1"] * 5, [0, 10, 40, 71, 101])
    out = m.assign_sessions(ev)
    # 10->40 is exactly 30 min (same session); 40->71 is 31 min (new); 71->101 is 30 (same)
    assert out["session_id"].tolist() == [1, 1, 1, 2, 2]


def test_returns_dataframe_and_keeps_extra_columns():
    ev = make(["u1", "u1"], [0, 5], page=["home", "cart"])
    out = m.assign_sessions(ev)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["user_id", "timestamp", "page", "session_id"]
    assert pd.api.types.is_integer_dtype(out["session_id"])


def test_unsorted_multi_user_and_index_reset():
    ev = make(["b", "a", "b", "a", "a", "b"], [100, 0, 0, 45, 20, 10], idx=list("pqrstu"))
    ev.index = [10, 11, 12, 13, 14, 15]
    out = m.assign_sessions(ev)
    assert out["user_id"].tolist() == ["a", "a", "a", "b", "b", "b"]
    assert out["timestamp"].tolist() == mins(0, 20, 45, 0, 10, 100)
    assert out["session_id"].tolist() == [1, 1, 1, 1, 1, 2]
    assert out.index.tolist() == list(range(6))


def test_identical_timestamps_and_stable_order():
    ev = make(["u", "u", "u"], [0, 0, 0], tag=["x", "y", "z"])
    out = m.assign_sessions(ev)
    assert out["session_id"].tolist() == [1, 1, 1]
    assert out["tag"].tolist() == ["x", "y", "z"]


def test_custom_gap():
    ev = make(["u"] * 3, [0, 6, 12])
    assert m.assign_sessions(ev, gap=pd.Timedelta("5min"))["session_id"].tolist() == [1, 2, 3]


def test_input_not_modified():
    ev = make(["u", "u"], [5, 0])
    before = ev.copy()
    m.assign_sessions(ev)
    m.summarize_sessions(ev)
    pd.testing.assert_frame_equal(ev, before)


def test_summary():
    ev = make(["u1", "u1", "u1", "u2", "u1"], [0, 10, 100, 3, 115])
    out = m.summarize_sessions(ev)
    expected = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "session_id": [1, 2, 1],
            "start": mins(0, 100, 3),
            "end": mins(10, 115, 3),
            "n_events": [2, 2, 1],
            "duration_seconds": [600.0, 900.0, 0.0],
        }
    )
    assert list(out.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected, check_dtype=False)
    assert pd.api.types.is_integer_dtype(out["n_events"])
    assert pd.api.types.is_float_dtype(out["duration_seconds"])


def test_matches_brute_force():
    rng = random.Random(3)
    rows = [(f"u{rng.randrange(20)}", rng.randrange(0, 2000)) for _ in range(800)]
    ev = make([u for u, _ in rows], [t for _, t in rows])
    out = m.assign_sessions(ev)

    expected = {}
    for u in sorted({u for u, _ in rows}):
        times = sorted(t for uu, t in rows if uu == u)
        sid, prev = 0, None
        for t in times:
            if prev is None or t - prev > 30:
                sid += 1
            expected.setdefault(u, []).append(sid)
            prev = t
    got = out.groupby("user_id", sort=True)["session_id"].apply(list).to_dict()
    assert got == expected


def test_empty_input():
    ev = pd.DataFrame({"user_id": pd.Series([], dtype=object), "timestamp": pd.Series([], dtype="datetime64[ns]")})
    out = m.summarize_sessions(ev)
    assert len(out) == 0
    assert list(out.columns) == ["user_id", "session_id", "start", "end", "n_events", "duration_seconds"]


def test_missing_column():
    with pytest.raises(ValueError):
        m.assign_sessions(pd.DataFrame({"user_id": ["a"]}))
