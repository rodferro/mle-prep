import numpy as np
import pandas as pd
import pytest

from conftest import load

m = load("p09_cohort_retention")

# 2026-01-05 and 2026-01-12 are Mondays
MON1 = pd.Timestamp("2026-01-05")
MON2 = pd.Timestamp("2026-01-12")


def ts(s):
    return pd.Timestamp(s)


@pytest.fixture
def data():
    signups = pd.DataFrame(
        {
            "user_id": ["a", "b", "c", "d"],
            "signup_ts": [ts("2026-01-05 10:00"), ts("2026-01-11 23:59"), ts("2026-01-12 00:00"), ts("2026-01-14 08:00")],
        }
    )
    activity = pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b", "b", "c", "c", "zz", "d"],
            "activity_ts": [
                ts("2026-01-06"),         # a, week 0
                ts("2026-01-07"),         # a, week 0 again (count once)
                ts("2026-01-20"),         # a, week 2
                ts("2026-01-12 00:00"),   # b, week 1 (Monday midnight starts a new week)
                ts("2026-01-01"),         # b, before signup week -> ignored
                ts("2026-01-26"),         # c, week 2
                ts("2026-01-13"),         # c, week 0
                ts("2026-01-06"),         # unknown user -> ignored
                ts("2026-01-14 09:00"),   # d, week 0
            ],
        }
    )
    return signups, activity


def check(out, index, columns, values):
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == index
    assert out.index.name == "cohort"
    assert list(out.columns) == columns
    assert out.columns.name == "week"
    assert np.allclose(out.to_numpy(dtype=float), np.array(values, dtype=float))


def test_basic(data):
    signups, activity = data
    out = m.retention_matrix(signups, activity)
    # cohort MON1: a, b (size 2); cohort MON2: c, d (size 2)
    check(
        out,
        [MON1, MON2],
        [0, 1, 2],
        [
            [0.5, 0.5, 0.5],   # week0: a ; week1: b ; week2: a
            [1.0, 0.0, 0.5],   # week0: c, d ; week2: c
        ],
    )


def test_max_weeks_truncates_and_pads(data):
    signups, activity = data
    check(m.retention_matrix(signups, activity, max_weeks=1), [MON1, MON2], [0, 1], [[0.5, 0.5], [1.0, 0.0]])
    out = m.retention_matrix(signups, activity, max_weeks=4)
    assert list(out.columns) == [0, 1, 2, 3, 4]
    assert np.allclose(out[[3, 4]].to_numpy(dtype=float), 0.0)


def test_cohort_with_no_activity_is_kept():
    signups = pd.DataFrame({"user_id": ["a", "b"], "signup_ts": [ts("2026-01-05"), ts("2026-01-19 12:00")]})
    activity = pd.DataFrame({"user_id": ["a"], "activity_ts": [ts("2026-01-13")]})
    check(m.retention_matrix(signups, activity), [MON1, ts("2026-01-19")], [0, 1], [[0.0, 1.0], [0.0, 0.0]])


def test_no_activity_at_all():
    signups = pd.DataFrame({"user_id": ["a"], "signup_ts": [ts("2026-01-07")]})
    activity = pd.DataFrame({"user_id": pd.Series([], dtype=object), "activity_ts": pd.Series([], dtype="datetime64[ns]")})
    check(m.retention_matrix(signups, activity), [MON1], [0], [[0.0]])


def test_missing_columns(data):
    signups, activity = data
    with pytest.raises(ValueError):
        m.retention_matrix(signups.rename(columns={"signup_ts": "ts"}), activity)
    with pytest.raises(ValueError):
        m.retention_matrix(signups, activity.drop(columns="activity_ts"))
