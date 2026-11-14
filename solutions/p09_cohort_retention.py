"""
Reference solution — Problem 9 (weekly cohort retention).

Steps
1. Monday of a timestamp's week: ts.dt.normalize() - weekday days.
2. Cohort per user from signups; cohort sizes via value_counts.
3. Join activity to cohorts (inner join drops unknown users), compute the week offset,
   drop k < 0, then drop_duplicates on (user, k) so each user counts once per week.
4. Count users per (cohort, k), unstack to a matrix, divide each row by its cohort size.
5. Reindex rows to every cohort and columns to 0..K, filling 0.0.

Right-censoring note: this fills "no data yet" with 0.0 as the spec asks. In a real
report, cells beyond the latest available week for a cohort should be NaN, so recent
cohorts don't look like they churned.
"""

from typing import Optional

import pandas as pd


def _week_start(ts: pd.Series) -> pd.Series:
    return ts.dt.normalize() - pd.to_timedelta(ts.dt.weekday, unit="D")


def retention_matrix(
    signups: pd.DataFrame,
    activity: pd.DataFrame,
    max_weeks: Optional[int] = None,
) -> pd.DataFrame:
    missing = {"user_id", "signup_ts"} - set(signups.columns)
    missing |= {"user_id", "activity_ts"} - set(activity.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    users = pd.DataFrame({"user_id": signups["user_id"], "cohort": _week_start(signups["signup_ts"])})
    sizes = users.groupby("cohort").size().sort_index()

    act = pd.DataFrame({"user_id": activity["user_id"], "week": _week_start(activity["activity_ts"])})
    act = act.merge(users, on="user_id", how="inner")
    act["k"] = ((act["week"] - act["cohort"]).dt.days // 7).astype("int64")
    act = act[act["k"] >= 0].drop_duplicates(["user_id", "k"])

    counts = act.groupby(["cohort", "k"]).size().unstack("k")

    if max_weeks is not None:
        last = max_weeks
    elif counts.empty:
        last = 0
    else:
        last = int(counts.columns.max())

    matrix = counts.reindex(index=sizes.index, columns=range(last + 1)).fillna(0)
    matrix = matrix.div(sizes, axis=0).astype("float64")
    matrix.index.name = "cohort"
    matrix.columns = pd.Index(list(range(last + 1)), name="week")
    return matrix
