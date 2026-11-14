"""
Problem 9 — Weekly cohort retention table (pandas)

Product asks: "Of the users who signed up in a given week, what fraction were active
1, 2, 3... weeks later?"

Inputs
    signups:  columns user_id, signup_ts (datetime64). One row per user.
    activity: columns user_id, activity_ts (datetime64). Many rows per user.

Definitions
    - Weeks start on Monday at 00:00.
    - A user's cohort = the Monday of their signup week.
    - An activity's week offset k = (Monday of the activity's week - cohort) / 7 days.
      Activity in the signup week is k = 0.
    - Ignore activity before the user's signup week (k < 0) and activity from users
      who are not in `signups`.
    - A user counts once per (cohort, k) no matter how many events they had.

Implement `retention_matrix(signups, activity, max_weeks=None) -> pd.DataFrame`
    - Index: cohort Mondays (timestamps at midnight), sorted ascending, named "cohort".
      Every cohort in `signups` appears, even if none of its users were ever active.
    - Columns: integers 0..K, named "week". K = the largest offset observed, or
      `max_weeks` if given (then columns are exactly 0..max_weeks, dropping later
      weeks and adding empty ones).
    - Values: float = active users / cohort size. Missing combinations are 0.0.
    - If there is no activity at all, columns are just [0] (or 0..max_weeks).

Raise ValueError if a required column is missing.

Use pandas. Hint: dt.normalize, dt.weekday, drop_duplicates, pivot_table / unstack,
reindex.

Follow-ups to discuss: calendar weeks vs "days since signup" buckets, right-censoring
(recent cohorts can't have week-8 data yet: show NaN instead of 0), and how to present
this (heatmap; retention curves per cohort).
"""

from typing import Optional

import pandas as pd


def retention_matrix(
    signups: pd.DataFrame,
    activity: pd.DataFrame,
    max_weeks: Optional[int] = None,
) -> pd.DataFrame:
    raise NotImplementedError
