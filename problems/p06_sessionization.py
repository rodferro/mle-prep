"""
Problem 6 — Sessionization (pandas)

You have a click/event log with one row per event. Split each user's activity into
sessions: a new session starts when more than `gap` has passed since that user's
previous event.

Input `events` columns:
    user_id     str
    timestamp   datetime64
(Rows are NOT sorted. Other columns may be present and must be kept.)

Part A — `assign_sessions(events, gap=pd.Timedelta("30min")) -> pd.DataFrame`
    Return a copy of `events` with a new integer column `session_id`:
    - Sessions are numbered per user, starting at 1, in time order.
    - A new session starts when the time since the user's previous event is
      STRICTLY greater than `gap`. Exactly `gap` apart is the same session.
    - Identical timestamps for the same user are the same session.
    - Sort the result by (user_id, timestamp), keeping the original relative order of
      rows with equal keys (stable sort), and reset the index (0..n-1).
    - Do not modify the input DataFrame.

Part B — `summarize_sessions(events, gap=pd.Timedelta("30min")) -> pd.DataFrame`
    One row per session with columns, in this order:
        user_id, session_id, start, end, n_events, duration_seconds
    - start / end: first and last event timestamp in the session
    - n_events: int, number of events in the session
    - duration_seconds: float, (end - start) in seconds (0.0 for one-event sessions)
    Sorted by (user_id, session_id), index reset.

Empty input returns an empty DataFrame with the right columns.
Raise ValueError if `user_id` or `timestamp` is missing.

Use pandas (no Python loops over rows). Hint: sort, diff within groups, cumsum.

Follow-ups to discuss: sessions that span midnight for daily reports, very large logs
(do it in SQL with LAG + SUM OVER, or in Spark), and choosing the gap from data.
"""

import pandas as pd


def assign_sessions(events: pd.DataFrame, gap: pd.Timedelta = pd.Timedelta("30min")) -> pd.DataFrame:
    raise NotImplementedError


def summarize_sessions(events: pd.DataFrame, gap: pd.Timedelta = pd.Timedelta("30min")) -> pd.DataFrame:
    raise NotImplementedError
