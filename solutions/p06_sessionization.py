"""
Reference solution — Problem 6 (sessionization).

Steps
1. Stable sort by (user_id, timestamp).
2. Within each user, diff() gives the time since the previous event (NaT for the first).
3. A row starts a new session when it is the user's first row or the gap is > `gap`.
4. groupby(user).cumsum() of that boolean gives 1-based session numbers.

All vectorized: O(n log n) for the sort, O(n) for the rest.

SQL equivalent:
    SUM(CASE WHEN ts - LAG(ts) OVER (PARTITION BY user ORDER BY ts) > gap
             OR LAG(ts) OVER (...) IS NULL THEN 1 ELSE 0 END)
        OVER (PARTITION BY user ORDER BY ts ROWS UNBOUNDED PRECEDING)
"""

import pandas as pd

SUMMARY_COLUMNS = ["user_id", "session_id", "start", "end", "n_events", "duration_seconds"]


def _check(events: pd.DataFrame) -> None:
    missing = {"user_id", "timestamp"} - set(events.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")


def assign_sessions(events: pd.DataFrame, gap: pd.Timedelta = pd.Timedelta("30min")) -> pd.DataFrame:
    _check(events)
    df = events.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    since_prev = df.groupby("user_id", sort=False)["timestamp"].diff()
    new_session = since_prev.isna() | (since_prev > gap)
    df["session_id"] = new_session.astype("int64").groupby(df["user_id"], sort=False).cumsum().astype("int64")
    return df


def summarize_sessions(events: pd.DataFrame, gap: pd.Timedelta = pd.Timedelta("30min")) -> pd.DataFrame:
    df = assign_sessions(events, gap)
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    out = (
        df.groupby(["user_id", "session_id"], sort=True)["timestamp"]
        .agg(start="min", end="max", n_events="size")
        .reset_index()
    )
    out["n_events"] = out["n_events"].astype("int64")
    out["duration_seconds"] = (out["end"] - out["start"]).dt.total_seconds().astype("float64")
    return out[SUMMARY_COLUMNS]
