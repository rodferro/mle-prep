"""
Reference solution — Problem 7 (point-in-time join).

pd.merge_asof does an "as of" lookup: for each left row it takes the last right row
whose key is <= (or <) the left key, optionally within groups (`by=`).

Details that trip people up
- Both frames must be sorted by the `on` keys (the time columns), not by customer.
- allow_exact_matches=False makes the comparison strict (feature_ts < cutoff), so a
  snapshot taken at the same instant as the transaction is not used.
- lag: compare against ts - lag instead of ts.
- merge_asof's `tolerance` is measured from the cutoff, but staleness here is defined
  from ts, so it's applied afterwards.
- Restore the original order and index at the end (merge_asof returns rows sorted by
  the `on` key).
- The two time keys must have the same dtype, including resolution: datetime64[ns]
  vs datetime64[us] raises MergeError. Cast both to one unit.
- Use .where() to blank stale features: it upcasts int columns to float, whereas
  assigning NaN into an int column with .loc fails or warns depending on the pandas
  version.

Complexity: O((n + m) log(n + m)) for the sorts, then linear.
"""

from typing import Optional

import numpy as np
import pandas as pd


def point_in_time_join(
    transactions: pd.DataFrame,
    features: pd.DataFrame,
    lag: pd.Timedelta = pd.Timedelta(0),
    max_staleness: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    missing = {"txn_id", "customer_id", "ts"} - set(transactions.columns)
    missing |= {"customer_id", "feature_ts"} - set(features.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    feature_cols = [c for c in features.columns if c not in ("customer_id", "feature_ts")]

    left = transactions.copy()
    left["_order"] = np.arange(len(left))
    # merge_asof needs both keys to have the SAME dtype. Datetime columns can differ in
    # resolution (pandas 3 parses many inputs as datetime64[us], others are [ns]).
    left["_cutoff"] = (left["ts"] - lag).astype("datetime64[ns]")
    left = left.sort_values("_cutoff", kind="stable")

    right = features.copy()
    right["feature_ts"] = right["feature_ts"].astype("datetime64[ns]")
    right = right.sort_values("feature_ts", kind="stable")

    merged = pd.merge_asof(
        left,
        right,
        left_on="_cutoff",
        right_on="feature_ts",
        by="customer_id",
        direction="backward",
        allow_exact_matches=False,
    )

    if max_staleness is not None:
        fresh = (merged["ts"] - merged["feature_ts"]) <= max_staleness
        for col in feature_cols:
            merged[col] = merged[col].where(fresh)

    merged = merged.sort_values("_order", kind="stable")
    merged.index = transactions.index
    return merged[list(transactions.columns) + feature_cols]
