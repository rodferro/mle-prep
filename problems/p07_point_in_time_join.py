"""
Problem 7 — Point-in-time feature join (pandas)

A feature store keeps snapshots of customer features over time. To build training
data you must attach to each transaction the features that were known BEFORE the
transaction happened. Using a snapshot taken at or after the transaction leaks
the future into training and makes offline metrics look better than production.

Inputs
    transactions: columns txn_id, customer_id, ts (datetime64), plus any others
    features:     columns customer_id, feature_ts (datetime64), plus feature columns
                  (every column other than customer_id and feature_ts is a feature)
Neither DataFrame is sorted.

Implement `point_in_time_join(transactions, features, lag=pd.Timedelta(0), max_staleness=None)`:
    For each transaction, pick the feature row for the same customer with the LATEST
    feature_ts such that
        feature_ts < ts - lag              (strictly earlier)
    `lag` models pipeline delay: a snapshot is only usable `lag` after it was taken.
    If `max_staleness` is given and  ts - feature_ts > max_staleness, treat it as missing.
    If no row qualifies, the feature columns are NaN.

Output
    - Same rows, same order and same index as `transactions`.
    - All transaction columns first, then the feature columns in the order they appear
      in `features`. Do not include customer_id twice or feature_ts.
    - Do not modify the inputs.

Raise ValueError if a required column is missing.

Use pandas. Hint: pd.merge_asof (read its sorting requirements and the
allow_exact_matches / by / direction parameters). The two time columns may come with
different datetime resolutions (e.g. datetime64[ns] vs datetime64[us]); merge_asof
requires identical key dtypes.

Follow-ups to discuss: doing this in SQL (window functions or a lateral join), how
label leakage differs from feature leakage, and how to test a training pipeline for
leakage automatically.
"""

from typing import Optional

import pandas as pd


def point_in_time_join(
    transactions: pd.DataFrame,
    features: pd.DataFrame,
    lag: pd.Timedelta = pd.Timedelta(0),
    max_staleness: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    raise NotImplementedError
