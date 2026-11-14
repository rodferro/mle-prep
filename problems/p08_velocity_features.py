"""
Problem 8 — Velocity features for fraud detection (pandas)

"How many times was this card used in the last hour?" is one of the strongest fraud
signals. Compute rolling count and sum features per card.

Input `txns` columns: txn_id, card_id, ts (datetime64), amount (float). Not sorted.
Timestamps can repeat, even for the same card.

Implement `add_velocity_features(txns, windows=("1h", "24h")) -> pd.DataFrame`
    For every transaction and every window w (a pandas offset string), add:
        count_<w>   int    number of OTHER transactions of the same card with
                           ts - w < other.ts < ts
        sum_<w>     float  sum of their amounts (0.0 if there are none)
    Note both bounds are strict:
        - a transaction exactly w ago no longer counts;
        - transactions at the SAME timestamp as the current one are excluded
          (their order is unknown, so counting them would leak).
    Column names use the window string exactly as given, e.g. "count_1h", "sum_24h".

Output
    - Same rows, same order and same index as the input (the index may be anything,
      not necessarily 0..n-1), with the new columns appended in the order
      count_<w1>, sum_<w1>, count_<w2>, sum_<w2>, ...
    - Do not modify the input.

Use pandas; a Python loop over rows is too slow (a test runs 50,000 transactions).
Hint: groupby + time-based rolling with `on=` and the `closed=` parameter.

Follow-ups to discuss: computing these online at scoring time (Redis sorted sets,
streaming windows), keeping training and serving consistent, and other velocity
features (distinct merchants, distinct devices, amount vs card's average).
"""

from typing import Sequence

import pandas as pd


def add_velocity_features(txns: pd.DataFrame, windows: Sequence[str] = ("1h", "24h")) -> pd.DataFrame:
    raise NotImplementedError
