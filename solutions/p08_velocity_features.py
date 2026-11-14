"""
Reference solution — Problem 8 (velocity features).

pandas time-based rolling windows bound rows by VALUE, so closed="neither" gives
exactly (ts - w, ts): it drops rows exactly w ago and rows at the same timestamp,
including the current row itself.

Steps
1. Work on a copy with a positional row id as the index, so results map back exactly
   even if the caller's index has duplicates.
2. Sort by (card_id, ts): rolling(on=) needs each group's time column sorted.
3. groupby(card_id).rolling(w, on="ts", closed="neither", min_periods=0) for count and
   sum. min_periods=0 makes empty windows return 0 instead of NaN.
4. The results come back in the sorted order; scatter them to the original positions
   using the row ids (don't rely on the result's index labels).

Complexity: O(n log n) for the sort; the rolling pass is linear per group.

In production, the same feature computed at scoring time must match this definition
exactly (training/serving skew). A common setup is a streaming job or Redis sorted
sets keyed by card, trimmed to the largest window.
"""

from typing import Sequence

import numpy as np
import pandas as pd


def add_velocity_features(txns: pd.DataFrame, windows: Sequence[str] = ("1h", "24h")) -> pd.DataFrame:
    missing = {"card_id", "ts", "amount"} - set(txns.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    work = pd.DataFrame(
        {
            "card_id": txns["card_id"].to_numpy(),
            "ts": txns["ts"].to_numpy(),
            "amount": txns["amount"].astype("float64").to_numpy(),
        },
        index=pd.RangeIndex(len(txns), name="_row"),
    )
    work = work.sort_values(["card_id", "ts"], kind="stable")

    out = txns.copy()
    # After sorting, each card's rows are contiguous, and groupby(sort=False).rolling
    # returns results group by group in that same order: the i-th result belongs to
    # the i-th row of `work`. Scatter results back to original positions by row id.
    row_ids = work.index.to_numpy()
    grouped = work.groupby("card_id", sort=False)
    for w in windows:
        roll = grouped.rolling(w, on="ts", closed="neither", min_periods=0)["amount"]
        counts = np.empty(len(work), dtype="int64")
        sums = np.empty(len(work), dtype="float64")
        counts[row_ids] = roll.count().to_numpy()
        sums[row_ids] = np.nan_to_num(roll.sum().to_numpy().astype("float64"))
        out[f"count_{w}"] = counts
        out[f"sum_{w}"] = sums
    return out
