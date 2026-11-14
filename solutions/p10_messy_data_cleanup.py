"""
Reference solution — Problem 10 (cleaning a messy export).

Part A
- Build one boolean mask per check, vectorized with .str methods.
- Assign the reason with np.select, which picks the FIRST matching condition, so the
  check order matches the spec without any loops.
- Parse dates with format="ISO8601" and errors="coerce": invalid values become NaT
  instead of raising, and NaT means reject.
- Balance: strip "$" and ",", then pd.to_numeric(errors="coerce"). A non-blank value
  that becomes NaN is invalid.
- Dedupe: a stable sort by updated_at then drop_duplicates(keep="last") keeps the
  latest row per id, and among ties the one that came last in the input.

Part B
- json_normalize(record_path="items", meta=[...]) drops events with empty or missing
  items, because there are no records to emit. Fix: give those events a placeholder
  item {} before normalizing (or build the rows by hand).

Talking points: keep rejects with reasons (don't silently drop), alert when the
reject rate jumps, and store the raw data so cleaning rules can be changed and re-run.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
CLEAN_COLUMNS = ["customer_id", "updated_at", "email", "balance"]


def clean_customers(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def text(col: str) -> pd.Series:
        # object Series of stripped strings, or None for missing values
        return raw[col].map(lambda v: None if pd.isna(v) else str(v).strip()).astype("object")

    cid = text("customer_id")
    email = text("email").map(lambda v: v.lower() if isinstance(v, str) else None)
    ts = pd.to_datetime(text("updated_at"), format="ISO8601", errors="coerce")

    bal_text = text("balance")
    bal_blank = bal_text.isna() | (bal_text == "")
    bal_digits = bal_text.str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    balance = pd.to_numeric(bal_digits.where(~bal_blank), errors="coerce").astype("float64")

    missing_id = cid.isna() | (cid == "")
    bad_ts = ts.isna()
    bad_balance = ~bal_blank & balance.isna()
    bad_email = ~email.fillna("").str.match(EMAIL_RE).astype(bool)

    reason = pd.Series(
        np.select(
            [missing_id, bad_ts, bad_balance, bad_email],
            ["missing_id", "bad_timestamp", "bad_balance", "bad_email"],
            default="",
        ),
        index=raw.index,
    )
    rejected = reason != ""

    rejects = raw[rejected].copy()
    rejects["reason"] = reason[rejected]

    valid = pd.DataFrame(
        {
            "customer_id": cid.str.upper(),
            "updated_at": ts,
            "email": email,
            "balance": balance,
        }
    )[~rejected]
    clean = (
        valid.sort_values("updated_at", kind="stable")
        .drop_duplicates("customer_id", keep="last")
        .sort_values("customer_id", kind="stable")
        .reset_index(drop=True)
    )
    return clean[CLEAN_COLUMNS], rejects


def flatten_events(events: List[Dict]) -> pd.DataFrame:
    rows = []
    for ev in events:
        user = ev.get("user") or {}
        items = ev.get("items") or [{}]
        for item in items:
            rows.append(
                {
                    "event_id": ev.get("id"),
                    "user_id": user.get("id"),
                    "user_country": user.get("country"),
                    "sku": item.get("sku"),
                    "qty": item.get("qty"),
                }
            )
    return pd.DataFrame(rows, columns=["event_id", "user_id", "user_country", "sku", "qty"])
