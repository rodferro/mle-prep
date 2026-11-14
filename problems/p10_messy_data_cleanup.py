"""
Problem 10 — Cleaning a messy export (pandas)

Part A — `clean_customers(raw) -> (clean, rejects)`

`raw` is a CRM export where every column is a string (or missing):
    customer_id   e.g. " c-001 ", "C-001"
    updated_at    ISO 8601 date or datetime, e.g. "2026-03-01" or "2026-03-01 14:30:00"
    email         e.g. " Ana@Example.COM "
    balance       e.g. "$1,234.50", "-$12.00", "300", or empty / missing

Validate each row with these checks, IN THIS ORDER. The first failing check is the
row's reject reason:
    1. "missing_id"     customer_id is missing or blank after stripping
    2. "bad_timestamp"  updated_at is missing or not parseable as ISO 8601
    3. "bad_balance"    balance is present (non-blank) but not a number after removing
                        "$" and "," (a leading "-" is allowed, before or after "$")
    4. "bad_email"      email, after strip + lowercase, does not match
                        ^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$   (missing email also fails)

Normalize valid rows:
    customer_id  stripped and uppercased
    updated_at   datetime64
    email        stripped and lowercased
    balance      float, NaN when blank / missing

Then deduplicate: keep ONE row per customer_id, the one with the latest updated_at.
If several rows tie on the latest updated_at, keep the one that appears LAST in `raw`.

Return:
    clean    columns [customer_id, updated_at, email, balance], sorted by customer_id,
             index reset.
    rejects  the rejected rows exactly as they were in `raw` (original values and
             original index, original row order), plus a column "reason".

Part B — `flatten_events(events) -> pd.DataFrame`

`events` is a list of nested dicts like:
    {"id": "e1", "user": {"id": "u1", "country": "BR"},
     "items": [{"sku": "A", "qty": 2}, {"sku": "B", "qty": 1}]}
Return one row per item with columns [event_id, user_id, user_country, sku, qty],
in input order (events, then items). An event whose "items" is empty or missing
still produces ONE row, with sku and qty missing (NaN/None).
A missing "country" becomes missing in user_country.

Watch out: pd.json_normalize(record_path="items", ...) silently drops events with
no items.

Use pandas (pd.to_datetime with format="ISO8601", .str methods, json_normalize).
"""

from typing import Dict, List, Tuple

import pandas as pd


def clean_customers(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raise NotImplementedError


def flatten_events(events: List[Dict]) -> pd.DataFrame:
    raise NotImplementedError
