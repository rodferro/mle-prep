import math

import numpy as np
import pandas as pd

from conftest import load

m = load("p10_messy_data_cleanup")


def raw_frame():
    return pd.DataFrame(
        {
            "customer_id": [" c-001 ", "C-002", "c-001", "", "C-003", "c-004", "C-005", "c-002", None, "c-006", "C-007", "c-007"],
            "updated_at": [
                "2026-03-01",
                "2026-03-02 10:00:00",
                "2026-03-05 08:30:00",
                "2026-03-01",
                "not a date",
                "2026-03-03",
                "2026-03-04",
                "2026-03-02 10:00:00",
                "2026-03-01",
                "2026-03-06",
                "2026-03-07",
                "2026-03-07",
            ],
            "email": [
                " Ana@Example.COM ",
                "bo@example.com",
                "ana.new@example.com",
                "x@example.com",
                "c@example.com",
                "not-an-email",
                "e@example.com",
                "bo2@example.com",
                "y@example.com",
                None,
                "first@example.com",
                "second@example.com",
            ],
            "balance": ["$1,234.50", "-$12.00", "300", "1", "2", "3", "12abc", "", "4", "5", "$-7.25", None],
        },
        index=[f"r{i}" for i in range(12)],
    )


def test_returns_two_frames():
    clean, rejects = m.clean_customers(raw_frame())
    assert isinstance(clean, pd.DataFrame) and isinstance(rejects, pd.DataFrame)
    assert list(clean.columns) == ["customer_id", "updated_at", "email", "balance"]


def test_rejects_and_reasons():
    raw = raw_frame()
    _, rejects = m.clean_customers(raw)
    assert rejects.index.tolist() == ["r3", "r4", "r5", "r6", "r8", "r9"]
    assert rejects["reason"].tolist() == [
        "missing_id", "bad_timestamp", "bad_email", "bad_balance", "missing_id", "bad_email",
    ]
    # original values are untouched
    assert rejects.loc["r5", "email"] == "not-an-email"
    assert list(rejects.columns) == list(raw.columns) + ["reason"]


def test_clean_rows():
    clean, _ = m.clean_customers(raw_frame())
    assert clean["customer_id"].tolist() == ["C-001", "C-002", "C-007"]
    assert clean.index.tolist() == [0, 1, 2]
    # C-001: latest is the 2026-03-05 row ; C-002 tie on updated_at -> last row wins ; C-007 tie -> last
    assert clean["email"].tolist() == ["ana.new@example.com", "bo2@example.com", "second@example.com"]
    assert list(clean["updated_at"]) == [pd.Timestamp("2026-03-05 08:30"), pd.Timestamp("2026-03-02 10:00"), pd.Timestamp("2026-03-07")]
    assert pd.api.types.is_datetime64_any_dtype(clean["updated_at"])
    bal = clean["balance"].tolist()
    assert bal[0] == 300.0 and math.isnan(bal[1]) and math.isnan(bal[2])
    assert pd.api.types.is_float_dtype(clean["balance"])


def test_balance_parsing_and_email_normalization():
    raw = pd.DataFrame(
        {
            "customer_id": ["a", "b", "c", "d"],
            "updated_at": ["2026-01-01"] * 4,
            "email": [" Ana@Example.COM ", "b@x.io", "c@x.io", "d@x.io"],
            "balance": ["$1,234.50", "-$12.00", "$-7.25", " 300 "],
        }
    )
    clean, rejects = m.clean_customers(raw)
    assert len(rejects) == 0
    assert clean["balance"].tolist() == [1234.50, -12.0, -7.25, 300.0]
    assert clean.loc[0, "email"] == "ana@example.com"


def test_input_not_modified():
    raw = raw_frame()
    before = raw.copy()
    m.clean_customers(raw)
    pd.testing.assert_frame_equal(raw, before)


# ---------- Part B ----------

EVENTS = [
    {"id": "e1", "user": {"id": "u1", "country": "BR"}, "items": [{"sku": "A", "qty": 2}, {"sku": "B", "qty": 1}]},
    {"id": "e2", "user": {"id": "u2"}, "items": []},
    {"id": "e3", "user": {"id": "u3", "country": "US"}},
    {"id": "e4", "user": {"id": "u1", "country": "BR"}, "items": [{"sku": "C", "qty": 5}]},
]


def test_flatten_events():
    out = m.flatten_events(EVENTS)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["event_id", "user_id", "user_country", "sku", "qty"]
    assert out["event_id"].tolist() == ["e1", "e1", "e2", "e3", "e4"]
    assert out["user_id"].tolist() == ["u1", "u1", "u2", "u3", "u1"]
    assert out["user_country"].isna().tolist() == [False, False, True, False, False]
    assert out["sku"].isna().tolist() == [False, False, True, True, False]
    assert out["sku"].dropna().tolist() == ["A", "B", "C"]
    qty = out["qty"].tolist()
    assert qty[0] == 2 and qty[1] == 1 and qty[4] == 5
    assert pd.isna(qty[2]) and pd.isna(qty[3])


def test_flatten_empty_list():
    out = m.flatten_events([])
    assert len(out) == 0
    assert list(out.columns) == ["event_id", "user_id", "user_country", "sku", "qty"]
