import pytest

from conftest import load

m = load("p03_near_duplicates")

BASE = "Please review and sign the master services agreement for Acme Corp by Friday"


def ev(ts, payload=BASE, agreement="A1", etype="envelope_sent"):
    return {"agreement_id": agreement, "event_type": etype, "timestamp": ts, "payload": payload}


def test_first_event_is_not_duplicate():
    d = m.NearDuplicateDetector()
    assert d.process(ev(0)) is False
    assert d.active_count() == 1


def test_exact_duplicate_within_window():
    d = m.NearDuplicateDetector(window_seconds=60)
    assert d.process(ev(0)) is False
    assert d.process(ev(30)) is True
    assert d.active_count() == 1  # duplicates are not stored


def test_window_boundary_is_inclusive():
    d = m.NearDuplicateDetector(window_seconds=60)
    d.process(ev(0))
    assert d.process(ev(60)) is True


def test_same_event_outside_window_is_new():
    d = m.NearDuplicateDetector(window_seconds=60)
    d.process(ev(0))
    assert d.process(ev(61)) is False


def test_case_and_punctuation_ignored():
    d = m.NearDuplicateDetector()
    d.process(ev(0))
    assert d.process(ev(5, BASE.upper() + "!!!")) is True


def test_small_edit_is_near_duplicate():
    d = m.NearDuplicateDetector(threshold=0.8)
    d.process(ev(0))
    edited = BASE.replace("Friday", "Monday")  # 12 shared / 14 total tokens ≈ 0.86
    assert d.process(ev(10, edited)) is True


def test_different_payload_is_not_duplicate():
    d = m.NearDuplicateDetector(threshold=0.8)
    d.process(ev(0))
    assert d.process(ev(10, "Reminder: your NDA with Globex expires next month")) is False


def test_threshold_respected():
    d = m.NearDuplicateDetector(threshold=0.95)
    d.process(ev(0))
    edited = BASE.replace("Friday", "Monday")
    assert d.process(ev(10, edited)) is False


def test_key_includes_agreement_and_event_type():
    d = m.NearDuplicateDetector()
    d.process(ev(0))
    assert d.process(ev(1, agreement="A2")) is False
    assert d.process(ev(2, etype="signed")) is False
    assert d.process(ev(3)) is True


def test_compares_against_all_kept_events_for_key():
    d = m.NearDuplicateDetector()
    d.process(ev(0, "alpha beta gamma delta"))
    d.process(ev(1, "one two three four"))
    assert d.process(ev(2, "alpha beta gamma delta")) is True
    assert d.process(ev(3, "one two three four")) is True


def test_empty_payloads_are_identical():
    d = m.NearDuplicateDetector()
    d.process(ev(0, ""))
    assert d.process(ev(1, "")) is True


def test_memory_is_bounded():
    d = m.NearDuplicateDetector(window_seconds=10)
    for t in range(10_000):
        d.process(ev(t, f"unique payload number {t}", agreement=f"A{t % 50}"))
    # only events with timestamp in [9989, 9999] can still be held
    assert d.active_count() <= 11


def test_out_of_order_raises():
    d = m.NearDuplicateDetector()
    d.process(ev(100))
    with pytest.raises(ValueError):
        d.process(ev(50))
