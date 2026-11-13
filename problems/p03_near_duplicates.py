"""
Problem 3 — Near-duplicate agreement events in a stream

Upstream systems sometimes emit the same agreement event more than once (retries,
double-clicks, webhooks re-delivered with slightly different payload text). Build a
streaming detector.

Each event is a dict:
    {
        "agreement_id": str,
        "event_type":   str,     # e.g. "envelope_sent", "signed", "voided"
        "timestamp":    float,   # seconds; events arrive in NON-DECREASING timestamp order
        "payload":      str,     # free text (e.g. message body, comment)
    }

An event is a NEAR-DUPLICATE if there is a previously KEPT (non-duplicate) event with:
    1. the same agreement_id AND the same event_type,
    2. timestamp difference <= window_seconds, and
    3. payload similarity >= threshold, where similarity is the Jaccard similarity of
       the two payloads' token SETS. Tokens = lowercase runs of letters/digits/underscore
       (i.e. re.findall(r"\\w+", text.lower())). Two empty payloads have similarity 1.0.

Implement class `NearDuplicateDetector`:
    __init__(window_seconds=60.0, threshold=0.8)
    process(event) -> bool       True if the event is a near-duplicate (and is NOT stored),
                                 False otherwise (the event is stored as kept).
    active_count() -> int        number of kept events currently held in memory.

Requirements
    - Memory must stay bounded: kept events older than the window (relative to the most
      recent timestamp seen) must be evicted. active_count() is used to check this.
    - Per-event cost should depend on the number of recent events for the SAME
      (agreement_id, event_type) key, not on everything in the window.
    - Raise ValueError for an event with a timestamp earlier than the latest one seen.

Follow-ups to discuss (not tested): out-of-order events / allowed lateness, using MinHash
+ LSH when one key has thousands of events, and running this across partitions
(e.g. Kafka keyed by agreement_id).
"""

from typing import Dict


class NearDuplicateDetector:
    def __init__(self, window_seconds: float = 60.0, threshold: float = 0.8):
        raise NotImplementedError

    def process(self, event: Dict) -> bool:
        raise NotImplementedError

    def active_count(self) -> int:
        raise NotImplementedError
