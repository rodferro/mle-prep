"""
Reference solution — Problem 3 (near-duplicate agreement events).

Design
- Partition kept events by key = (agreement_id, event_type) -> deque of (timestamp, tokens).
  A new event is only compared with recent events of its own key.
- A global deque of (timestamp, key) in arrival order drives eviction: because timestamps
  are non-decreasing, the oldest kept event is always at the left of both the global
  deque and its key's deque, so eviction is amortized O(1) per event.
- Similarity: Jaccard over token sets. Pre-tokenize once when storing.

Complexity per event: O(k * t) where k = kept events for that key inside the window and
t = tokens per payload. Memory: O(events kept inside the window).

Scaling notes
- Many events per key: MinHash signatures + LSH buckets instead of pairwise Jaccard.
- Distributed: partition the stream by agreement_id (e.g. Kafka key) so each key's state
  lives in one consumer; keep state in a store with TTL = window (Redis, Flink state).
- Out-of-order data: buffer with an allowed-lateness watermark instead of raising.
"""

import re
from collections import deque
from typing import Deque, Dict, FrozenSet, Tuple

_TOKEN = re.compile(r"\w+")


def _tokens(text: str) -> FrozenSet[str]:
    return frozenset(_TOKEN.findall(text.lower()))


def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class NearDuplicateDetector:
    def __init__(self, window_seconds: float = 60.0, threshold: float = 0.8):
        if window_seconds < 0:
            raise ValueError("window_seconds must be >= 0")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.window = window_seconds
        self.threshold = threshold
        self._by_key: Dict[Tuple[str, str], Deque[Tuple[float, FrozenSet[str]]]] = {}
        self._order: Deque[Tuple[float, Tuple[str, str]]] = deque()
        self._latest = float("-inf")

    def _evict(self, now: float) -> None:
        while self._order and now - self._order[0][0] > self.window:
            _, key = self._order.popleft()
            bucket = self._by_key[key]
            bucket.popleft()
            if not bucket:
                del self._by_key[key]

    def process(self, event: Dict) -> bool:
        ts = float(event["timestamp"])
        if ts < self._latest:
            raise ValueError("events must arrive in non-decreasing timestamp order")
        self._latest = ts
        self._evict(ts)

        key = (event["agreement_id"], event["event_type"])
        toks = _tokens(event.get("payload", "") or "")

        for _, prev in self._by_key.get(key, ()):
            if _jaccard(toks, prev) >= self.threshold:
                return True

        self._by_key.setdefault(key, deque()).append((ts, toks))
        self._order.append((ts, key))
        return False

    def active_count(self) -> int:
        return len(self._order)
