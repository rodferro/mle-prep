"""
Reference solution — Problem 4 (rolling metrics in O(1)).

Keep the window in a deque and maintain running TP/FP/FN/TN counts. On each update add
the new pair's contribution and, if the window is full, subtract the evicted pair's.
All metrics are then simple arithmetic on four integers.

Time: O(1) per update and per metric. Memory: O(window_size).

Notes
- Time-based windows: store (timestamp, outcome) and evict by time instead of length.
- Delayed labels: join predictions with labels when they arrive and window on
  prediction time, otherwise recent windows look artificially good/bad.
- Rolling AUC isn't O(1) with counts alone; use score histograms (binned) per window.
"""

from collections import deque
from typing import Deque, Tuple


class RollingMetrics:
    def __init__(self, window_size: int):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self._window: Deque[Tuple[int, int]] = deque()
        self.tp = self.fp = self.fn = self.tn = 0

    def _apply(self, y_true: int, y_pred: int, delta: int) -> None:
        if y_true == 1 and y_pred == 1:
            self.tp += delta
        elif y_true == 0 and y_pred == 1:
            self.fp += delta
        elif y_true == 1 and y_pred == 0:
            self.fn += delta
        else:
            self.tn += delta

    def update(self, y_true: int, y_pred: int) -> None:
        if y_true not in (0, 1) or y_pred not in (0, 1):
            raise ValueError("labels must be 0 or 1")
        if len(self._window) == self.window_size:
            old_t, old_p = self._window.popleft()
            self._apply(old_t, old_p, -1)
        self._window.append((y_true, y_pred))
        self._apply(y_true, y_pred, +1)

    def count(self) -> int:
        return len(self._window)

    @staticmethod
    def _div(a: int, b: int) -> float:
        return a / b if b else 0.0

    def accuracy(self) -> float:
        return self._div(self.tp + self.tn, len(self._window))

    def precision(self) -> float:
        return self._div(self.tp, self.tp + self.fp)

    def recall(self) -> float:
        return self._div(self.tp, self.tp + self.fn)

    def f1(self) -> float:
        # Equivalent to 2PR/(P+R), but exact and division-safe.
        return self._div(2 * self.tp, 2 * self.tp + self.fp + self.fn)
