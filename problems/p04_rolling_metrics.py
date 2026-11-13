"""
Problem 4 — Rolling model performance metrics

You monitor a binary classifier in production. Labels arrive one at a time as
(y_true, y_pred) pairs, each 0 or 1. Track metrics over the LAST `window_size` pairs.

Implement class `RollingMetrics`:
    __init__(window_size)          raise ValueError if window_size < 1
    update(y_true, y_pred)         add one observation; raise ValueError if a label isn't 0/1
    count() -> int                 observations currently in the window
    accuracy() -> float
    precision() -> float
    recall() -> float
    f1() -> float

Any metric whose denominator is 0 (including an empty window) returns 0.0.

Efficiency requirement
    update() and every metric call must be O(1). Recomputing over the whole window on
    each call is too slow: a test streams 200,000 observations with window_size=5,000
    and reads all four metrics after every update.

Follow-ups to discuss (not tested): time-based windows (last 1 hour) instead of
count-based, delayed labels (fraud/chargebacks arrive weeks later), rolling AUC, and
alerting on a drop with something better than a fixed threshold.
"""


class RollingMetrics:
    def __init__(self, window_size: int):
        raise NotImplementedError

    def update(self, y_true: int, y_pred: int) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def accuracy(self) -> float:
        raise NotImplementedError

    def precision(self) -> float:
        raise NotImplementedError

    def recall(self) -> float:
        raise NotImplementedError

    def f1(self) -> float:
        raise NotImplementedError
