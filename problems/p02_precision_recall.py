"""
Problem 2 — Precision and recall from a confusion matrix

Convention (same as sklearn): cm[i][j] = number of samples whose TRUE class is i
and whose PREDICTED class is j. Rows = actual, columns = predicted.

For a binary 2x2 matrix this means:
    [[TN, FP],
     [FN, TP]]

Part A — `precision_recall(cm, positive_class=1)`
    Works for any N x N matrix. Returns (precision, recall) for `positive_class`.
    If a denominator is 0, that metric is 0.0 (do not raise).

Part B — `classification_report(cm)`
    Returns a dict:
    {
        "per_class": [{"precision": p, "recall": r, "f1": f, "support": s}, ...],  # one per class
        "macro":     {"precision": p, "recall": r, "f1": f},   # unweighted mean over classes
        "micro":     {"precision": p, "recall": r, "f1": f},   # from global TP/FP/FN
        "accuracy":  a,
    }
    f1 = 2pr / (p + r), or 0.0 when p + r == 0.
    support = number of TRUE samples of that class (row sum).

Validation
    Raise ValueError if the matrix is empty, not square, or contains negative values.
    Raise ValueError if positive_class is out of range.

Follow-up to think about (not tested): when would you prefer macro over micro averaging
for, say, a contract-clause classifier with a long tail of rare clause types?
"""

from typing import Dict, List, Sequence, Tuple

Matrix = Sequence[Sequence[int]]


def precision_recall(cm: Matrix, positive_class: int = 1) -> Tuple[float, float]:
    raise NotImplementedError


def classification_report(cm: Matrix) -> Dict:
    raise NotImplementedError
