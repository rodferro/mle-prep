"""
Reference solution — Problem 2 (precision/recall from a confusion matrix).

For class k (rows = actual, cols = predicted):
    TP = cm[k][k]
    FP = column k sum - TP     (predicted k, actually something else)
    FN = row k sum - TP        (actually k, predicted something else)
    precision = TP / (TP + FP),  recall = TP / (TP + FN)

Micro averaging pools TP/FP/FN over all classes. For single-label multiclass, every
error is one FP (for the predicted class) and one FN (for the true class), so
micro precision = micro recall = micro F1 = accuracy.
Macro averaging gives every class equal weight, so rare classes count as much as common
ones; micro is dominated by the frequent classes.
"""

from typing import Dict, List, Sequence, Tuple

Matrix = Sequence[Sequence[int]]


def _validate(cm: Matrix) -> int:
    n = len(cm)
    if n == 0:
        raise ValueError("empty confusion matrix")
    for row in cm:
        if len(row) != n:
            raise ValueError("confusion matrix must be square")
        for v in row:
            if v < 0:
                raise ValueError("confusion matrix cannot contain negative counts")
    return n


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2 * p * r, p + r)


def _counts(cm: Matrix, k: int) -> Tuple[int, int, int]:
    tp = cm[k][k]
    fp = sum(cm[i][k] for i in range(len(cm))) - tp
    fn = sum(cm[k]) - tp
    return tp, fp, fn


def precision_recall(cm: Matrix, positive_class: int = 1) -> Tuple[float, float]:
    n = _validate(cm)
    if not 0 <= positive_class < n:
        raise ValueError("positive_class out of range")
    tp, fp, fn = _counts(cm, positive_class)
    return _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)


def classification_report(cm: Matrix) -> Dict:
    n = _validate(cm)
    per_class: List[Dict] = []
    tot_tp = tot_fp = tot_fn = 0

    for k in range(n):
        tp, fp, fn = _counts(cm, k)
        tot_tp, tot_fp, tot_fn = tot_tp + tp, tot_fp + fp, tot_fn + fn
        p, r = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
        per_class.append({"precision": p, "recall": r, "f1": _f1(p, r), "support": sum(cm[k])})

    macro_p = sum(c["precision"] for c in per_class) / n
    macro_r = sum(c["recall"] for c in per_class) / n
    macro_f1 = sum(c["f1"] for c in per_class) / n  # mean of per-class F1 (sklearn style)

    micro_p = _safe_div(tot_tp, tot_tp + tot_fp)
    micro_r = _safe_div(tot_tp, tot_tp + tot_fn)

    total = sum(sum(row) for row in cm)
    return {
        "per_class": per_class,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "micro": {"precision": micro_p, "recall": micro_r, "f1": _f1(micro_p, micro_r)},
        "accuracy": _safe_div(sum(cm[k][k] for k in range(n)), total),
    }
