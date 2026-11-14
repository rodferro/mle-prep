"""
Reference solution — Problem 11 (ROC-AUC and average precision).

ROC-AUC via ranks (Mann-Whitney U)
- Rank all scores ascending, 1-based; tied scores all get the AVERAGE rank of their run.
- U = (sum of positives' ranks) - P(P+1)/2 counts the (positive, negative) pairs the
  positive wins, with ties worth 1/2 thanks to the average ranks.
- AUC = U / (P * N).   O(n log n) for the sort.

Average precision
- Sort by score descending and walk run-by-run (all tied scores enter together).
- After each run: precision = TP / (TP + FP), recall = TP / P.
- AP += (recall - previous recall) * precision. This is a step-wise sum, not the
  trapezoid rule; the trapezoid rule over-estimates PR area.
"""

from typing import List, Sequence


def _validate(y_true: Sequence[int], scores: Sequence[float]) -> None:
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if len(y_true) == 0:
        raise ValueError("empty input")
    if any(y not in (0, 1) for y in y_true):
        raise ValueError("labels must be 0 or 1")


def _average_ranks(scores: Sequence[float]) -> List[float]:
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # ranks i+1..j+1 averaged
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    _validate(y_true, scores)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC is undefined with only one class")
    ranks = _average_ranks(scores)
    rank_sum = sum(r for r, y in zip(ranks, y_true) if y == 1)
    u = rank_sum - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def average_precision(y_true: Sequence[int], scores: Sequence[float]) -> float:
    _validate(y_true, scores)
    n_pos = sum(y_true)
    if n_pos == 0:
        raise ValueError("average precision is undefined without positives")

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(order):
        s = scores[order[i]]
        while i < len(order) and scores[order[i]] == s:
            if y_true[order[i]] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap
