"""
Problem 11 — ROC-AUC and average precision from scratch

Part A — `roc_auc(y_true, scores) -> float`
    ROC-AUC = probability that a random positive gets a higher score than a random
    negative, counting ties as 1/2:

        AUC = ( #{(p, n): s_p > s_n} + 0.5 * #{(p, n): s_p == s_n} ) / (P * N)

    - y_true: 0/1 labels. scores: floats (any scale).
    - Raise ValueError if the lengths differ, the input is empty, a label isn't 0/1,
      or only one class is present (AUC is undefined).
    - Must be O(n log n): a test uses 100,000 samples, so the O(P*N) pair loop above
      is too slow. Hint: sort by score and use ranks (Mann-Whitney U), with average
      ranks for ties.

Part B — `average_precision(y_true, scores) -> float`
    Area under the precision-recall curve, computed as in sklearn:

        AP = Σ_k (R_k - R_{k-1}) * P_k

    where thresholds are the DISTINCT scores in descending order; at each threshold t
    everything with score >= t is predicted positive, P_k / R_k are precision and
    recall at that threshold, and R_0 = 0. (Tied scores enter together.)
    - Same validation as Part A, except: raise ValueError only if there are no
      positives (all-positive input is fine; AP = 1.0).

Follow-ups to discuss: why ROC-AUC looks great on very imbalanced data (fraud at 0.1%)
while PR-AUC doesn't, partial AUC at low false-positive rates, and AUC not caring
about calibration.
"""

from typing import Sequence


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    raise NotImplementedError


def average_precision(y_true: Sequence[int], scores: Sequence[float]) -> float:
    raise NotImplementedError
