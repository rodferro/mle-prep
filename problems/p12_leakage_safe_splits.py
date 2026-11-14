"""
Problem 12 — Leakage-safe train/validation splits

Random splits leak information when rows are ordered in time or share an entity
(customer, device, document). Implement two splitters. Both return lists of integer
row positions (0..n-1), sorted ascending.

Part A — `time_split(timestamps, train_end, val_end, gap=0)`
    `timestamps` are numbers (e.g. epoch seconds or day numbers); not sorted.
    Returns (train, val, test):
        train:  ts < train_end
        val:    train_end + gap <= ts < val_end
        test:   ts >= val_end + gap
    Rows inside a gap belong to no set (an "embargo", so labels that mature late or
    features computed over trailing windows can't leak across the boundary).
    Raise ValueError if gap < 0 or train_end + gap > val_end.

Part B — `group_kfold(groups, n_splits)`
    `groups[i]` is the entity of row i (any hashable). Return a list of n_splits pairs
    (train, val) such that:
        - every group's rows are all in the same validation fold;
        - each row is in exactly one validation fold;
        - train is every row not in that fold's val;
        - folds are balanced by number of ROWS: assign groups from largest to smallest
          (ties broken by first appearance), each to the fold with the fewest rows so
          far (ties: lowest fold number).
    Raise ValueError if n_splits < 2 or n_splits > number of distinct groups.

Pure Python is fine (no pandas needed).

Follow-ups to discuss: combining both (time split + no customer overlap), stratifying
by label within group k-fold, and why a model can look great in random CV and fail in
production when the same customer appears in train and test.
"""

from typing import Hashable, List, Sequence, Tuple


def time_split(
    timestamps: Sequence[float],
    train_end: float,
    val_end: float,
    gap: float = 0,
) -> Tuple[List[int], List[int], List[int]]:
    raise NotImplementedError


def group_kfold(groups: Sequence[Hashable], n_splits: int) -> List[Tuple[List[int], List[int]]]:
    raise NotImplementedError
