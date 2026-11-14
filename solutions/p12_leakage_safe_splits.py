"""
Reference solution — Problem 12 (leakage-safe splits).

time_split: one pass, comparing each timestamp to the boundaries; rows in a gap
are skipped. O(n).

group_kfold: greedy bin packing, which is the same idea as sklearn's GroupKFold.
Sorting groups by size, largest first, and always filling the currently smallest
fold keeps fold sizes within one group of each other. O(n + g log g) for g groups
(a heap would make choosing the fold O(log k) instead of O(k)).

Talking points
- Group leakage is common: the same customer, device or document in both train and
  val lets the model memorize the entity instead of learning the pattern.
- For time series with groups, split by time first, then drop val rows whose group
  appears in train if the question is "how do we do on NEW customers?"
"""

from typing import Dict, Hashable, List, Sequence, Tuple


def time_split(
    timestamps: Sequence[float],
    train_end: float,
    val_end: float,
    gap: float = 0,
) -> Tuple[List[int], List[int], List[int]]:
    if gap < 0:
        raise ValueError("gap must be >= 0")
    if train_end + gap > val_end:
        raise ValueError("train_end + gap must be <= val_end")
    train, val, test = [], [], []
    for i, ts in enumerate(timestamps):
        if ts < train_end:
            train.append(i)
        elif train_end + gap <= ts < val_end:
            val.append(i)
        elif ts >= val_end + gap:
            test.append(i)
    return train, val, test


def group_kfold(groups: Sequence[Hashable], n_splits: int) -> List[Tuple[List[int], List[int]]]:
    rows: Dict[Hashable, List[int]] = {}
    first_seen: Dict[Hashable, int] = {}
    for i, g in enumerate(groups):
        rows.setdefault(g, []).append(i)
        first_seen.setdefault(g, i)

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_splits > len(rows):
        raise ValueError("n_splits cannot exceed the number of groups")

    order = sorted(rows, key=lambda g: (-len(rows[g]), first_seen[g]))
    fold_rows: List[List[int]] = [[] for _ in range(n_splits)]
    for g in order:
        target = min(range(n_splits), key=lambda f: (len(fold_rows[f]), f))
        fold_rows[target].extend(rows[g])

    n = len(groups)
    folds = []
    for val in fold_rows:
        val_sorted = sorted(val)
        in_val = set(val_sorted)
        folds.append(([i for i in range(n) if i not in in_val], val_sorted))
    return folds
