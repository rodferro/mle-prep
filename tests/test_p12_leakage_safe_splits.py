import random

import pytest

from conftest import load

m = load("p12_leakage_safe_splits")


# ---------- time_split ----------

def test_time_split_no_gap():
    ts = [5, 1, 9, 3, 7, 10, 0]
    train, val, test = m.time_split(ts, train_end=4, val_end=8)
    assert train == [1, 3, 6]     # ts 1, 3, 0
    assert val == [0, 4]          # ts 5, 7
    assert test == [2, 5]         # ts 9, 10


def test_time_split_with_gap_drops_embargoed_rows():
    ts = list(range(12))          # row i has ts i
    train, val, test = m.time_split(ts, train_end=4, val_end=8, gap=2)
    assert train == [0, 1, 2, 3]
    assert val == [6, 7]          # 4 and 5 are in the gap
    assert test == [10, 11]       # 8 and 9 are in the gap


def test_time_split_sets_are_disjoint_and_ordered():
    rng = random.Random(0)
    ts = [rng.uniform(0, 100) for _ in range(500)]
    train, val, test = m.time_split(ts, 50, 80, gap=5)
    assert not (set(train) & set(val)) and not (set(val) & set(test)) and not (set(train) & set(test))
    assert train == sorted(train) and val == sorted(val) and test == sorted(test)
    assert max(ts[i] for i in train) < min(ts[i] for i in val)
    assert max(ts[i] for i in val) < min(ts[i] for i in test)


def test_time_split_invalid():
    with pytest.raises(ValueError):
        m.time_split([1, 2], 5, 8, gap=-1)
    with pytest.raises(ValueError):
        m.time_split([1, 2], 5, 8, gap=4)


# ---------- group_kfold ----------

def check_folds(groups, folds, n_splits):
    n = len(groups)
    assert len(folds) == n_splits
    seen = []
    for train, val in folds:
        assert train == sorted(train) and val == sorted(val)
        assert sorted(train + val) == list(range(n))
        assert not ({groups[i] for i in train} & {groups[i] for i in val})
        seen += val
    assert sorted(seen) == list(range(n))


def test_group_kfold_exact_assignment():
    groups = ["a", "b", "a", "c", "b", "a", "d", "c", "e"]
    # sizes: a=3, b=2, c=2, d=1, e=1 ; order a, b, c (b first seen before c), d, e
    folds = m.group_kfold(groups, 2)
    check_folds(groups, folds, 2)
    # a -> fold0 (3) ; b -> fold1 (2) ; c -> fold1 (4) ; d -> fold0 (4) ; e -> fold0 (5)
    assert folds[0][1] == [0, 2, 5, 6, 8]
    assert folds[1][1] == [1, 3, 4, 7]


def test_group_kfold_properties_and_balance():
    rng = random.Random(4)
    groups = [f"cust{int(rng.paretovariate(1.5)) % 60}" for _ in range(2000)]
    for k in (2, 3, 5):
        folds = m.group_kfold(groups, k)
        check_folds(groups, folds, k)
        sizes = [len(v) for _, v in folds]
        biggest_group = max(groups.count(g) for g in set(groups))
        assert max(sizes) - min(sizes) <= biggest_group


def test_group_kfold_accepts_any_hashable():
    groups = [(1, "x"), (2, "y"), (1, "x"), 7, 7, None]
    check_folds(groups, m.group_kfold(groups, 3), 3)


def test_group_kfold_invalid():
    with pytest.raises(ValueError):
        m.group_kfold(["a", "b"], 1)
    with pytest.raises(ValueError):
        m.group_kfold(["a", "b", "a"], 3)
