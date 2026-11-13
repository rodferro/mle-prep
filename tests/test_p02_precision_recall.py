import pytest

from conftest import load

m = load("p02_precision_recall")


def test_binary_basic():
    # [[TN, FP], [FN, TP]]
    cm = [[50, 10],
          [5, 35]]
    p, r = m.precision_recall(cm)
    assert p == pytest.approx(35 / 45)
    assert r == pytest.approx(35 / 40)


def test_binary_negative_class_as_positive():
    cm = [[50, 10],
          [5, 35]]
    p, r = m.precision_recall(cm, positive_class=0)
    assert p == pytest.approx(50 / 55)
    assert r == pytest.approx(50 / 60)


def test_zero_division_returns_zero():
    cm = [[10, 0],
          [0, 0]]  # model never predicts 1, and there are no actual 1s
    assert m.precision_recall(cm) == (0.0, 0.0)


def test_no_positive_predictions_but_actual_positives():
    cm = [[10, 0],
          [7, 0]]
    p, r = m.precision_recall(cm)
    assert p == 0.0
    assert r == 0.0


def test_multiclass_per_class():
    cm = [[5, 1, 0],
          [2, 3, 1],
          [0, 0, 8]]
    p, r = m.precision_recall(cm, positive_class=1)
    assert p == pytest.approx(3 / 4)   # column 1 sum = 4
    assert r == pytest.approx(3 / 6)   # row 1 sum = 6


def test_report_multiclass():
    cm = [[5, 1, 0],
          [2, 3, 1],
          [0, 0, 8]]
    rep = m.classification_report(cm)

    expected = [
        (5 / 7, 5 / 6, 6),
        (3 / 4, 3 / 6, 6),
        (8 / 9, 8 / 8, 8),
    ]
    assert len(rep["per_class"]) == 3
    for got, (p, r, s) in zip(rep["per_class"], expected):
        assert got["precision"] == pytest.approx(p)
        assert got["recall"] == pytest.approx(r)
        assert got["f1"] == pytest.approx(2 * p * r / (p + r))
        assert got["support"] == s

    assert rep["macro"]["precision"] == pytest.approx(sum(e[0] for e in expected) / 3)
    assert rep["macro"]["recall"] == pytest.approx(sum(e[1] for e in expected) / 3)
    f1s = [2 * p * r / (p + r) for p, r, _ in expected]
    assert rep["macro"]["f1"] == pytest.approx(sum(f1s) / 3)

    acc = 16 / 20
    assert rep["accuracy"] == pytest.approx(acc)
    # single-label multiclass: micro P = micro R = micro F1 = accuracy
    for k in ("precision", "recall", "f1"):
        assert rep["micro"][k] == pytest.approx(acc)


def test_report_handles_empty_class():
    cm = [[4, 0, 0],
          [0, 0, 0],
          [1, 0, 5]]
    rep = m.classification_report(cm)
    assert rep["per_class"][1] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


@pytest.mark.parametrize(
    "cm",
    [
        [],
        [[1, 2], [3]],
        [[1, 2, 3], [4, 5, 6]],
        [[1, -1], [0, 2]],
    ],
)
def test_invalid_matrix(cm):
    with pytest.raises(ValueError):
        m.precision_recall(cm, positive_class=0)
    with pytest.raises(ValueError):
        m.classification_report(cm)


def test_positive_class_out_of_range():
    with pytest.raises(ValueError):
        m.precision_recall([[1, 0], [0, 1]], positive_class=2)
