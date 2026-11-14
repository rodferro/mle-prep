import math
import random

import pytest

from conftest import load

m = load("p15_calibration")


def sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


# ---------- Part A / B ----------

def test_reliability_bins_basic():
    probs = [0.05, 0.15, 0.12, 0.95, 1.0, 0.5]
    labels = [0, 0, 1, 1, 1, 0]
    bins = m.reliability_bins(probs, labels, n_bins=10)
    assert [b["bin"] for b in bins] == [0, 1, 5, 9]
    b1 = bins[1]
    assert b1["count"] == 2
    assert b1["mean_pred"] == pytest.approx(0.135)
    assert b1["frac_pos"] == pytest.approx(0.5)
    b9 = bins[3]                                   # 0.95 and 1.0 both land in the last bin
    assert b9["count"] == 2 and b9["frac_pos"] == 1.0
    assert b9["mean_pred"] == pytest.approx(0.975)


def test_ece_known_value():
    probs = [0.2, 0.2, 0.8, 0.8]
    labels = [0, 0, 1, 0]
    # bin 2: mean 0.2, frac 0.0 -> 0.2 ; bin 8: mean 0.8, frac 0.5 -> 0.3 ; ECE = 0.5*0.2 + 0.5*0.3
    assert m.expected_calibration_error(probs, labels) == pytest.approx(0.25)


def test_ece_perfect_calibration_is_small():
    rng = random.Random(0)
    probs = [rng.random() for _ in range(20_000)]
    labels = [int(rng.random() < p) for p in probs]
    assert m.expected_calibration_error(probs, labels) < 0.02


@pytest.mark.parametrize(
    "probs, labels",
    [([], []), ([0.5], [1, 0]), ([1.2], [1]), ([-0.1], [0]), ([0.5], [3])],
)
def test_bins_invalid(probs, labels):
    with pytest.raises(ValueError):
        m.reliability_bins(probs, labels)


# ---------- Part C ----------

def newton_platt(scores, labels, iters=50):
    """Exact maximum-likelihood fit, used as the reference."""
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for s, y in zip(scores, labels):
            p = sigmoid(a * s + b)
            w = p * (1 - p)
            ga += (p - y) * s
            gb += p - y
            haa += w * s * s
            hab += w * s
            hbb += w
        det = haa * hbb - hab * hab
        a -= (hbb * ga - hab * gb) / det
        b -= (-hab * ga + haa * gb) / det
    return a, b


@pytest.fixture(scope="module")
def miscalibrated():
    rng = random.Random(42)
    scores = [rng.gauss(0, 1) for _ in range(800)]
    labels = [int(rng.random() < sigmoid(2.0 * s - 1.0)) for s in scores]
    return scores, labels


def test_platt_matches_maximum_likelihood(miscalibrated):
    scores, labels = miscalibrated
    a, b = m.fit_platt(scores, labels, learning_rate=0.5, n_iters=1500)
    ea, eb = newton_platt(scores, labels)
    assert a == pytest.approx(ea, abs=0.02)
    assert b == pytest.approx(eb, abs=0.02)
    assert a == pytest.approx(2.0, abs=0.4) and b == pytest.approx(-1.0, abs=0.3)


def test_platt_improves_ece(miscalibrated):
    scores, labels = miscalibrated
    raw = [sigmoid(s) for s in scores]               # naive: treat scores as logits
    a, b = m.fit_platt(scores, labels, learning_rate=0.5, n_iters=1500)
    calibrated = m.apply_platt(scores, a, b)
    assert all(0.0 <= p <= 1.0 for p in calibrated)
    assert m.expected_calibration_error(calibrated, labels) < m.expected_calibration_error(raw, labels)


def test_apply_platt_is_stable_for_extreme_scores():
    out = m.apply_platt([-1e6, 0.0, 1e6], 1.0, 0.0)
    assert out[0] == pytest.approx(0.0) and out[1] == pytest.approx(0.5) and out[2] == pytest.approx(1.0)


def test_fit_platt_invalid():
    with pytest.raises(ValueError):
        m.fit_platt([], [])
    with pytest.raises(ValueError):
        m.fit_platt([0.1, 0.2], [1])
    with pytest.raises(ValueError):
        m.fit_platt([0.1], [5])
