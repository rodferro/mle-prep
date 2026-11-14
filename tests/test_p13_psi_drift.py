import math
import random

import pytest

from conftest import load

m = load("p13_psi_drift")


def test_identical_distributions_is_zero():
    x = [random.Random(0).gauss(0, 1) for _ in range(1000)]
    assert m.psi(x, x) == pytest.approx(0.0)


def test_hand_computed_example():
    expected = list(range(1, 11))                  # edges for 2 bins: [6]
    actual = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]     # bin0: 3/10, bin1: 7/10
    want = (0.3 - 0.5) * math.log(0.3 / 0.5) + (0.7 - 0.5) * math.log(0.7 / 0.5)
    assert m.psi(expected, actual, n_bins=2) == pytest.approx(want)


def test_empty_bin_uses_eps():
    expected = list(range(1, 11))
    actual = [1, 2, 3, 4, 5] * 2                   # everything in bin 0, bin 1 empty
    eps = 1e-4
    want = (1.0 - 0.5) * math.log(1.0 / 0.5) + (eps - 0.5) * math.log(eps / 0.5)
    assert m.psi(expected, actual, n_bins=2, eps=eps) == pytest.approx(want)


def test_ties_reduce_bins():
    expected = [0] * 50 + [1] * 50                 # 10 candidate edges collapse to [0, 1]
    actual = [0] * 50 + [1] * 50
    assert m.psi(expected, actual) == pytest.approx(0.0)
    shifted = [1] * 100
    # bins: v<0 (empty both), 0 (0.5 vs eps), 1 (0.5 vs 1.0)
    eps = 1e-4
    want = (eps - 0.5) * math.log(eps / 0.5) + (1.0 - 0.5) * math.log(1.0 / 0.5)
    assert m.psi(expected, shifted) == pytest.approx(want)


def test_constant_expected_gives_two_bins():
    # edges collapse to [3.0]: bin0 = v < 3, bin1 = v >= 3
    eps = 1e-4
    want = (1 / 3 - eps) * math.log((1 / 3) / eps) + (2 / 3 - 1.0) * math.log((2 / 3) / 1.0)
    assert m.psi([3.0] * 20, [1.0, 9.0, 5.0]) == pytest.approx(want)
    assert m.psi([3.0] * 20, [3.0, 4.0]) == pytest.approx(0.0)


def test_shift_grows_psi():
    rng = random.Random(1)
    ref = [rng.gauss(0, 1) for _ in range(5000)]
    small = [rng.gauss(0.1, 1) for _ in range(5000)]
    big = [rng.gauss(1.0, 1) for _ in range(5000)]
    p_small, p_big = m.psi(ref, small), m.psi(ref, big)
    assert 0 < p_small < p_big
    assert m.drift_level(p_small) == "none"
    assert m.drift_level(p_big) == "significant"


def test_drift_level_boundaries():
    assert m.drift_level(0.0) == "none"
    assert m.drift_level(0.0999) == "none"
    assert m.drift_level(0.1) == "moderate"
    assert m.drift_level(0.2499) == "moderate"
    assert m.drift_level(0.25) == "significant"


def test_invalid():
    with pytest.raises(ValueError):
        m.psi([], [1.0])
    with pytest.raises(ValueError):
        m.psi([1.0], [])
    with pytest.raises(ValueError):
        m.psi([1.0, 2.0], [1.0], n_bins=1)
