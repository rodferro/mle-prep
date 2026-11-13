import random

import pytest

from conftest import load

m = load("p01_gradient_descent")


def closed_form(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    slope = sxy / sxx
    return slope, my - slope * mx


def test_perfect_line():
    slope, intercept = m.fit_line([0, 1, 2, 3], [1, 3, 5, 7])
    assert slope == pytest.approx(2.0, abs=1e-3)
    assert intercept == pytest.approx(1.0, abs=1e-3)


def test_returns_floats():
    out = m.fit_line([0, 1, 2], [0, 1, 2])
    assert isinstance(out, tuple) and len(out) == 2
    assert all(isinstance(v, float) for v in out)


def test_negative_slope():
    x = [1, 2, 3, 4, 5]
    y = [10 - 3 * v for v in x]
    slope, intercept = m.fit_line(x, y)
    assert slope == pytest.approx(-3.0, abs=1e-3)
    assert intercept == pytest.approx(10.0, abs=1e-3)


def test_noisy_data_matches_closed_form():
    rng = random.Random(0)
    x = [rng.uniform(0, 10) for _ in range(200)]
    y = [1.5 * v - 4 + rng.gauss(0, 1) for v in x]
    exp_s, exp_b = closed_form(x, y)
    slope, intercept = m.fit_line(x, y)
    assert slope == pytest.approx(exp_s, abs=1e-3)
    assert intercept == pytest.approx(exp_b, abs=1e-2)


def test_constant_y_gives_zero_slope():
    slope, intercept = m.fit_line([1, 2, 3, 4], [5, 5, 5, 5])
    assert slope == pytest.approx(0.0, abs=1e-6)
    assert intercept == pytest.approx(5.0, abs=1e-3)


def test_large_x_values_still_converge():
    """Fails with naive GD at lr=0.01 (it diverges). Feature scaling fixes it."""
    rng = random.Random(1)
    x = [rng.uniform(1000, 2000) for _ in range(100)]
    y = [0.5 * v + 30 + rng.gauss(0, 2) for v in x]
    exp_s, exp_b = closed_form(x, y)
    slope, intercept = m.fit_line(x, y)
    assert slope == pytest.approx(exp_s, rel=1e-4)
    assert intercept == pytest.approx(exp_b, abs=0.5)


@pytest.mark.parametrize(
    "x, y",
    [
        ([1, 2, 3], [1, 2]),  # length mismatch
        ([1], [1]),           # too few points
        ([], []),
        ([2, 2, 2], [1, 2, 3]),  # identical x -> slope undefined
    ],
)
def test_invalid_input(x, y):
    with pytest.raises(ValueError):
        m.fit_line(x, y)
