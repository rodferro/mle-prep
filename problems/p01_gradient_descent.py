"""
Problem 1 — Linear regression with gradient descent (from scratch)

Implement `fit_line(x, y, learning_rate, n_iters)` that fits y ≈ slope * x + intercept
by minimizing the mean squared error with batch gradient descent.

    MSE(m, b) = (1/n) * Σ (m*x_i + b - y_i)^2

Rules
-----
- Do NOT use numpy.polyfit, np.linalg.lstsq, sklearn, or the closed-form solution
  to produce the answer. Plain Python (or numpy for vector math) + gradient descent.
- Return a tuple (slope, intercept) of floats.
- Raise ValueError if x and y have different lengths, or fewer than 2 points.
- If all x values are identical the slope is undefined: raise ValueError.

Follow-ups the tests check
--------------------------
- It must also converge when x values are large (e.g. x in [1000, 2000]) using the
  default learning rate. Hint: think about what large x does to the gradient.
- Stop early when the improvement in loss is below `tol`.

Example
-------
>>> fit_line([0, 1, 2, 3], [1, 3, 5, 7])
(2.0, 1.0)   # approximately
"""

from typing import Sequence, Tuple


def fit_line(
    x: Sequence[float],
    y: Sequence[float],
    learning_rate: float = 0.01,
    n_iters: int = 20_000,
    tol: float = 1e-12,
) -> Tuple[float, float]:
    raise NotImplementedError
