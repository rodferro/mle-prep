"""
Reference solution — Problem 1 (gradient descent for a best-fit line).

Key ideas
- Gradients of MSE:  dL/dm = (2/n) Σ (ŷ_i - y_i) x_i,   dL/db = (2/n) Σ (ŷ_i - y_i)
- Large x makes dL/dm huge -> the step overshoots and diverges at a fixed learning rate.
  Standardizing x (z = (x - μ) / σ) makes the loss surface well conditioned, so one
  learning rate works for any scale. Fit on z, then map back:
      y = a*z + c = a*(x - μ)/σ + c   ->   slope = a/σ,  intercept = c - a*μ/σ
- Early stopping on loss improvement.

Complexity: O(n_iters * n) time, O(n) extra memory (could be O(1) with streaming sums).
"""

import math
from typing import Sequence, Tuple


def fit_line(
    x: Sequence[float],
    y: Sequence[float],
    learning_rate: float = 0.01,
    n_iters: int = 20_000,
    tol: float = 1e-12,
) -> Tuple[float, float]:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 points")

    mu = sum(x) / n
    sigma = math.sqrt(sum((xi - mu) ** 2 for xi in x) / n)
    if sigma == 0:
        raise ValueError("all x values are identical; slope is undefined")

    z = [(xi - mu) / sigma for xi in x]
    a, c = 0.0, 0.0
    prev_loss = math.inf

    for _ in range(n_iters):
        residuals = [a * zi + c - yi for zi, yi in zip(z, y)]
        loss = sum(r * r for r in residuals) / n
        if prev_loss - loss < tol * max(1.0, loss) and prev_loss != math.inf:
            break
        prev_loss = loss

        grad_a = 2.0 / n * sum(r * zi for r, zi in zip(residuals, z))
        grad_c = 2.0 / n * sum(residuals)
        a -= learning_rate * grad_a
        c -= learning_rate * grad_c

    slope = a / sigma
    intercept = c - a * mu / sigma
    return slope, intercept
