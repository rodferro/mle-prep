"""
Problem 13 — Drift detection with the Population Stability Index (PSI)

PSI compares a feature's (or score's) distribution in production ("actual") with the
training/reference distribution ("expected"):

    PSI = Σ_bins (a_i - e_i) * ln(a_i / e_i)

where e_i and a_i are the fractions of each sample that fall in bin i.

Implement `psi(expected, actual, n_bins=10, eps=1e-4) -> float`
    Bins come from the EXPECTED sample's quantiles, defined exactly like this:
        - sort expected; let n = len(expected)
        - candidate edges: sorted_expected[floor(i * n / n_bins)] for i = 1..n_bins-1
        - remove duplicate edges (ties can make fewer bins)
        - a value v goes to bin = number of edges <= v   (so bin 0 is v < first edge)
    Both samples use the same bins. Fractions are floored at `eps` before the formula
    so empty bins don't produce log(0) or division by zero:  f = max(count / total, eps).
    (A constant expected sample gives a single edge c, so two bins: v < c and v >= c.)
    Raise ValueError if either sample is empty or n_bins < 2.

Implement `drift_level(value) -> str`
    "none" if value < 0.1, "moderate" if 0.1 <= value < 0.25, "significant" otherwise.
    (Common industry rule of thumb.)

Pure Python is fine. Hint: bisect.

Follow-ups to discuss: PSI vs KS test vs KL divergence, monitoring the model's score
distribution vs each input feature, categorical features (bins = categories, plus an
"unseen" bucket), and why big samples make statistical tests flag tiny, harmless shifts.
"""

from typing import Sequence


def psi(expected: Sequence[float], actual: Sequence[float], n_bins: int = 10, eps: float = 1e-4) -> float:
    raise NotImplementedError


def drift_level(value: float) -> str:
    raise NotImplementedError
