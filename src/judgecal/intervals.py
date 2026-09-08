"""Uncertainty quantification for evaluation statistics.

Provides frequentist and Bayesian intervals for proportions (agreement,
win-rate) plus a generic percentile bootstrap for arbitrary statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable, Sequence

import numpy as np
from scipy import stats


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    return float(alpha)


def _validate_probability(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


def _validate_n_boot(n_boot: int) -> int:
    if isinstance(n_boot, (bool, np.bool_)) or not isinstance(n_boot, Integral) or n_boot < 2:
        raise ValueError("n_boot must be an integer of at least 2")
    return int(n_boot)


def _as_1d(data: Sequence, name: str, dtype=None) -> np.ndarray:
    if isinstance(data, (str, bytes)):
        raise ValueError(f"{name} must be a one-dimensional sequence, not a string")
    try:
        arr = np.asarray(list(data), dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def _validate_counts(k: int, n: int) -> None:
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in (k, n)) or not 0 <= k <= n:
        raise ValueError("k and n must be integers satisfying 0 <= k <= n")


@dataclass
class Interval:
    point: float
    low: float
    high: float
    alpha: float
    method: str

    def excludes(self, value: float) -> bool:
        """True if `value` lies outside the interval (e.g. 0.5 for a win-rate)."""
        if not all(np.isfinite(v) for v in (value, self.low, self.high)):
            return False
        return self.low <= self.high and (value < self.low or value > self.high)

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "alpha": self.alpha,
            "method": self.method,
        }


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a binomial proportion."""
    alpha = _validate_alpha(alpha)
    _validate_counts(k, n)
    if n == 0:
        return Interval(float("nan"), 0.0, 1.0, alpha, "wilson")
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return Interval(phat, max(0.0, center - half), min(1.0, center + half), alpha, "wilson")


def _binomial_exact_interval(k: int, n: int, alpha: float) -> Interval:
    """Clopper-Pearson interval, with nonzero width at boundary counts."""
    alpha = _validate_alpha(alpha)
    _validate_counts(k, n)
    if n == 0:
        return Interval(float("nan"), 0.0, 1.0, alpha, "clopper-pearson")
    low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return Interval(k / n, low, high, alpha, "clopper-pearson")


def _bounded_mean_interval(data: np.ndarray, low: float, high: float, alpha: float) -> Interval:
    """Conservative Hoeffding fallback for a degenerate bounded iid sample."""
    alpha = _validate_alpha(alpha)
    if len(data) == 0:
        return Interval(float("nan"), low, high, alpha, "hoeffding")
    point = float(np.mean(data))
    radius = (high - low) * np.sqrt(np.log(2 / alpha) / (2 * len(data)))
    return Interval(point, max(low, point - radius), min(high, point + radius), alpha, "hoeffding")


def beta_binomial_interval(
    k: int,
    n: int,
    alpha: float = 0.05,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
) -> Interval:
    """Bayesian credible interval for a proportion under a Beta(prior) prior.

    With a Beta(prior_a, prior_b) prior and k successes in n trials, the
    posterior is Beta(prior_a + k, prior_b + n - k); the interval is the
    equal-tailed credible interval of that posterior.

    Point is the posterior mean. With no observations this returns the
    prior, not evidence of observed agreement.
    """
    alpha = _validate_alpha(alpha)
    _validate_counts(k, n)
    for value in (prior_a, prior_b):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError("Beta prior parameters must be finite and positive")
    a_post = prior_a + k
    b_post = prior_b + (n - k)
    low = stats.beta.ppf(alpha / 2, a_post, b_post)
    high = stats.beta.ppf(1 - alpha / 2, a_post, b_post)
    point = a_post / (a_post + b_post)
    return Interval(point, float(low), float(high), alpha, "beta-binomial")


def bootstrap_ci(
    data: Sequence,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> Interval:
    """Approximate percentile bootstrap for an arbitrary 1-D iid sample.

    Constant data produce a degenerate interval. Bounded-mean inference
    callers should handle that boundary explicitly; repeated prompts need
    cluster-aware resampling rather than this item-level routine.
    """
    alpha = _validate_alpha(alpha)
    n_boot = _validate_n_boot(n_boot)
    arr = _as_1d(data, "data")
    if arr.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), alpha, "bootstrap")
    rng = np.random.default_rng(random_state)
    point = float(statistic(arr))
    if not np.isfinite(point):
        raise ValueError("statistic must return a finite scalar")
    n = arr.shape[0]
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = statistic(arr[idx])
    if not np.all(np.isfinite(boot)):
        raise ValueError("statistic returned a non-finite bootstrap value")
    low, high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(low), float(high), alpha, "bootstrap")
