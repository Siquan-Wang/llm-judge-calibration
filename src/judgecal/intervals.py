"""Uncertainty quantification for evaluation statistics.

Provides frequentist and Bayesian intervals for proportions (agreement,
win-rate) plus a generic percentile bootstrap for arbitrary statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats


@dataclass
class Interval:
    point: float
    low: float
    high: float
    alpha: float
    method: str

    def excludes(self, value: float) -> bool:
        """True if `value` lies outside the interval (e.g. 0.5 for a win-rate)."""
        return value < self.low or value > self.high

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
    if n == 0:
        return Interval(float("nan"), 0.0, 1.0, alpha, "wilson")
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return Interval(phat, max(0.0, center - half), min(1.0, center + half), alpha, "wilson")


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
    """
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
    """Percentile bootstrap CI for an arbitrary statistic of a 1-D sample."""
    arr = np.asarray(data)
    if arr.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), alpha, "bootstrap")
    rng = np.random.default_rng(random_state)
    n = arr.shape[0]
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = statistic(arr[idx])
    low, high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(statistic(arr)), float(low), float(high), alpha, "bootstrap")
