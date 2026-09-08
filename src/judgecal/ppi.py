"""Human-audit corrected win rates using the original PPI mean estimator.

This implements the fixed-weight (lambda=1) mean correction of Angelopoulos
et al., "Prediction-powered inference" (2023), not PPI++ or a new estimator:
https://arxiv.org/abs/2301.09633
https://ppi-py.readthedocs.io/en/latest/ppi.html

Intervals use a large-sample normal approximation, optionally with one-way
cluster-robust variances. They are not distribution-free guarantees.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real
from typing import Sequence

import numpy as np
from scipy import special

from .intervals import Interval


_METHOD = "ppi-mean-normal-asymptotic"
_REFERENCES = (
    "https://arxiv.org/abs/2301.09633",
    "https://ppi-py.readthedocs.io/en/latest/ppi.html",
    "https://github.com/aangelopoulos/ppi_py",
)


@dataclass(frozen=True)
class PPIWinRateResult:
    """Untruncated estimate and interval, with auditable sampling metadata.

    ``raw_rate`` uses only the unlabeled judge pool. ``human_only_rate`` uses
    the labeled audit; ``residual_correction`` is its mean human-minus-judge
    score. Group counts are ``None`` when the caller requests iid variance.
    """

    point: float
    raw_rate: float
    human_only_rate: float
    residual_correction: float
    standard_error: float
    interval: Interval
    n_labeled: int
    n_unlabeled: int
    n_labeled_groups: int | None
    n_unlabeled_groups: int | None
    target: str
    variance_method: str
    method: str
    assumptions: tuple[str, ...]
    references: tuple[str, ...]

    def as_dict(self) -> dict:
        """Serialize the result, including the nested Interval, to a dict."""
        return asdict(self)


def _scores(labels: Sequence[str], name: str, target: str) -> np.ndarray:
    try:
        arr = np.asarray(labels, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional label sequence") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional label sequence")
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two observations")
    allowed = {"A", "B", "tie"}
    if any(not isinstance(value, str) or value not in allowed for value in arr):
        raise ValueError(f"{name} labels must be exactly 'A', 'B', or 'tie'")
    return np.asarray(
        [0.5 if value == "tie" else float(value == target) for value in arr],
        dtype=float,
    )


def _group_codes(
    groups: Sequence, n: int, name: str
) -> tuple[np.ndarray, set]:
    try:
        arr = np.asarray(groups, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional group sequence") from exc
    if arr.ndim != 1 or arr.size != n:
        raise ValueError(f"{name} must be one-dimensional with length {n}")
    lookup = {}
    codes = np.empty(n, dtype=int)
    for index, value in enumerate(arr):
        # Reject null/non-reflexive IDs (including NaN/NaT) and containers.
        try:
            valid = value is not None and bool(value == value)
            if isinstance(value, Real):
                valid = valid and bool(np.isfinite(value))
            hash(value)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"{name} requires nonmissing, finite, hashable scalar IDs")
        if value not in lookup:
            lookup[value] = len(lookup)
        codes[index] = lookup[value]
    if len(lookup) < 2:
        raise ValueError(f"{name} must contain at least two independent groups")
    return codes, set(lookup)


def _variance_of_mean(values: np.ndarray, groups: np.ndarray | None) -> float:
    if groups is None:
        return float(np.var(values, ddof=1) / values.size)
    group_sums = np.bincount(groups, weights=values - values.mean())
    n_groups = group_sums.size
    return float(
        n_groups / (n_groups - 1) * np.dot(group_sums, group_sums) / values.size**2
    )


def prediction_powered_win_rate(
    judge_labeled: Sequence[str],
    human_labeled: Sequence[str],
    judge_unlabeled: Sequence[str],
    target: str = "A",
    alpha: float = 0.05,
    labeled_groups: Sequence | None = None,
    unlabeled_groups: Sequence | None = None,
) -> PPIWinRateResult:
    """Estimate a population human preference rate with a small labeled audit.

    Labels must be ``'A'``, ``'B'`` or ``'tie'``. The target scores 1, the
    other response 0, and ties 0.5. A/B orientation must be consistent across
    all observations. Aligned human/judge audit scores are Y_L and F_L; the
    separate judge-only pool is F_U. The fixed-weight estimator is

        mean(F_U) + mean(Y_L - F_L).

    Its standard error is sqrt(var_mean(F_U) + var_mean(Y_L - F_L)), using
    sample variance / n in iid mode. If both group arrays are supplied, each
    variance is G/(G-1) * sum_g(sum_i_in_g(x_i - mean(x)))**2 / n**2.
    The point estimate remains observation-weighted, including unequal
    clusters; grouping changes only variance estimation. Group IDs must be
    nonmissing, finite, hashable scalars, globally consistent across pools.
    Both arrays are required together, and overlapping IDs are rejected.

    The pools must be independent, disjoint, and representative of the same
    target population; without group IDs their disjointness is the caller's
    responsibility. Predictions must come from the same frozen judge, not
    one trained/calibrated on this audit. Rows must be independent in iid
    mode, or groups independent in clustered mode. Grouping does not repair
    distribution shift, selection bias, crossed dependence, or audit leakage.

    At least two rows per pool (and two groups per pool in clustered mode)
    are required to compute variance, but are NOT enough to establish the
    large-sample approximation. Intervals need sufficiently many independent
    units/groups and nondegenerate sampling variance. Zero empirical variance
    can yield a zero-width interval; this is not evidence of certainty.

    Returns a two-sided asymptotic normal interval at level 1-alpha. Neither
    estimate nor bounds are clipped to [0, 1]. This is original PPI with
    lambda=1, with no tuning and no guaranteed precision/MSE improvement over
    the human-only estimator. It estimates human preference under this tie
    convention, not objective correctness or latent truth.
    """
    if not isinstance(target, str) or target not in {"A", "B"}:
        raise ValueError("target must be 'A' or 'B'")
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, Real)
        or not np.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    alpha = float(alpha)
    f_l = _scores(judge_labeled, "judge_labeled", target)
    y_l = _scores(human_labeled, "human_labeled", target)
    f_u = _scores(judge_unlabeled, "judge_unlabeled", target)
    if f_l.shape != y_l.shape:
        raise ValueError("judge_labeled and human_labeled must have the same length")
    if (labeled_groups is None) != (unlabeled_groups is None):
        raise ValueError("labeled_groups and unlabeled_groups must be supplied together")

    labeled_codes = unlabeled_codes = None
    n_labeled_groups = n_unlabeled_groups = None
    variance_method = "iid-sample-variance"
    if labeled_groups is not None:
        labeled_codes, labeled_ids = _group_codes(
            labeled_groups, f_l.size, "labeled_groups"
        )
        unlabeled_codes, unlabeled_ids = _group_codes(
            unlabeled_groups, f_u.size, "unlabeled_groups"
        )
        if labeled_ids & unlabeled_ids:
            raise ValueError("labeled and unlabeled group IDs must be disjoint")
        n_labeled_groups = len(labeled_ids)
        n_unlabeled_groups = len(unlabeled_ids)
        variance_method = "one-way-cluster-robust"

    residual = y_l - f_l
    raw_rate = float(f_u.mean())
    correction = float(residual.mean())
    point = raw_rate + correction
    variance = _variance_of_mean(f_u, unlabeled_codes) + _variance_of_mean(
        residual, labeled_codes
    )
    standard_error = float(np.sqrt(variance))
    # A log-tail quantile avoids cancellation in 1-alpha/2 and underflow
    # in alpha/2, including the smallest positive representable alpha.
    z = -float(special.ndtri_exp(np.log(alpha) - np.log(2.0)))
    half_width = z * standard_error
    interval = Interval(point, point - half_width, point + half_width, alpha, _METHOD)
    assumptions = (
        "Independent, disjoint audit and judge-only pools from the same target population.",
        "Same frozen judge for both pools; no fitting or calibration on the audit labels.",
        "Independent rows, or independent one-way clusters when group IDs are supplied.",
        "Observation-weighted human preference; target win=1, loss=0, tie=0.5.",
        "Large-sample normal interval; no finite-sample or distribution-free guarantee.",
        "Fixed lambda=1; no tuning, clipping, or guaranteed efficiency improvement.",
    )
    return PPIWinRateResult(
        point=point,
        raw_rate=raw_rate,
        human_only_rate=float(y_l.mean()),
        residual_correction=correction,
        standard_error=standard_error,
        interval=interval,
        n_labeled=int(f_l.size),
        n_unlabeled=int(f_u.size),
        n_labeled_groups=n_labeled_groups,
        n_unlabeled_groups=n_unlabeled_groups,
        target=target,
        variance_method=variance_method,
        method=_METHOD,
        assumptions=assumptions,
        references=_REFERENCES,
    )
