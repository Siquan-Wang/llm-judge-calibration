"""Human-audit corrected win rates using PPI and bounded mean power tuning.

The default implements the fixed-weight (lambda=1) mean correction of
Angelopoulos et al., "Prediction-powered inference" (2023):
https://arxiv.org/abs/2301.09633
https://ppi-py.readthedocs.io/en/latest/ppi.html

Optional power tuning follows the mean-estimation principle in PPI++
(https://arxiv.org/abs/2311.01453), minimizing this implementation's
per-pool variance estimate. Clustered tuning is a sandwich adaptation;
neither tuning formula is a claim of numerical identity to ppi_py.

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
_TUNED_METHOD = "ppi-plus-plus-mean-normal-asymptotic"
_WEIGHTED_METHOD = "ppi-weighted-mean-normal-asymptotic"
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
    score, independent of the selected power. With power other than one,
    ``point`` is NOT ``raw_rate + residual_correction``. The general formula
    is mean(Y_L) + power * (mean(F_U) - mean(F_L)). Group counts are ``None``
    when the caller requests iid variance. ``estimated_variance_ratio``
    compares the selected estimated variance with the human-only estimate;
    it is not an observed error ratio or a finite-sample efficiency guarantee.
    The ratio is ``None`` when the human-only estimated variance is zero.
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
    selected_power: float = 1.0
    power_method: str = "fixed"
    estimated_variance_ratio: float | None = None

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


def _covariance_of_means(
    first: np.ndarray, second: np.ndarray, groups: np.ndarray | None
) -> float:
    """Paired covariance using the same iid/cluster convention as variance."""
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    if groups is None:
        return float(
            np.dot(first_centered, second_centered)
            / (first.size * (first.size - 1))
        )
    first_sums = np.bincount(groups, weights=first_centered)
    second_sums = np.bincount(groups, weights=second_centered)
    n_groups = first_sums.size
    return float(
        n_groups / (n_groups - 1)
        * np.dot(first_sums, second_sums) / first.size**2
    )


def prediction_powered_win_rate(
    judge_labeled: Sequence[str],
    human_labeled: Sequence[str],
    judge_unlabeled: Sequence[str],
    target: str = "A",
    alpha: float = 0.05,
    labeled_groups: Sequence | None = None,
    unlabeled_groups: Sequence | None = None,
    *,
    power: float | str = 1.0,
) -> PPIWinRateResult:
    """Estimate a population human preference rate with a small labeled audit.

    Labels must be ``'A'``, ``'B'`` or ``'tie'``. The target scores 1, the
    other response 0, and ties 0.5. A/B orientation must be consistent across
    all observations. Aligned human/judge audit scores are Y_L and F_L; the
    separate judge-only pool is F_U. For a power lambda, the estimator is

        lambda * mean(F_U) + mean(Y_L - lambda * F_L).

    ``power=1.0`` (the default) recovers original PPI; ``power=0.0`` recovers
    the human-only mean and its normal interval. Any finite fixed power in
    [0, 1] is allowed. The variance estimate is

        Q(lambda) = lambda**2 * var_mean(F_U)
                    + var_mean(Y_L - lambda * F_L).

    ``power='auto'`` selects the minimizer of Q over [0, 1]:

        clip(cov_mean(Y_L, F_L)
             / (var_mean(F_L) + var_mean(F_U)), 0, 1).

    A zero denominator selects zero power. The formula uses audit human
    labels and both judge pools; it needs no evaluation human labels. This
    is a PPI++ mean plug-in variant using per-pool variances, rather than
    ppi_py's pooled judge variance. Its asymptotic justification requires
    consistent covariance estimates and a nondegenerate limiting variance.
    Tuning on the audit is permitted for this scalar power; fitting the
    underlying judge on those labels is a different operation and is not.

    Variances use sample variance / n in iid mode. If both group arrays are supplied, each
    variance is G/(G-1) * sum_g(sum_i_in_g(x_i - mean(x)))**2 / n**2.
    The point estimate remains observation-weighted, including unequal
    clusters; grouping changes only variance estimation. Group IDs must be
    nonmissing, finite, hashable scalars, globally consistent across pools.
    Both arrays are required together, and overlapping IDs are rejected.
    Clustered covariance uses the analogous product of centered cluster
    sums. Auto power then minimizes that cluster sandwich estimate, an
    adaptation requiring sufficiently many independent clusters and no
    dominating cluster; the iid PPI++ theorem alone does not establish this.

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
    estimate nor bounds are clipped to [0, 1]. Auto power minimizes estimated
    variance on these data, not realized MSE, and is not finite-sample
    unbiased in general. No setting guarantees finite-sample coverage or
    improvement over the human-only estimator. It estimates human preference
    under this tie convention, not objective correctness or latent truth.
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
    auto_power = isinstance(power, str) and power == "auto"
    if not auto_power:
        if (
            isinstance(power, (bool, np.bool_))
            or not isinstance(power, Real)
            or not np.isfinite(power)
            or not 0 <= power <= 1
        ):
            raise ValueError("power must be a finite number in [0, 1] or 'auto'")
        power = float(power)
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

    judge_l_variance = _variance_of_mean(f_l, labeled_codes)
    judge_u_variance = _variance_of_mean(f_u, unlabeled_codes)
    human_variance = _variance_of_mean(y_l, labeled_codes)
    if auto_power:
        denominator = judge_l_variance + judge_u_variance
        covariance = _covariance_of_means(y_l, f_l, labeled_codes)
        selected_power = (
            float(np.clip(covariance / denominator, 0.0, 1.0))
            if denominator > 0 else 0.0
        )
        power_method = (
            "auto-per-pool-cluster-sandwich" if labeled_codes is not None
            else "auto-per-pool-sample-variance"
        )
        method = _TUNED_METHOD
    else:
        selected_power = power
        power_method = "fixed"
        method = _METHOD if selected_power == 1.0 else _WEIGHTED_METHOD

    residual = y_l - f_l
    raw_rate = float(f_u.mean())
    correction = float(residual.mean())
    # Keep the original operation order for the backward-compatible default.
    point = (
        raw_rate + correction if selected_power == 1.0
        else float(y_l.mean() + selected_power * (raw_rate - f_l.mean()))
    )
    variance = selected_power**2 * judge_u_variance + _variance_of_mean(
        y_l - selected_power * f_l, labeled_codes
    )
    variance_ratio = float(variance / human_variance) if human_variance > 0 else None
    standard_error = float(np.sqrt(variance))
    # A log-tail quantile avoids cancellation in 1-alpha/2 and underflow
    # in alpha/2, including the smallest positive representable alpha.
    z = -float(special.ndtri_exp(np.log(alpha) - np.log(2.0)))
    half_width = z * standard_error
    interval = Interval(point, point - half_width, point + half_width, alpha, method)
    assumptions = (
        "Independent, disjoint audit and judge-only pools from the same target population.",
        "Same frozen judge for both pools; no fitting or calibration on the audit labels.",
        "Independent rows, or independent one-way clusters when group IDs are supplied.",
        "Observation-weighted human preference; target win=1, loss=0, tie=0.5.",
        "Large-sample normal interval; no finite-sample or distribution-free guarantee.",
        (
            "Power minimizes estimated variance over [0,1]; no finite-sample MSE or coverage guarantee."
            if auto_power else
            f"Fixed lambda={selected_power:g}; no tuning or guaranteed efficiency improvement."
        ),
        "Point and interval are not clipped to the parameter range.",
    )
    if auto_power and labeled_codes is not None:
        assumptions += (
            "Cluster-sandwich power adaptation requires many independent clusters and no dominating cluster.",
        )
    references = _REFERENCES
    if auto_power or selected_power != 1.0:
        references += ("https://arxiv.org/abs/2311.01453",)
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
        method=method,
        assumptions=assumptions,
        references=references,
        selected_power=selected_power,
        power_method=power_method,
        estimated_variance_ratio=variance_ratio,
    )
