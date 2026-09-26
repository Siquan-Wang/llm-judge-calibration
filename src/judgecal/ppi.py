"""Prediction-powered bounded means and human-audit corrected win rates.

The default implements the fixed-weight (lambda=1) mean correction of
Angelopoulos et al., "Prediction-powered inference" (2023):
https://arxiv.org/abs/2301.09633
https://ppi-py.readthedocs.io/en/latest/ppi.html

Optional power tuning follows the mean-estimation principle in PPI++
(https://arxiv.org/abs/2311.01453), minimizing this implementation's
per-pool variance estimate. Clustered tuning is a sandwich adaptation;
neither tuning formula is a claim of numerical identity to ppi_py.
The numeric mean API optionally permits a signed coefficient range, following
PPI++ Example 6.1 (https://arxiv.org/pdf/2311.01453v2); the default stays [0,1].

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
class PPIMeanResult:
    """Inference for a population mean of an explicitly defined bounded score.

    ``prediction_mean`` is mean(F_U); ``outcome_mean`` is mean(Y_L).
    ``residual_correction`` always records mean(Y_L - F_L), regardless of
    selected power. Generally the point is outcome_mean + power *
    (mean(F_U) - mean(F_L)); only power one gives prediction_mean plus
    residual_correction. Fractions are scores, not implicit win labels.
    ``estimated_variance_ratio`` compares estimated variance to the audit-only
    mean's estimated variance and is None if that denominator is zero.
    ``power_bounds`` records the prespecified feasible coefficient interval;
    its default is (0,1), while (-1,1) explicitly permits inverse proxy signals.
    """

    point: float
    prediction_mean: float
    outcome_mean: float
    residual_correction: float
    standard_error: float
    interval: Interval
    n_labeled: int
    n_unlabeled: int
    n_labeled_groups: int | None
    n_unlabeled_groups: int | None
    estimand: str
    variance_method: str
    method: str
    assumptions: tuple[str, ...]
    references: tuple[str, ...]
    selected_power: float = 1.0
    power_method: str = "fixed"
    estimated_variance_ratio: float | None = None
    power_bounds: tuple[float, float] = (0.0, 1.0)

    def as_dict(self) -> dict:
        """Serialize the result, including the nested Interval, to a dict."""
        return asdict(self)


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
    if np.ma.isMaskedArray(labels) and np.any(np.ma.getmaskarray(labels)):
        raise ValueError(f"{name} must not contain masked or missing labels")
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


def _bounded_scores(values: Sequence[float], name: str) -> np.ndarray:
    """Require actual real-valued scores, without coercing strings or nulls."""
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{name} must not contain masked or missing scores")
    # Large simulation pools already have real NumPy dtypes. Validate those
    # in array operations; object conversion would box and inspect every row.
    # Restrict this path to existing numeric arrays so mixed Python sequences
    # cannot coerce booleans, strings or missing objects into valid floats.
    if isinstance(values, np.ndarray) and values.dtype.kind in "iuf":
        arr = np.asarray(values)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional numeric score sequence")
        if arr.size < 2:
            raise ValueError(f"{name} must contain at least two observations")
        if not np.all(np.isfinite(arr)) or np.any(arr < 0) or np.any(arr > 1):
            raise ValueError(f"{name} requires finite real numeric scores in [0, 1]")
        return np.asarray(arr, dtype=float)
    try:
        arr = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric score sequence") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional numeric score sequence")
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two observations")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not 0 <= value <= 1
        or not np.isfinite(float(value))
        for value in arr
    ):
        raise ValueError(f"{name} requires finite real numeric scores in [0, 1]; booleans are not scores")
    return np.asarray(arr, dtype=float)


def _validate_power_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    """Normalize a prespecified bounded interval that includes audit-only zero."""
    message = "power_bounds must be a two-real-number tuple with -1 <= lower <= 0 <= upper <= 1 and lower < upper"
    if (not isinstance(bounds, tuple) or len(bounds) != 2
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                   for value in bounds)):
        raise ValueError(message)
    lower, upper = bounds
    # Check boundedness before float conversion, including arbitrarily large ints.
    if not (-1 <= lower <= 0 <= upper <= 1 and lower < upper):
        raise ValueError(message)
    result = (float(lower), float(upper))
    if not all(np.isfinite(result)) or not result[0] < result[1]:
        raise ValueError(message + "; bounds must remain distinct at floating-point precision")
    return result


def _inference_options(
    alpha: float, power: float | str, *, power_bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float | str]:
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, Real)
        or not 0 < alpha < 1
        or not np.isfinite(float(alpha))
    ):
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    if not (isinstance(power, str) and power == "auto"):
        if (
            isinstance(power, (bool, np.bool_))
            or not isinstance(power, Real)
            or not power_bounds[0] <= power <= power_bounds[1]
            or not np.isfinite(float(power))
        ):
            if power_bounds == (0.0, 1.0):
                raise ValueError("power must be a finite number in [0, 1] or 'auto'")
            raise ValueError("power must be finite and within power_bounds, or 'auto'")
        power = float(power)
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must remain strictly between 0 and 1 at floating-point precision")
    return alpha, power


def _group_codes(
    groups: Sequence, n: int, name: str
) -> tuple[np.ndarray, set]:
    if np.ma.isMaskedArray(groups) and np.any(np.ma.getmaskarray(groups)):
        raise ValueError(f"{name} must not contain masked or missing group IDs")
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
    # A decimal constant need not equal its rounded computed mean. Preserve
    # exact zero empirical variation instead of creating a centering artifact.
    if np.all(values == values[0]):
        return 0.0
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
    if np.all(first == first[0]) or np.all(second == second[0]):
        return 0.0
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


def prediction_powered_mean(
    predictions_labeled: Sequence[float],
    outcomes_labeled: Sequence[float],
    predictions_unlabeled: Sequence[float],
    alpha: float = 0.05,
    labeled_groups: Sequence | None = None,
    unlabeled_groups: Sequence | None = None,
    *,
    power: float | str = 1.0,
    power_bounds: tuple[float, float] = (0.0, 1.0),
) -> PPIMeanResult:
    """Estimate a population mean using audited outcomes and frozen predictions.

    The aligned audit predictions F_L and outcomes Y_L, and separate
    prediction-only scores F_U, must be real finite one-dimensional numeric
    sequences in [0, 1], with at least two observations per pool. Strings,
    complex values, booleans and missing values are rejected. Fractional
    outcomes are allowed and retain their literal score meaning. For example,
    a per-comparison average annotation score defines a different outcome
    than a plurality winner. This API does not infer latent truth, convert
    fractional scores into labels, or weight rows by annotation counts.

    For fixed power lambda within ``power_bounds``, the estimator and variance are

        mean(Y_L) + lambda * (mean(F_U) - mean(F_L)),
        Q(lambda) = var_mean(Y_L - lambda * F_L)
                    + lambda**2 * var_mean(F_U).

    Default power one gives original PPI; zero gives the audit-only mean.
    ``power='auto'`` uses the bounded PPI++ mean plug-in minimizer
    clip(cov_mean(Y_L, F_L)/(var_mean(F_L)+var_mean(F_U)), lower, upper), falling back
    to zero for a zero denominator. It minimizes estimated variance, not
    realized MSE; it is not generally finite-sample unbiased. This per-pool
    variance implementation differs from ppi_py's pooled variance convention.

    ``power_bounds`` must be a two-real-number tuple satisfying
    -1 <= lower <= 0 <= upper <= 1 and lower < upper. Its default (0,1)
    preserves positive-only tuning and rejects negative fixed power. The
    explicit option (-1,1) also uses inverse proxy signals. Bounds must be
    selected before examining evaluation outcomes. For means, negative
    coefficients follow PPI++ Example 6.1; this is not a new estimator.
    Power -a on F has the same point and variance as +a on 1-F. This numeric
    complement includes all proxy values; it does not relabel outcomes.
    The bounds stabilize tuning and need not contain the unconstrained
    optimal coefficient for continuous proxies or arbitrary cluster designs.

    Both pools must be independent, disjoint, and representative of the same
    target population. All predictions must use the same frozen predictor,
    not one trained or calibrated on these audit outcomes. Scalar power
    tuning on the audit is allowed; predictor training is a different step.
    Without group IDs the caller must ensure disjointness and independent
    rows. Both group arrays, if given, must contain finite nonmissing scalar
    IDs, at least two independent groups per pool, and no overlapping IDs.
    One-way cluster covariance/variance uses centered cluster sums and the
    G/(G-1) correction. Unequal clusters retain observation weighting.

    Returns an untruncated two-sided asymptotic normal interval at 1-alpha.
    Neither the estimate nor its bounds are clipped to [0, 1]. Two rows or
    groups merely permit variance computation; valid normal approximation
    needs sufficiently many independent units, nondegenerate limiting
    variance, and no dominating cluster. Cluster power tuning is a sandwich
    adaptation, not a direct consequence of the iid PPI++ theorem. Zero
    empirical variance can yield zero width without implying certainty.
    Grouping does not repair crossed dependence, selection bias, distribution
    shift or audit leakage. No finite-sample coverage or efficiency guarantee
    is made, including for fractional outcomes.
    """
    requested_bounds = power_bounds
    power_bounds = _validate_power_bounds(requested_bounds)
    # Compare real-valued fixed powers before rounding their feasible bounds:
    # e.g. a Fraction exactly on the lower boundary must remain admissible.
    alpha, power = _inference_options(alpha, power, power_bounds=requested_bounds)
    f_l = _bounded_scores(predictions_labeled, "predictions_labeled")
    y_l = _bounded_scores(outcomes_labeled, "outcomes_labeled")
    f_u = _bounded_scores(predictions_unlabeled, "predictions_unlabeled")
    if f_l.shape != y_l.shape:
        raise ValueError("predictions_labeled and outcomes_labeled must have the same length")
    return _mean_inference(
        f_l, y_l, f_u, alpha, labeled_groups, unlabeled_groups, power,
        power_bounds=power_bounds,
        population_assumptions=(
            "Independent, disjoint audit and prediction-only pools from the same target population.",
            "Same frozen predictor for both pools; no fitting or calibration on the audit outcomes.",
            "Observation-weighted mean of the bounded outcome; fractional scores retain their defined meaning.",
        ),
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
    alpha, power = _inference_options(alpha, power)
    f_l = _scores(judge_labeled, "judge_labeled", target)
    y_l = _scores(human_labeled, "human_labeled", target)
    f_u = _scores(judge_unlabeled, "judge_unlabeled", target)
    if f_l.shape != y_l.shape:
        raise ValueError("judge_labeled and human_labeled must have the same length")
    result = _mean_inference(
        f_l, y_l, f_u, alpha, labeled_groups, unlabeled_groups, power,
        population_assumptions=(
            "Independent, disjoint audit and judge-only pools from the same target population.",
            "Same frozen judge for both pools; no fitting or calibration on the audit labels.",
            "Observation-weighted human preference; target win=1, loss=0, tie=0.5.",
        ),
    )
    fields = result.as_dict()
    # Preserve the categorical result's public field names and Interval type.
    fields["raw_rate"] = fields.pop("prediction_mean")
    fields["human_only_rate"] = fields.pop("outcome_mean")
    fields.pop("estimand")
    fields.pop("power_bounds")
    fields["target"] = target
    fields["interval"] = result.interval
    return PPIWinRateResult(**fields)


def _mean_inference(
    f_l: np.ndarray,
    y_l: np.ndarray,
    f_u: np.ndarray,
    alpha: float,
    labeled_groups: Sequence | None,
    unlabeled_groups: Sequence | None,
    power: float | str,
    *,
    population_assumptions: tuple[str, str, str],
    power_bounds: tuple[float, float] = (0.0, 1.0),
) -> PPIMeanResult:
    """Shared inference on validated scores; preserves original arithmetic."""
    auto_power = isinstance(power, str) and power == "auto"
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
            float(np.clip(covariance / denominator, *power_bounds))
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
        population_assumptions[0],
        population_assumptions[1],
        "Independent rows, or independent one-way clusters when group IDs are supplied.",
        population_assumptions[2],
        "Large-sample normal interval; no finite-sample or distribution-free guarantee.",
        (
            f"Power minimizes estimated variance over [{power_bounds[0]:g},{power_bounds[1]:g}]; no finite-sample MSE or coverage guarantee."
            if auto_power else
            f"Fixed lambda={selected_power:g}; no tuning or guaranteed efficiency improvement."
        ),
        "Point and interval are not clipped to the parameter range.",
    )
    if auto_power and labeled_codes is not None:
        assumptions += (
            "Cluster-sandwich power adaptation requires many independent clusters and no dominating cluster.",
        )
    if power_bounds != (0.0, 1.0):
        assumptions += (
            f"Prespecified coefficient bounds [{power_bounds[0]:g},{power_bounds[1]:g}]; evaluation outcomes cannot select the range or coefficient.",
            "Negative mean coefficients are equivalent to positive coefficients on the complemented numeric proxy; outcomes are unchanged.",
        )
    references = _REFERENCES
    if auto_power or selected_power != 1.0:
        references += ("https://arxiv.org/abs/2311.01453",)
    if power_bounds != (0.0, 1.0):
        references += ("https://arxiv.org/pdf/2311.01453v2",)
    return PPIMeanResult(
        point=point,
        prediction_mean=raw_rate,
        outcome_mean=float(y_l.mean()),
        residual_correction=correction,
        standard_error=standard_error,
        interval=interval,
        n_labeled=int(f_l.size),
        n_unlabeled=int(f_u.size),
        n_labeled_groups=n_labeled_groups,
        n_unlabeled_groups=n_unlabeled_groups,
        estimand="population mean of the bounded outcome",
        variance_method=variance_method,
        method=method,
        assumptions=assumptions,
        references=references,
        selected_power=selected_power,
        power_method=power_method,
        estimated_variance_ratio=variance_ratio,
        power_bounds=power_bounds,
    )
