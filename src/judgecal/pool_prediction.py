"""Prediction of a random held-out mean, separate from population inference.

For independent iid pools and fixed power lambda, the error relative to the
held-out outcome mean is Rbar_L - Rbar_U, where R = Y - lambda*F. Its marginal
variance is (1/n + 1/N)*Var(R), whereas population-mean PPI has a different
variance and oracle coefficient. This is a target-specific specialization of
the established mean correction, not a new PPI estimator or finite-sample bound.

The public point-only audit-residual tuner also accepts instruction groups.
It deliberately supplies no grouped prediction interval: unequal-cluster ratio
means and fixed-corpus assignment require a separate sampling-design analysis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy import special

from .intervals import Interval
from .ppi import (
    _bounded_scores, _covariance_of_means, _group_codes, _inference_options,
    _validate_power_bounds, _variance_of_mean,
)


_REFERENCES = (
    "https://arxiv.org/pdf/2311.01453v2",
    "https://arxiv.org/abs/2301.09633v4",
)
_CRITERION = "audit-residual-mean-variance"


@dataclass(frozen=True)
class PoolMeanPredictionResult:
    """Marginal asymptotic prediction interval for a random held-out mean.

    ``audit_residual_variance`` is the audit sample residual variance (ddof=1).
    Prediction error includes the unobserved held-out residual mean's sampling
    variation. This result has no population ``standard_error`` or ``interval``
    alias, to keep the target/scope visible at use sites.
    """

    point: float
    prediction_standard_error: float
    prediction_interval: Interval
    audit_residual_variance: float
    selected_power: float
    power_method: str
    power_bounds: tuple[float, float]
    n_labeled: int
    n_unlabeled: int
    criterion: str
    estimand: str
    coverage_scope: str
    method: str
    assumptions: tuple[str, ...]
    references: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AuditResidualMeanResult:
    """Point-only residual-tuned candidate; intentionally no uncertainty output.

    ``criterion_value`` is the minimized estimated variance of the audit
    residual mean, using the declared iid/cluster convention. It is not an
    estimate of population estimation error or held-out prediction error.
    All means retain observation weighting, including unequal-size clusters.
    """

    point: float
    selected_power: float
    power_method: str
    power_bounds: tuple[float, float]
    n_labeled: int
    n_unlabeled: int
    n_labeled_groups: int | None
    n_unlabeled_groups: int | None
    criterion: str
    criterion_value: float
    variance_method: str
    estimand: str
    method: str
    assumptions: tuple[str, ...]
    references: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def _validated_scores(predictions_labeled, outcomes_labeled, predictions_unlabeled):
    f = _bounded_scores(predictions_labeled, "predictions_labeled")
    y = _bounded_scores(outcomes_labeled, "outcomes_labeled")
    u = _bounded_scores(predictions_unlabeled, "predictions_unlabeled")
    if f.shape != y.shape:
        raise ValueError("predictions_labeled and outcomes_labeled must have the same length")
    return f, y, u


def _audit_power(f, y, groups, bounds):
    # A representably constant proxy has no audit signal, even when summing
    # its repeated decimal values perturbs the computed mean by one ulp.
    if np.all(f == f[0]):
        return 0.0
    denominator = _variance_of_mean(f, groups)
    if denominator <= 0:
        return 0.0
    covariance = _covariance_of_means(y, f, groups)
    return float(np.clip(covariance / denominator, *bounds))


def _point(f, y, u, power):
    # Match the existing mean-correction arithmetic for an identical coefficient.
    return (float(u.mean()) + float((y-f).mean()) if power == 1.0
            else float(y.mean() + power*(u.mean()-f.mean())))


def predict_heldout_mean(
    predictions_labeled: Sequence[float],
    outcomes_labeled: Sequence[float],
    predictions_unlabeled: Sequence[float],
    alpha: float = 0.05,
    *,
    power: float | str = 1.0,
    power_bounds: tuple[float, float] = (-1.0, 1.0),
) -> PoolMeanPredictionResult:
    """Predict a random held-out outcome mean under independent iid sampling.

    Every input is a one-dimensional finite real score in [0,1] with at least
    two observations; audit predictions/outcomes are aligned. The same frozen
    proxy rule applies to both pools. No held-out outcomes are accepted.

    The point is mean(Y_L)+lambda*(mean(F_U)-mean(F_L)). For fixed lambda,
    its error relative to random mean(Y_U) is mean(R_L)-mean(R_U), where
    R=Y-lambda*F. Estimate its variance by (1/n+1/N)*sample_var(R_L,ddof=1).
    This includes the covariance between the point and its random target;
    simply adding Var(Y_U) to a population PPI variance is incorrect.

    Default power=1 uses the fixed correction. ``power='auto'`` minimizes
    audit residual variance: clip(sample_cov(Y_L,F_L)/sample_var(F_L), bounds).
    A zero audit proxy variance selects zero. Unlike population tuning, this
    ratio has no unlabeled proxy variance term. Bounds must be a prespecified
    tuple satisfying -1<=lower<=0<=upper<=1 and lower<upper. Same-audit tuning
    can bias small-sample estimates and variances; no dominance is guaranteed.

    The untruncated normal interval predicts the random held-out mean, with
    marginal coverage over repeated draws of BOTH independent iid pools from
    the same joint law. It is not a confidence interval for E[Y], nor a
    conditional guarantee for a fixed U, its observed proxies, or an audit.
    Both pool sizes must grow with a controlled ratio, the coefficient must
    stabilize, and the limiting residual variance must be nondegenerate for
    the stated asymptotic approximation. Zero empirical variance alone does
    not establish certainty. Distribution shift and fitted-on-audit proxies
    invalidate the assumed transfer of residual moments.

    Group IDs are intentionally unsupported. A coincident finite-population
    variance identity for fixed-coefficient random disjoint row partitions
    does not establish a normal guarantee for arbitrary clustered corpora or
    an audit-adaptive residual function. Use ``audit_residual_mean`` only as
    a point candidate in such grouped exploratory analyses.
    """
    bounds = _validate_power_bounds(power_bounds)
    alpha, power = _inference_options(alpha, power, power_bounds=power_bounds)
    f, y, u = _validated_scores(predictions_labeled, outcomes_labeled, predictions_unlabeled)
    automatic = isinstance(power, str) and power == "auto"
    selected = _audit_power(f, y, None, bounds) if automatic else power
    point = _point(f, y, u, selected)
    residual = y-selected*f
    residual_variance = (0.0 if np.all(residual == residual[0])
                         else float(np.var(residual, ddof=1)))
    prediction_se = float(np.sqrt((1.0/len(y)+1.0/len(u))*residual_variance))
    z = -float(special.ndtri_exp(np.log(alpha)-np.log(2.0)))
    method = "iid-random-heldout-mean-normal-prediction"
    interval = Interval(point, point-z*prediction_se, point+z*prediction_se, alpha, method)
    return PoolMeanPredictionResult(
        point=point, prediction_standard_error=prediction_se, prediction_interval=interval,
        audit_residual_variance=residual_variance, selected_power=selected,
        power_method="auto-audit-residual-sample-variance" if automatic else "fixed",
        power_bounds=bounds, n_labeled=len(y), n_unlabeled=len(u),
        criterion=_CRITERION if automatic else "fixed-coefficient",
        estimand="random held-out outcome mean",
        coverage_scope="marginal over independent iid audit and held-out pool draws",
        method=method,
        assumptions=(
            "Independent iid audit and held-out pools from the same joint outcome/proxy distribution.",
            "Same frozen proxy rule in both pools; no fitting or calibration on audit outcomes.",
            "Random held-out mean target; no population-mean confidence interval or conditional fixed-pool guarantee.",
            "Both pool sizes grow with controlled ratio, stable coefficient and nondegenerate residual variance.",
            "Same-audit tuning can introduce finite-sample bias and undercoverage; zero sample variance is not certainty.",
            "No distribution-free guarantee; neither points nor prediction endpoints are clipped.",
            "Prespecified power bounds; hidden outcomes cannot choose a range or coefficient.",
        ), references=_REFERENCES,
    )


def audit_residual_mean(
    predictions_labeled: Sequence[float],
    outcomes_labeled: Sequence[float],
    predictions_unlabeled: Sequence[float],
    labeled_groups: Sequence | None = None,
    unlabeled_groups: Sequence | None = None,
    *,
    power_bounds: tuple[float, float] = (-1.0, 1.0),
) -> AuditResidualMeanResult:
    """Return an audit-residual tuned point candidate, with no interval or SE.

    Choose lambda by minimizing the audit residual mean's estimated variance
    within prespecified bounds; covariance/variance use audit data only. Both
    pools' proxy means enter the observation-weighted corrected point. A zero
    audit proxy variance selects zero even if the held-out proxy varies.

    Inputs obey the same strict bounded-score and bound validation as
    ``predict_heldout_mean``. Optional group IDs must be supplied for both
    pools, have aligned lengths, contain at least two finite nonmissing scalar
    groups per pool, and be disjoint across pools. With groups, use centered
    cluster totals and G/(G-1)/n² for the audit covariance and variance.
    Unequal groups retain row weighting. Held-out group IDs are validated to
    prevent leakage; their variance and outcomes never determine lambda.

    For independent iid pools this criterion corresponds to the pool-error
    variance optimum. For grouped fixed-corpus benchmarks it is an exploratory
    audit-residual criterion, not an established conditional/design optimum.
    No grouped uncertainty estimate or coverage guarantee is supplied.
    """
    bounds = _validate_power_bounds(power_bounds)
    f, y, u = _validated_scores(predictions_labeled, outcomes_labeled, predictions_unlabeled)
    if (labeled_groups is None) != (unlabeled_groups is None):
        raise ValueError("labeled_groups and unlabeled_groups must be supplied together")
    audit_codes = None
    n_labeled_groups = n_unlabeled_groups = None
    variance_method = "iid-sample-variance"
    if labeled_groups is not None:
        audit_codes, audit_ids = _group_codes(labeled_groups, len(y), "labeled_groups")
        _, heldout_ids = _group_codes(unlabeled_groups, len(u), "unlabeled_groups")
        if audit_ids & heldout_ids:
            raise ValueError("labeled and unlabeled group IDs must be disjoint")
        n_labeled_groups, n_unlabeled_groups = len(audit_ids), len(heldout_ids)
        variance_method = "one-way-cluster-robust"
    selected = _audit_power(f, y, audit_codes, bounds)
    return AuditResidualMeanResult(
        point=_point(f, y, u, selected), selected_power=selected,
        power_method=("auto-audit-residual-cluster-sandwich" if audit_codes is not None
                      else "auto-audit-residual-sample-variance"),
        power_bounds=bounds, n_labeled=len(y), n_unlabeled=len(u),
        n_labeled_groups=n_labeled_groups, n_unlabeled_groups=n_unlabeled_groups,
        criterion=_CRITERION,
        criterion_value=_variance_of_mean(y-selected*f, audit_codes),
        variance_method=variance_method,
        estimand="held-out outcome mean point-prediction candidate",
        method="audit-residual-tuned-mean-point-only",
        assumptions=(
            "Only audit outcomes select the coefficient; held-out proxies enter the corrected point only.",
            "Same frozen proxy rule for both pools; prespecified bounds and no hidden-label tuning.",
            "Observation-weighted means; complete independent instruction groups when grouping is supplied.",
            "Audit residual variance criterion; not a claim of optimal realized error or exact conditional corpus optimality.",
            "Point estimate only: no prediction interval, standard error or coverage guarantee.",
            "Unequal-group sampling and fixed-corpus design require separate uncertainty analysis.",
        ), references=_REFERENCES,
    )
