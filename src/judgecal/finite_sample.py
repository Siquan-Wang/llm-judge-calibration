"""Finite-sample PPI mean intervals from established bounded-mean inequalities.

Hoeffding (1963), Theorem 2: https://doi.org/10.1080/01621459.1963.10500830
Maurer and Pontil (2009), Theorem 4: https://arxiv.org/abs/0907.3740
PPI composition, Algorithm 10: https://arxiv.org/abs/2301.09633

These conservative iid baselines preserve within-pair outcome/prediction
dependence. Finite-grid selection uses simultaneous bounds, not the continuous
plug-in coefficient of the asymptotic PPI++ API. This is an application of
existing concentration inequalities, not a new inference theorem.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .intervals import Interval
from .ppi import _bounded_scores, _inference_options


_REFERENCES = (
    "https://doi.org/10.1080/01621459.1963.10500830",
    "https://arxiv.org/pdf/0907.3740v1",
    "https://arxiv.org/abs/2301.09633",
)


@dataclass(frozen=True)
class FiniteSampleMeanResult:
    """Untruncated finite-sample interval with explicit selection/allocation.

    ``candidate_powers``, ``candidate_points`` and ``candidate_radii`` align
    in ascending power order. ``residual_radius`` and ``prediction_radius``
    are the selected EB components; the prediction component already includes
    the selected power. They are None for direct weighted Hoeffding, which
    bounds the whole independent sum rather than two separate events.

    Log terms record the exact bound constants without requiring extremely
    small allocated error probabilities to be representable as floats.
    ``residual_correction`` always means mean(Y_L-F_L), independently of power.
    A finite-sample guarantee is conditional on the stated sampling assumptions;
    arrays alone cannot verify independence or representativeness.
    """

    point: float
    interval: Interval
    radius: float
    alpha: float
    prediction_mean: float
    outcome_mean: float
    residual_correction: float
    n_labeled: int
    n_unlabeled: int
    selected_power: float
    power_method: str
    method: str
    candidate_powers: tuple[float, ...]
    candidate_points: tuple[float, ...]
    candidate_radii: tuple[float, ...]
    residual_radius: float | None
    prediction_radius: float | None
    hoeffding_log_term: float | None
    residual_log_term: float | None
    prediction_log_term: float | None
    error_allocation: str
    estimand: str
    assumptions: tuple[str, ...]
    references: tuple[str, ...]

    def as_dict(self) -> dict:
        """Serialize the nested interval and candidate metadata."""
        return asdict(self)


def _power_candidates(alpha: float, power: float | tuple[float, ...]):
    """Distinguish one fixed power from an ex ante finite candidate tuple."""
    is_grid = isinstance(power, tuple)
    values = power if is_grid else (power,)
    if not values:
        raise ValueError("power grid must be a nonempty tuple of numbers in [0, 1]")
    parsed = []
    for value in values:
        if isinstance(value, str):
            raise ValueError("power must be a number or a prespecified tuple; continuous 'auto' is unsupported")
        _, numeric = _inference_options(alpha, value)
        parsed.append(numeric)
    if len(set(parsed)) != len(parsed):
        raise ValueError("power grid must not contain duplicate powers")
    return tuple(sorted(parsed)), is_grid


def _eb_radius(values: np.ndarray, length: float, log_term: float) -> float:
    """MP Theorem 4 applied to both tails, with the log already allocated."""
    count = values.size
    return float(
        np.sqrt(2 * np.var(values, ddof=1) * log_term / count)
        + 7 * length * log_term / (3 * (count - 1))
    )


def finite_sample_mean(
    predictions_labeled: Sequence[float],
    outcomes_labeled: Sequence[float],
    predictions_unlabeled: Sequence[float],
    alpha: float = 0.05,
    *,
    method: str = "hoeffding",
    power: float | tuple[float, ...] = 1.0,
) -> FiniteSampleMeanResult:
    """Bound a population mean with fixed power or simultaneous grid selection.

    Inputs are aligned audit scores F_L,Y_L and a separate prediction pool F_U,
    each finite, real, one-dimensional, in [0,1], with at least two rows per
    pool. Fractional outcomes retain their literal meaning. Both pools must
    be iid and independent of each other, with the same predictor marginal;
    audit outcomes must represent the target population. The common predictor
    must be frozen independently of the inference data. Pairing of F_L,Y_L
    is intentional; their independence is not required. Sample sizes and the
    error level must be fixed without optional stopping.

    A scalar ``power`` in [0,1] must be fixed in advance, or chosen on an
    independent pilot. A nonempty tuple declares a prespecified finite grid.
    Every grid interval covers simultaneously with probability >=1-alpha;
    the function selects the smallest untruncated radius, breaking ties by
    smaller power. Choosing the grid itself from these samples is unsupported,
    as is passing a data-tuned scalar (including the continuous 'auto' power
    of prediction_powered_mean). Duplicate grid values are rejected.

    For each lambda the point is mean(Y_L)+lambda*(mean(F_U)-mean(F_L)).
    ``method='hoeffding'`` bounds the independent weighted sum directly:

        r = sqrt(log(2*K/alpha)/2 * ((1+lambda)**2/n + lambda**2/N)),

    where K is one for fixed power. This range-only radius increases with
    power; grid selection therefore chooses the smallest candidate power.

    ``method='empirical_bernstein'`` applies Maurer-Pontil (2009), Theorem 4,
    separately to residuals Y_L-lambda F_L and predictions F_U. For a sample
    z of size m, known range length L, and two-sided failure budget delta:

        B(z,delta,L) = sqrt(2*var(z,ddof=1)*log(4/delta)/m)
                       + 7*L*log(4/delta)/(3*(m-1)).

    Each residual receives delta=alpha/(2*K) and uses range length 1+lambda.
    One shared prediction bound receives alpha/2 and is multiplied by lambda.
    Thus the logs are log(8*K/alpha) and log(8/alpha), respectively. Fixed zero
    (including a singleton zero tuple) instead uses B(Y_L,alpha,1). Zero in a
    larger grid retains its simultaneous-selection penalty. Observed zero
    residual variance never permits reducing the known residual range.

    Returned points and intervals are untruncated. Intersecting with [0,1]
    preserves coverage but is not performed here. No normal standard error
    or finite-sample efficiency claim is supplied. Grouped/dependent rows,
    arbitrary weights, population shift, data-trained predictors, optional
    stopping and adaptive selection across bound methods require other
    arguments; this iid API does not support them. In particular, repeated
    rows cannot be treated as extra independent information.
    """
    if not isinstance(method, str) or method not in {"hoeffding", "empirical_bernstein"}:
        raise ValueError("method must be 'hoeffding' or 'empirical_bernstein'")
    alpha, _ = _inference_options(alpha, 0.0)
    powers, is_grid = _power_candidates(alpha, power)
    f_l = _bounded_scores(predictions_labeled, "predictions_labeled")
    y_l = _bounded_scores(outcomes_labeled, "outcomes_labeled")
    f_u = _bounded_scores(predictions_unlabeled, "predictions_unlabeled")
    if f_l.size != y_l.size:
        raise ValueError("predictions_labeled and outcomes_labeled must have the same length")

    n, N, K = len(y_l), len(f_u), len(powers)
    prediction_mean = float(f_u.mean())
    outcome_mean = float(y_l.mean())
    audit_prediction_mean = float(f_l.mean())
    correction = float((y_l - f_l).mean())
    points = tuple(
        prediction_mean + correction if value == 1.0
        else float(outcome_mean + value * (prediction_mean - audit_prediction_mean))
        for value in powers
    )
    log_alpha = float(np.log(alpha))
    h_log = r_log = u_log = None
    residual_radii = prediction_radii = None
    if method == "hoeffding":
        h_log = float(np.log(2.0) + np.log(K) - log_alpha)
        radii = tuple(
            float(np.sqrt(0.5 * h_log * ((1 + value)**2 / n + value**2 / N)))
            for value in powers
        )
        allocation = "alpha/K per candidate; two tails of the independent weighted sum"
    elif powers == (0.0,):
        r_log = float(np.log(4.0) - log_alpha)
        residual_radii = (_eb_radius(y_l, 1.0, r_log),)
        prediction_radii = (0.0,)
        radii = residual_radii
        allocation = "alpha to the audit-only two-sided bound; prediction bound unused"
    else:
        r_log = float(np.log(8.0) + np.log(K) - log_alpha)
        u_log = float(np.log(8.0) - log_alpha)
        prediction_bound = _eb_radius(f_u, 1.0, u_log)
        residual_radii = tuple(
            _eb_radius(y_l - value * f_l, 1 + value, r_log) for value in powers
        )
        prediction_radii = tuple(value * prediction_bound for value in powers)
        radii = tuple(r + u for r, u in zip(residual_radii, prediction_radii))
        allocation = "alpha/(2*K) to each residual; alpha/2 to one shared prediction bound; each two-sided"

    selected = int(np.argmin(radii))
    point, radius = points[selected], radii[selected]
    interval_method = f"ppi-mean-{method.replace('_', '-')}-finite-sample-iid"
    return FiniteSampleMeanResult(
        point=point,
        interval=Interval(point, point - radius, point + radius, alpha, interval_method),
        radius=radius, alpha=alpha,
        prediction_mean=prediction_mean, outcome_mean=outcome_mean,
        residual_correction=correction,
        n_labeled=n, n_unlabeled=N,
        selected_power=powers[selected],
        power_method="finite-grid-minimum-radius" if is_grid else "fixed",
        method=method,
        candidate_powers=powers, candidate_points=points, candidate_radii=radii,
        residual_radius=None if residual_radii is None else residual_radii[selected],
        prediction_radius=None if prediction_radii is None else prediction_radii[selected],
        hoeffding_log_term=h_log, residual_log_term=r_log, prediction_log_term=u_log,
        error_allocation=allocation,
        estimand="population mean of the bounded outcome",
        assumptions=(
            "Scores lie in [0,1]; iid audit pairs represent the outcome population.",
            "The iid prediction pool is independent of the audit and has the same predictor marginal.",
            "A common predictor is frozen independently of both inference pools.",
            "Sample sizes, alpha, bound method, and scalar power or finite candidate grid are fixed independently of these data.",
            "Within-pair outcome/prediction dependence is allowed; between-row dependence and arbitrary cluster weighting are unsupported.",
            "Data-dependent power selection is covered only among the prespecified simultaneous grid candidates.",
        ),
        references=_REFERENCES,
    )
