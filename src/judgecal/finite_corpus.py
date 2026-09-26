"""Fixed-corpus difference estimation under SRS of complete, unequal groups.

The concentration bounds are Bardenet and Maillard (2015), Corollary 2.5
and Theorem 4.3, https://arxiv.org/pdf/1309.4029v2. This applies established
sampling and concentration results, not a new PPI theorem. All corpus
values are fixed; only the without-replacement group sample is random.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import fsum
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from .finite_sample import _power_candidates
from .intervals import Interval
from .ppi import _inference_options


_METHODS = ("hoeffding_serfling", "empirical_bernstein_serfling")
_REFERENCES = ("https://arxiv.org/pdf/1309.4029v2", "https://doi.org/10.3150/14-BEJ605")
_KAPPA = 7/3 + 3/np.sqrt(2.)


@dataclass(frozen=True)
class FiniteCorpusMeanResult:
    """Finite-population interval and its sampling/selection contract.

    Candidate tuples align in ascending power order. ``candidate_ranges``
    contain known group-total bounds, not observed residual extrema.
    ``candidate_sample_variances`` are the ddof=0 empirical variances of
    sampled residual totals; they are not design variances of selected
    estimators. ``design_unbiased`` concerns the point under the declared
    randomization, separately from simultaneous finite-interval coverage.
    """

    point: float
    interval: Interval
    radius: float
    alpha: float
    method: str
    selected_power: float
    power_method: str
    candidate_powers: tuple[float, ...]
    candidate_points: tuple[float, ...]
    candidate_radii: tuple[float, ...]
    candidate_ranges: tuple[tuple[float, float], ...]
    candidate_sample_variances: tuple[float, ...]
    n_groups: int
    n_rows: int
    n_audited_groups: int
    audited_rows: int
    expected_audited_rows: float
    max_group_size: int
    rho: float
    finite_population_correction: float
    log_term: float
    error_allocation: str
    census: bool
    selection_uses_audit_outcomes: bool
    design_unbiased: bool
    estimand: str
    finite_coverage_scope: str
    assumptions: tuple[str, ...]
    references: tuple[str, ...]

    @property
    def point_estimate(self) -> float:
        """Explicit alias for the conventional ``point`` field."""
        return self.point

    def as_dict(self) -> dict:
        return asdict(self)


def _unmasked_1d(values: Sequence, name: str) -> np.ndarray:
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{name} must not contain masked or missing values")
    try:
        # Preserve Python sequence element types so [True,1] cannot silently
        # become integer counts, and ['0',1] cannot become numeric totals.
        array = np.asarray(values) if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    return array


def _integers(values: Sequence, name: str, minimum: int) -> tuple[int, ...]:
    array = _unmasked_1d(values, name)
    if array.dtype.kind in "iu":
        if np.any(array < minimum):
            raise ValueError(f"{name} requires integers at least {minimum}")
        return tuple(array.tolist())
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           or value < minimum for value in array):
        raise ValueError(f"{name} requires strict integers at least {minimum}")
    return tuple(int(value) for value in array)


def _totals(values: Sequence, limits: tuple[int, ...], name: str) -> np.ndarray:
    array = _unmasked_1d(values, name)
    if array.size != len(limits):
        raise ValueError(f"{name} must have length {len(limits)}")
    if array.dtype.kind in "iuf" and max(limits) < 2**53:
        if not np.all(np.isfinite(array)) or np.any(array < 0) or np.any(array > np.asarray(limits)):
            raise ValueError(f"{name} must contain finite real totals within corresponding group sizes")
    else:
        for value, limit in zip(array, limits):
            # Python's mixed int/float comparison preserves integer bounds
            # beyond 2**53; NumPy promotion can round an invalid total down.
            value = value.item() if isinstance(value, np.generic) else value
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not 0 <= value <= limit):
                raise ValueError(f"{name} must contain finite real totals within corresponding group sizes")
    try:
        result = np.asarray(array, dtype=float)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be representable as finite floating-point totals") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be representable as finite floating-point totals")
    return result


def _sample_variance(residual: np.ndarray) -> float:
    """Translation-stable ddof=0 variance, preserving exact constants."""
    if np.all(residual == residual[0]):
        return 0.
    # Removing the first observation makes variance invariant to exact common
    # shifts (e.g. equal-size groups and a constant row proxy). No tolerance
    # threshold erases real, small variation.
    with np.errstate(over="ignore", invalid="ignore"):
        variance = float(np.var(residual-residual[0], ddof=0))
    if not np.isfinite(variance):
        raise ValueError("sampled residual-total variance exceeds floating-point range")
    return variance


def finite_corpus_mean(
    group_sizes: Sequence[int],
    group_prediction_totals: Sequence[float],
    audited_group_indices: Sequence[int],
    audited_outcome_totals: Sequence[float],
    alpha: float = .05,
    *,
    power: float | tuple[float, ...] = 1.,
    method: str = "hoeffding_serfling",
) -> FiniteCorpusMeanResult:
    """Infer the full fixed-corpus row mean from a uniform group sample.

    The complete frame contains G>=2 disjoint nonempty groups with known
    positive integer sizes and proxy totals. The M=sum(group_sizes) row
    outcomes/proxies are bounded in [0,1]; totals therefore lie in [0,n_g].
    Sample exactly k>=1 whole groups uniformly without replacement and
    supply their distinct zero-based indices and complete outcome totals.
    Arrays cannot certify this randomization. All values are fixed: no IID
    outcome assumption or independent prediction pool is required.

    For fixed lambda, R_g=Y_g-lambda F_g gives the design-unbiased estimate
    lambda*sum_all(F_g)/M + G*sum_sample(R_g)/(k*M). This is not a sampled-row
    ratio, a future population mean, or a random held-out target. Points and
    intervals are untruncated. The random row-label cost is sum_sample(n_g).

    A scalar power in [0,1] must be fixed before observing the random audit
    (it may depend on the known frame). A nonempty distinct tuple declares
    a prespecified finite family. Candidates are sorted and the smallest
    radius selected, breaking exact ties by smaller power. Every candidate
    is protected simultaneously; the grid and bound family cannot be chosen
    on these outcomes without an additional error allocation. EBS selection
    can depend on outcomes, so its selected point need not be unbiased.

    Known residual bounds are a=min_g(-lambda F_g),
    b=max_g(n_g-lambda F_g), W=b-a. Let
    rho=1-(k-1)/G for k<=G/2, else (1-k/G)*(1+1/k).
    For K candidates, Hoeffding-Serfling's two-sided radius is
    (G/M)*W*sqrt(rho*log(2K/alpha)/(2k)). Empirical Bernstein-Serfling uses
    (G/M)*(sqrt(2*rho*vhat*log(10K/alpha)/k)
             +(7/3+3/sqrt(2))*W*log(10K/alpha)/k),
    where vhat is the sampled residual variance with divisor k. The latter
    allocation accounts for the theorem's one-sided failure probability
    5*delta. For k=1, vhat=0 and the known-range term remains. At census,
    every candidate returns the direct outcome total/M and exactly zero
    radius, overriding the EBS additive term.

    Fixed size k, complete responses, uniform sampling, fixed alpha and
    prespecified family are required. PPS/unequal probabilities, partial
    groups, unknown frames, adaptive stopping, and same-corpus pilot reuse
    are unsupported. This implements established concentration results and
    exposes no normal approximation or selected-power variance guarantee.
    """
    if not isinstance(method, str) or method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    alpha, _ = _inference_options(alpha, 0.)
    powers, is_grid = _power_candidates(alpha, power)
    sizes = _integers(group_sizes, "group_sizes", 1)
    count_g = len(sizes)
    if count_g < 2:
        raise ValueError("group_sizes must define at least two groups")
    count_m = sum(sizes)
    try:
        sizes_float = np.asarray(sizes, dtype=float)
        rows_float = float(count_m)
    except (OverflowError, ValueError) as exc:
        raise ValueError("group sizes and total rows must be representable as finite floats") from exc
    if not np.isfinite(rows_float) or not np.all(np.isfinite(sizes_float)):
        raise ValueError("group sizes and total rows must be representable as finite floats")
    f_all = _totals(group_prediction_totals, sizes, "group_prediction_totals")
    indices = _integers(audited_group_indices, "audited_group_indices", 0)
    count_k = len(indices)
    if not count_k or count_k > count_g:
        raise ValueError("audit must contain between one and G distinct groups")
    if len(set(indices)) != count_k or any(index >= count_g for index in indices):
        raise ValueError("audited_group_indices must be unique valid zero-based group indices")
    index_array = np.asarray(indices, dtype=np.intp)
    y_sample = _totals(audited_outcome_totals, tuple(sizes[index] for index in indices), "audited_outcome_totals")
    f_sample = f_all[index_array]
    census = count_k == count_g
    fpc = (count_g-count_k)/count_g
    rho = ((count_g-count_k+1)/count_g if 2*count_k <= count_g
           else fpc*(1+1/count_k))
    constant = 2. if method == "hoeffding_serfling" else 10.
    log_term = float(np.log(constant)+np.log(len(powers))-np.log(alpha))
    scale = count_g/rows_float
    # Equivalent difference form; separating the sample outcome mean from
    # the full-minus-sampled proxy mean preserves equal-size constant-proxy
    # cancellation without clipping any point or coefficient.
    try:
        y_mean = fsum(y_sample)/count_k
        f_full_mean = fsum(f_all)/count_g
        f_sample_mean = fsum(f_sample)/count_k
        census_point = fsum(y_sample)/rows_float if census else None
    except OverflowError as exc:
        raise ValueError("group totals exceed floating-point summation range") from exc
    points, radii, ranges, variances = [], [], [], []
    for value in powers:
        low = float(np.min(-value*f_all))
        high = float(np.max(sizes_float-value*f_all))
        residual = y_sample-value*f_sample
        variance = _sample_variance(residual)
        point = (census_point if census else
                 scale*y_mean + value*scale*(f_full_mean-f_sample_mean))
        # Scaling each endpoint first avoids overflowing a raw range at
        # extreme numeric inputs even though its target-scale bound is small.
        scaled_range = scale*high-scale*low
        if census:
            radius = 0.
        elif method == "hoeffding_serfling":
            radius = scaled_range*np.sqrt(rho*log_term/(2*count_k))
        else:
            radius = (scale*np.sqrt(variance)*np.sqrt(2*rho*log_term/count_k)
                      + _KAPPA*scaled_range*log_term/count_k)
        if not np.isfinite(point) or not np.isfinite(radius):
            raise ValueError("point and bound must remain finite at floating-point precision")
        points.append(float(point))
        radii.append(float(radius))
        ranges.append((low, high))
        variances.append(variance)
    chosen = int(np.argmin(radii))
    point, radius = points[chosen], radii[chosen]
    # k=1 has zero empirical variance for every power, so even EBS selection
    # then depends only on the known frame, not the observed outcome.
    uses_outcomes = bool(method == "empirical_bernstein_serfling" and len(powers) > 1
                         and count_k > 1 and not census)
    return FiniteCorpusMeanResult(
        point=point, interval=Interval(point, point-radius, point+radius, alpha,
                                      f"finite-corpus-{method.replace('_', '-')}-group-srs"),
        radius=radius, alpha=alpha, method=method, selected_power=powers[chosen],
        power_method="finite-grid-minimum-radius" if is_grid else "fixed",
        candidate_powers=tuple(powers), candidate_points=tuple(points),
        candidate_radii=tuple(radii), candidate_ranges=tuple(ranges),
        candidate_sample_variances=tuple(variances), n_groups=count_g, n_rows=count_m,
        n_audited_groups=count_k, audited_rows=sum(sizes[index] for index in indices),
        expected_audited_rows=(count_k/count_g)*count_m, max_group_size=max(sizes),
        rho=float(rho), finite_population_correction=float(fpc), log_term=log_term,
        error_allocation=("alpha/(2*K) per one-sided event" if method == "hoeffding_serfling" else
                          "delta=alpha/(10*K); each one-sided theorem has failure at most 5*delta"),
        census=census, selection_uses_audit_outcomes=uses_outcomes,
        design_unbiased=not uses_outcomes,
        estimand="full fixed-corpus row-weighted mean of bounded outcomes",
        finite_coverage_scope="Design probability over uniform without-replacement samples of k complete groups; simultaneous over prespecified power candidates",
        assumptions=(
            "The complete fixed corpus is partitioned into known nonempty disjoint groups; all row scores lie in [0,1].",
            "All group sizes and proxy totals are known before sampling; only audited outcome totals are supplied.",
            "Exactly k groups are sampled uniformly without replacement; all outcomes in each sampled group are observed.",
            "Sample size, alpha, bound family and scalar coefficient or candidate grid are prespecified independently of audit outcomes.",
            "Arbitrary fixed outcome dependence is allowed; no future-population or held-out-mean guarantee follows.",
            "Separate bound/proxy families do not have joint selection protection unless error is allocated across them.",
            "Sample-selected EBS grid points need not be design unbiased; fixed or known-frame-only selections are design unbiased.",
        ),
        references=_REFERENCES,
    )
