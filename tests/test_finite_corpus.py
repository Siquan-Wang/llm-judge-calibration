"""Fixed-corpus contracts, bound constants, census and strict validation."""
from dataclasses import FrozenInstanceError
from fractions import Fraction
from math import fsum, log, sqrt
import json

import numpy as np
import pytest

from judgecal.finite_corpus import FiniteCorpusMeanResult, finite_corpus_mean


SIZES = np.array([1, 2, 4, 3])
PROXIES = np.array([.2, .4, 3.6, 1.2])
INDICES = np.array([0, 2, 3])
OUTCOMES = np.array([.1, 2.8, 2.])
METHODS = ("hoeffding_serfling", "empirical_bernstein_serfling")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("power", [0., .25, 1.])
def test_scalar_hand_calculation_and_sampling_metadata(method, power):
    result = finite_corpus_mean(SIZES, PROXIES, INDICES, OUTCOMES, power=power, method=method)
    residual = OUTCOMES-power*PROXIES[INDICES]
    variance = float(np.var(residual, ddof=0))
    lower = min(-power*PROXIES)
    upper = max(SIZES-power*PROXIES)
    rho = (1-3/4)*(1+1/3)
    ell = log((2 if method == "hoeffding_serfling" else 10)/.05)
    if method == "hoeffding_serfling":
        radius = .4*(upper-lower)*sqrt(rho*ell/6)
    else:
        radius = .4*(sqrt(2*rho*variance*ell/3)+(7/3+3/sqrt(2))*(upper-lower)*ell/3)
    point = power*sum(PROXIES)/10+4/(3*10)*sum(residual)
    assert isinstance(result, FiniteCorpusMeanResult)
    assert result.point == result.point_estimate == pytest.approx(point)
    assert result.candidate_sample_variances == pytest.approx((variance,))
    assert result.candidate_ranges[0] == pytest.approx((lower, upper))
    assert result.radius == pytest.approx(radius)
    assert result.interval.low == pytest.approx(point-radius)
    assert result.interval.high == pytest.approx(point+radius)
    assert result.rho == pytest.approx(rho)
    assert result.finite_population_correction == .25
    assert (result.n_groups, result.n_rows, result.n_audited_groups) == (4, 10, 3)
    assert result.audited_rows == 8
    assert result.expected_audited_rows == 7.5
    assert result.max_group_size == 4
    assert result.design_unbiased and not result.selection_uses_audit_outcomes
    assert not result.census
    assert result.power_method == "fixed"


@pytest.mark.parametrize("method", METHODS)
def test_census_returns_exact_direct_mean_for_all_candidates(method):
    sizes = [1, 2, 1, 3]
    proxies = [.3, 1.7, .9, 2.6]
    outcomes = [.1, .2, .3, .4]
    indices = [3, 1, 0, 2]
    aligned = [outcomes[i] for i in indices]
    result = finite_corpus_mean(sizes, proxies, indices, aligned,
                                power=(1., .25, 0., .75), method=method)
    truth = fsum(outcomes)/7
    assert result.point == result.interval.point == result.interval.low == result.interval.high == truth
    assert result.candidate_points == (truth,)*4
    assert result.candidate_radii == (0.,)*4
    assert result.radius == result.rho == result.finite_population_correction == 0.
    assert result.census and result.design_unbiased
    assert not result.selection_uses_audit_outcomes
    assert result.selected_power == 0.
    assert result.audited_rows == result.expected_audited_rows == result.n_rows


@pytest.mark.parametrize("method", METHODS)
def test_single_group_sample_has_zero_ddof0_variance_and_positive_bound(method):
    result = finite_corpus_mean([1, 8, 2], [1, 8, 2], [0], [1], power=(0., .5, 1.), method=method)
    assert result.candidate_sample_variances == (0., 0., 0.)
    assert all(radius > 0 for radius in result.candidate_radii)
    assert result.rho == 1.
    assert result.selected_power == 0.
    assert result.design_unbiased and not result.selection_uses_audit_outcomes


@pytest.mark.parametrize("method", METHODS)
def test_zero_power_ignores_valid_proxy_values(method):
    a = finite_corpus_mean(SIZES, PROXIES, INDICES, OUTCOMES, power=0, method=method)
    b = finite_corpus_mean(SIZES, SIZES, INDICES, OUTCOMES, power=0, method=method)
    assert a.point == b.point
    assert a.radius == b.radius
    assert a.candidate_ranges == b.candidate_ranges
    assert a.candidate_sample_variances == b.candidate_sample_variances


def test_grid_selection_scope_differs_from_simultaneous_coverage():
    sizes, proxy, indices = [1, 1, 3, 5], [1, 1, 3, 5], [0, 3]
    powers = (1., .5, 0.)
    hs = finite_corpus_mean(sizes, proxy, indices, [0., 5.], power=powers)
    eb = finite_corpus_mean(sizes, proxy, indices, [0., 5.], power=powers,
                            method="empirical_bernstein_serfling")
    other = finite_corpus_mean(sizes, proxy, indices, [0., 0.], power=powers,
                               method="empirical_bernstein_serfling")
    assert hs.candidate_powers == eb.candidate_powers == (0., .5, 1.)
    assert hs.design_unbiased and not hs.selection_uses_audit_outcomes
    assert not eb.design_unbiased and eb.selection_uses_audit_outcomes
    assert hs.selected_power == 0.
    assert eb.selected_power == 1.
    assert other.selected_power == 0.
    assert "simultaneous" in eb.finite_coverage_scope
    assert eb.log_term == pytest.approx(log(10*3/.05))


@pytest.mark.parametrize("method", METHODS)
def test_fixed_and_singleton_grid_agree(method):
    scalar = finite_corpus_mean(SIZES, PROXIES, INDICES, OUTCOMES, power=.5, method=method)
    grid = finite_corpus_mean(SIZES, PROXIES, INDICES, OUTCOMES, power=(.5,), method=method)
    assert scalar.point == grid.point
    assert scalar.radius == grid.radius
    assert scalar.log_term == grid.log_term
    assert grid.design_unbiased and not grid.selection_uses_audit_outcomes
    assert grid.power_method == "finite-grid-minimum-radius"


def test_point_and_interval_remain_untruncated_for_unequal_groups():
    result = finite_corpus_mean([1, 100], [0, 0], [1], [100], power=0.)
    assert result.point == pytest.approx(200/101)
    assert result.point > 1
    assert result.interval.high > 1
    assert result.audited_rows == 100
    assert result.expected_audited_rows == 50.5


def test_fractional_real_totals_and_decimal_constants_are_supported():
    result = finite_corpus_mean([1]*4, [Fraction(1, 3)]*4, [0, 1, 2],
                                [Fraction(1, 10)]*3, power=0., method=METHODS[1])
    assert result.candidate_sample_variances == (0.,)
    assert result.point == pytest.approx(.1)
    assert result.radius > 0
    assert result.as_dict()["interval"]["point"] == result.point
    json.dumps(result.as_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        result.point = 0.


def test_equal_size_constant_proxy_has_exact_shift_invariance():
    result = finite_corpus_mean([2]*5, [2]*5, [0, 1, 3], [0, 1, 2],
                                power=(0., .25, .5, .75, 1.), method=METHODS[1])
    assert len(set(result.candidate_points)) == 1
    assert len(set(result.candidate_sample_variances)) == 1
    assert len(set(result.candidate_radii)) == 1
    assert result.selected_power == 0.


def test_smallest_positive_alpha_uses_log_domain_allocation():
    alpha = np.nextafter(0., 1.)
    for method, constant in ((METHODS[0], 2), (METHODS[1], 10)):
        result = finite_corpus_mean(SIZES, PROXIES, INDICES, OUTCOMES, alpha,
                                    power=(0., .5, 1.), method=method)
        assert np.isfinite(result.radius)
        assert result.log_term == pytest.approx(log(constant)+log(3)-log(alpha))


@pytest.mark.parametrize("sizes", [[], [1], [0, 1], [-1, 1], [True, 1], [1., 2.],
                                   ["1", 2], [None, 2], [np.nan, 2], [[1, 2]], [10**1000, 1]])
def test_invalid_group_sizes(sizes):
    with pytest.raises(ValueError):
        finite_corpus_mean(sizes, [0, 0], [0], [0])


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [4], [True], [0.], ["0"], [None], [[0]]])
def test_invalid_audited_indices(indices):
    with pytest.raises(ValueError):
        finite_corpus_mean(SIZES, PROXIES, indices, [0]*len(indices))


@pytest.mark.parametrize("field", ["group_prediction_totals", "audited_outcome_totals"])
@pytest.mark.parametrize("bad", [True, "0", None, complex(.1), np.nan, np.inf, -.1, 10**1000])
def test_strict_total_validation(field, bad):
    kwargs = dict(group_sizes=SIZES, group_prediction_totals=list(PROXIES),
                  audited_group_indices=INDICES, audited_outcome_totals=list(OUTCOMES))
    kwargs[field][0] = bad
    with pytest.raises(ValueError):
        finite_corpus_mean(**kwargs)


def test_totals_are_aligned_to_corresponding_sizes_and_masks_rejected():
    with pytest.raises(ValueError, match="within"):
        finite_corpus_mean([1, 4], [1.1, 0.], [0], [0.])
    with pytest.raises(ValueError, match="within"):
        finite_corpus_mean([1, 4], [0., 0.], [0], [2.])
    with pytest.raises(ValueError, match="length"):
        finite_corpus_mean(SIZES, PROXIES, INDICES, [0, 1])
    for name in ("group_sizes", "group_prediction_totals", "audited_group_indices", "audited_outcome_totals"):
        kwargs = dict(group_sizes=SIZES, group_prediction_totals=PROXIES,
                      audited_group_indices=INDICES, audited_outcome_totals=OUTCOMES)
        values = np.ma.array(kwargs[name], mask=[True]+[False]*(len(kwargs[name])-1))
        kwargs[name] = values
        with pytest.raises(ValueError, match="masked"):
            finite_corpus_mean(**kwargs)


def test_large_integer_bounds_are_checked_before_float_rounding():
    size = 2**53
    with pytest.raises(ValueError, match="within"):
        finite_corpus_mean([size, 1], [size+1, 0], [0], [0])
    with pytest.raises(ValueError, match="within"):
        finite_corpus_mean([size, 1], [0, 0], [0], np.array([size+1], dtype=np.int64))


@pytest.mark.parametrize("power", [True, -.1, 1.1, "auto", None, np.nan, (), (0., 0.),
                                   (0., True), (0., np.inf), [0., 1.]])
def test_invalid_power_or_grid(power):
    with pytest.raises(ValueError):
        finite_corpus_mean(SIZES, PROXIES, INDICES, OUTCOMES, power=power)


@pytest.mark.parametrize("alpha", [0., 1., True, np.nan, np.inf, "0.05", None,
                                   Fraction(1, 10**1000), Fraction(10**1000-1, 10**1000)])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        finite_corpus_mean(SIZES, PROXIES, INDICES, OUTCOMES, alpha)


@pytest.mark.parametrize("method", ["normal", "empirical_bernstein", None, True, []])
def test_invalid_method(method):
    with pytest.raises(ValueError):
        finite_corpus_mean(SIZES, PROXIES, INDICES, OUTCOMES, method=method)
