"""Independent finite-frame enumeration and primary-formula regression tests."""
from fractions import Fraction
from itertools import combinations
import math

import numpy as np
import pytest

from judgecal.finite_corpus import finite_corpus_mean


SIZES = (1, 1, 3, 5)
OUTCOMES = (0, 1, 0, 5)
PROXIES = (1, 0, 2, 4)
GRID = (0., .25, .5, .75, 1.)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
@pytest.mark.parametrize("power", [0., .25, .7, 1.])
def test_every_subset_recovers_fixed_power_row_target_and_exact_design_variance(count, power):
    groups, rows = len(SIZES), sum(SIZES)
    truth = Fraction(sum(OUTCOMES), rows)
    exact_power = Fraction(str(power))
    residuals = [Fraction(y) - exact_power*f for y, f in zip(OUTCOMES, PROXIES)]
    residual_mean = sum(residuals) / groups
    finite_variance = sum((value-residual_mean)**2 for value in residuals) / (groups-1)
    design_variance = Fraction(groups, rows)**2 * (1-Fraction(count, groups)) * finite_variance/count
    points, costs = [], []
    for selected in combinations(range(groups), count):
        result = finite_corpus_mean(SIZES, PROXIES, selected, [OUTCOMES[i] for i in selected], power=power)
        exact_point = exact_power*Fraction(sum(PROXIES), rows) + Fraction(groups, count*rows)*sum(residuals[i] for i in selected)
        assert result.point == pytest.approx(float(exact_point), abs=2e-15)
        assert result.design_unbiased is True
        assert result.selection_uses_audit_outcomes is False
        assert result.audited_rows == sum(SIZES[i] for i in selected)
        assert result.expected_audited_rows == pytest.approx(count*rows/groups)
        assert result.interval.low <= float(truth) <= result.interval.high
        points.append(result.point)
        costs.append(result.audited_rows)
    assert np.mean(points) == pytest.approx(float(truth), abs=2e-15)
    assert np.mean((np.asarray(points)-float(truth))**2) == pytest.approx(float(design_variance), abs=2e-15)
    assert np.mean(costs) == count*rows/groups
    if count == 2:
        # Both tempting alternative normalizations target different quantities.
        sampled_row_ratio = sum(Fraction(sum(OUTCOMES[i] for i in selected), sum(SIZES[i] for i in selected))
                                for selected in combinations(range(groups), count)) / math.comb(groups, count)
        assert sampled_row_ratio == Fraction(77, 144)
        assert sampled_row_ratio != truth
        assert sum(Fraction(y, n) for y, n in zip(OUTCOMES, SIZES))/groups != truth


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("method", ["hoeffding_serfling", "empirical_bernstein_serfling"])
def test_candidate_formulas_use_correct_rho_divisor_tail_and_grid_allocation(count, method):
    sizes = np.array([1, 2, 1, 3, 2, 5])
    proxies = np.array([0., 1., 1., 2., 0., 4.])
    outcomes = np.array([1., .7, .1, 2., 1., 3.])
    powers = (0., .5, 1.)
    alpha = .1
    result = finite_corpus_mean(sizes, proxies, range(count), outcomes[:count], alpha=alpha, power=powers, method=method)
    groups, rows = len(sizes), int(sizes.sum())
    rho = 1-(count-1)/groups if count <= groups/2 else (1-count/groups)*(1+1/count)
    log_term = math.log((2 if method == "hoeffding_serfling" else 10)*len(powers)/alpha)
    assert result.rho == pytest.approx(rho)
    assert result.finite_population_correction == pytest.approx(1-count/groups)
    assert result.log_term == pytest.approx(log_term)
    assert result.candidate_powers == powers
    radii = []
    for i, power in enumerate(powers):
        values = outcomes[:count] - power*proxies[:count]
        mean = math.fsum(values)/count
        empirical_variance = math.fsum((value-mean)**2 for value in values)/count
        lower = min(-power*proxies)
        upper = max(sizes-power*proxies)
        length = upper-lower
        if method == "hoeffding_serfling":
            radius = groups/rows * length * math.sqrt(rho*log_term/(2*count))
        else:
            radius = groups/rows * (math.sqrt(2*rho*empirical_variance*log_term/count)
                                    + (7/3+3/math.sqrt(2))*length*log_term/count)
        point = power*sum(proxies)/rows + groups/rows*mean
        assert result.candidate_ranges[i] == pytest.approx((lower, upper))
        assert result.candidate_sample_variances[i] == pytest.approx(empirical_variance)
        assert result.candidate_points[i] == pytest.approx(point)
        assert result.candidate_radii[i] == pytest.approx(radius)
        radii.append(radius)
    selected = min(range(len(powers)), key=lambda i: (radii[i], powers[i]))
    assert result.selected_power == powers[selected]
    assert result.radius == pytest.approx(radii[selected])
    uses_outcomes = method == "empirical_bernstein_serfling" and count > 1
    assert result.design_unbiased is (not uses_outcomes)
    assert result.selection_uses_audit_outcomes is uses_outcomes
    if count == 1:
        assert all(value == 0 for value in result.candidate_sample_variances)
        assert result.radius > 0


def test_single_group_eb_grid_selection_is_frame_only_and_design_unbiased():
    results = [finite_corpus_mean(SIZES, PROXIES, [index], [outcome], power=GRID,
                                  method="empirical_bernstein_serfling")
               for index, outcome in enumerate(OUTCOMES)]
    assert len({result.selected_power for result in results}) == 1
    assert all(result.candidate_radii == results[0].candidate_radii for result in results)
    assert all(result.design_unbiased and not result.selection_uses_audit_outcomes for result in results)
    assert np.mean([result.point for result in results]) == pytest.approx(sum(OUTCOMES)/sum(SIZES))
    for index, result in enumerate(results):
        changed = finite_corpus_mean(SIZES, PROXIES, [index], [SIZES[index]-OUTCOMES[index]],
                                     power=GRID, method="empirical_bernstein_serfling")
        assert changed.selected_power == result.selected_power
        assert changed.candidate_radii == result.candidate_radii


@pytest.mark.parametrize("method", ["hoeffding_serfling", "empirical_bernstein_serfling"])
def test_zero_observed_residual_range_does_not_replace_known_frame_range(method):
    # Audited residuals are exactly zero. Unsampled outcomes could still range
    # over each known group size, so a positive concentration radius is needed.
    result = finite_corpus_mean([1, 2, 3, 5], [0., 1., 2., 5.], [0, 1], [0., .5], power=.5, method=method)
    assert result.candidate_sample_variances == (0.,)
    assert result.candidate_ranges[0] == (-2.5, 2.5)
    assert result.radius > 0
    assert result.n_groups == 4 and result.n_rows == 11
    assert result.n_audited_groups == 2 and result.audited_rows == 3


@pytest.mark.parametrize("method", ["hoeffding_serfling", "empirical_bernstein_serfling"])
@pytest.mark.parametrize("power", [.7, GRID])
def test_census_is_exact_for_every_candidate_without_an_eb_remainder(method, power):
    indices = [3, 1, 0, 2]
    outcomes = np.array([.1, .7, 1.2, 3.3])
    result = finite_corpus_mean(SIZES, PROXIES, indices, outcomes[indices], power=power, method=method)
    truth = float(outcomes.sum()/sum(SIZES))
    assert result.census is True
    assert result.design_unbiased is True
    assert result.rho == result.finite_population_correction == 0
    assert result.radius == 0
    assert result.interval.low == result.interval.high == result.point
    assert result.point == pytest.approx(truth, abs=2e-15)
    assert all(point == result.point for point in result.candidate_points)
    assert all(radius == 0 for radius in result.candidate_radii)
    assert result.audited_rows == result.n_rows == 10


def test_equal_group_sizes_make_size_control_identical_and_ties_choose_zero():
    result = finite_corpus_mean([3]*5, [3.]*5, [0, 1], [0., 3.], power=GRID,
                                method="empirical_bernstein_serfling")
    assert result.candidate_points == (.5,)*len(GRID)
    assert all(value == result.candidate_radii[0] for value in result.candidate_radii)
    assert result.selected_power == 0
    assert all(value == 3 for low, high in result.candidate_ranges for value in [high-low])


def test_size_grid_coverage_does_not_imply_selected_point_is_design_unbiased():
    # A direct six-subset counterexample: fixed-lambda means average to .6,
    # while residual-variance grid selection averages to 13/30.
    selected_points = []
    for selected in combinations(range(4), 2):
        result = finite_corpus_mean(SIZES, SIZES, selected, [OUTCOMES[i] for i in selected],
                                    power=GRID, method="empirical_bernstein_serfling")
        assert result.design_unbiased is False
        assert result.selection_uses_audit_outcomes is True
        assert result.interval.low <= .6 <= result.interval.high
        selected_points.append(result.point)
    assert np.mean(selected_points) == pytest.approx(13/30)
    assert np.mean(selected_points) != pytest.approx(.6)


@pytest.mark.parametrize("method", ["hoeffding_serfling", "empirical_bernstein_serfling"])
def test_frame_reordering_and_audit_reordering_preserve_the_estimator(method):
    sizes, proxies, outcomes = map(np.asarray, (SIZES, PROXIES, OUTCOMES))
    original_ids = np.array([0, 2, 3])
    original = finite_corpus_mean(sizes, proxies, original_ids, outcomes[original_ids], power=GRID, method=method)
    permutation = np.array([3, 0, 2, 1])
    inverse = np.argsort(permutation)
    reordered_ids = inverse[original_ids[::-1]]
    reordered = finite_corpus_mean(sizes[permutation], proxies[permutation], reordered_ids,
                                  outcomes[original_ids[::-1]], power=GRID, method=method)
    assert reordered.selected_power == original.selected_power
    assert reordered.point == pytest.approx(original.point)
    assert reordered.radius == pytest.approx(original.radius)
    assert reordered.candidate_points == pytest.approx(original.candidate_points)
    assert reordered.candidate_radii == pytest.approx(original.candidate_radii)
    assert reordered.audited_rows == original.audited_rows


def test_dominating_group_is_not_clipped_or_counted_as_one_reference_label():
    result = finite_corpus_mean([1, 1, 1, 1000], [0., 0., 0., 0.], [3], [1000.], power=0.)
    assert result.point == pytest.approx(4000/1003)
    assert result.point > 1
    assert result.audited_rows == 1000
    assert result.expected_audited_rows == pytest.approx(1003/4)
    assert result.candidate_ranges == ((0., 1000.),)
    assert result.radius > 0
