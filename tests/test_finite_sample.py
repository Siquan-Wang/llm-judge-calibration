"""Finite-sample constants, simultaneous selection, and sampling contracts."""
import json
from fractions import Fraction
from math import comb, log, sqrt

import numpy as np
import pytest

from judgecal import FiniteSampleMeanResult, finite_sample_mean, prediction_powered_mean


F_L = np.array([.2, .8])
Y_L = np.array([.4, .6])
F_U = np.array([.1, .3, .7, .9])


def test_weighted_hoeffding_uses_independent_sum_ranges_not_separate_pool_union():
    result = finite_sample_mean(F_L, Y_L, F_U, power=.3)
    # Two audit terms have length 1.3/2; four prediction terms length .3/4.
    sum_squared_lengths = 2 * (.65)**2 + 4 * (.075)**2
    expected = sqrt(log(40) * sum_squared_lengths / 2)
    assert isinstance(result, FiniteSampleMeanResult)
    assert result.point == pytest.approx(.5)
    assert result.radius == pytest.approx(expected)
    assert result.interval.low == pytest.approx(.5 - expected)
    assert result.interval.high == pytest.approx(.5 + expected)
    assert result.hoeffding_log_term == pytest.approx(log(40))
    assert result.residual_radius is result.prediction_radius is None
    assert result.residual_log_term is result.prediction_log_term is None
    assert result.power_method == "fixed"
    assert result.interval.method == "ppi-mean-hoeffding-finite-sample-iid"


def test_empirical_bernstein_hand_calculated_paired_residual_and_prediction():
    result = finite_sample_mean(F_L, Y_L, F_U, method="empirical_bernstein", power=.3)
    # Residuals [.34,.36] have unbiased variance .0002; F_U variance 2/15.
    # Each two-sided pool gets alpha/2, each tail alpha/4; MP log is log(160).
    ell = log(160)
    residual = sqrt(.0002 * ell) + 7 * 1.3 * ell / 3
    prediction = .3 * (sqrt(ell / 15) + 7 * ell / 9)
    assert result.residual_radius == pytest.approx(residual)
    assert result.prediction_radius == pytest.approx(prediction)
    assert result.radius == pytest.approx(residual + prediction)
    assert result.residual_log_term == result.prediction_log_term == pytest.approx(ell)
    assert result.hoeffding_log_term is None


@pytest.mark.parametrize("method", ["hoeffding", "empirical_bernstein"])
@pytest.mark.parametrize("power", [0, .3, 1])
def test_finite_sample_and_asymptotic_methods_share_the_same_point(method, power):
    finite = finite_sample_mean(F_L, Y_L, F_U, method=method, power=power)
    asymptotic = prediction_powered_mean(F_L, Y_L, F_U, power=power)
    assert finite.point == asymptotic.point
    assert finite.prediction_mean == asymptotic.prediction_mean
    assert finite.outcome_mean == asymptotic.outcome_mean
    assert finite.residual_correction == asymptotic.residual_correction


@pytest.mark.parametrize("method", ["hoeffding", "empirical_bernstein"])
def test_fixed_zero_is_human_only_and_prediction_pool_cannot_change_it(method):
    first = finite_sample_mean(F_L, Y_L, F_U, method=method, power=0)
    second = finite_sample_mean([1, 0], Y_L, [1] * 100, method=method, power=0)
    assert first.point == second.point == pytest.approx(.5)
    assert first.radius == second.radius
    if method == "empirical_bernstein":
        # Y variance .02 and n2; full alpha goes to audit, not alpha/2.
        assert first.radius == pytest.approx(sqrt(.02 * log(80)) + 7 * log(80) / 3)
        assert first.prediction_radius == 0
        assert first.prediction_log_term is None
    else:
        assert first.radius == pytest.approx(sqrt(log(40) / 4))


@pytest.mark.parametrize("method", ["hoeffding", "empirical_bernstein"])
@pytest.mark.parametrize("power", [0, .6, 1])
def test_singleton_grid_matches_corresponding_fixed_interval(method, power):
    fixed = finite_sample_mean(F_L, Y_L, F_U, method=method, power=power)
    grid = finite_sample_mean(F_L, Y_L, F_U, method=method, power=(power,))
    assert fixed.interval == grid.interval
    assert fixed.radius == grid.radius
    assert grid.power_method == "finite-grid-minimum-radius"
    assert grid.candidate_powers == (float(power),)


def test_hoeffding_grid_pays_one_family_penalty_and_selects_minimum_power():
    result = finite_sample_mean(F_L, Y_L, F_U, power=(1, .5, 0))
    assert result.candidate_powers == (0, .5, 1)
    assert result.hoeffding_log_term == pytest.approx(log(120))
    assert result.selected_power == 0
    expected = tuple(sqrt(log(120) / 2 * ((1 + v)**2 / 2 + v*v / 4)) for v in (0, .5, 1))
    assert result.candidate_radii == pytest.approx(expected)
    assert result.radius > finite_sample_mean(F_L, Y_L, F_U, power=0).radius


def test_eb_grid_allocates_unlabeled_event_once_and_residual_events_per_power():
    fixed = finite_sample_mean(F_L, Y_L, F_U, method="empirical_bernstein", power=1)
    grid = finite_sample_mean(F_L, Y_L, F_U, method="empirical_bernstein", power=(1, 0, .5))
    assert grid.candidate_powers == (0, .5, 1)
    assert grid.residual_log_term == pytest.approx(log(480))
    assert grid.prediction_log_term == fixed.prediction_log_term == pytest.approx(log(160))
    # Normalizing each residual to [0,1] gives the same theorem application.
    expected = []
    for lam in grid.candidate_powers:
        z = (Y_L - lam * F_L + lam) / (1 + lam)
        normalized_bound = sqrt(2 * np.var(z, ddof=1) * log(480) / 2) + 7 * log(480) / 3
        expected.append((1 + lam) * normalized_bound + lam * fixed.prediction_radius)
    assert grid.candidate_radii == pytest.approx(expected)
    assert grid.radius == min(grid.candidate_radii)
    assert grid.selected_power == grid.candidate_powers[int(np.argmin(expected))]
    # Zero inside a larger family must not reuse the standalone alpha budget.
    human = finite_sample_mean(F_L, Y_L, F_U, method="empirical_bernstein", power=0)
    assert grid.candidate_radii[0] > human.radius


def test_eb_grid_can_select_positive_power_from_data_without_zero_variance_certainty():
    y = np.tile([0., 1.], 1000)
    u = np.tile([0., 1.], 10000)
    result = finite_sample_mean(y, y, u, method="empirical_bernstein", power=(0, .25, .5, .75, 1))
    assert result.selected_power > 0
    assert result.radius < result.candidate_radii[0]
    assert result.residual_radius > 0  # Known residual range is not its observed range.
    assert result.radius > 0


def test_eb_preserves_pairing_instead_of_using_only_marginal_variances():
    f = np.tile([0., 0., 1., 1.], 30)
    aligned = finite_sample_mean(f, f, f, method="empirical_bernstein")
    scrambled = finite_sample_mean(f, np.tile([0., 1., 0., 1.], 30), f, method="empirical_bernstein")
    assert aligned.point == scrambled.point
    assert aligned.prediction_radius == scrambled.prediction_radius
    assert aligned.residual_radius < scrambled.residual_radius
    assert aligned.residual_radius == pytest.approx(14 * log(160) / (3 * 119))


@pytest.mark.parametrize("method", ["hoeffding", "empirical_bernstein"])
@pytest.mark.parametrize("power", [0, .3, 1, (0, .25, .75, 1)])
def test_score_complement_reverses_every_candidate_and_selected_interval(method, power):
    f, y, u = np.array([.1, .7, .4, .8]), np.array([.2, .5, .9, .3]), np.array([.6, .05, .5])
    result = finite_sample_mean(f, y, u, method=method, power=power)
    reverse = finite_sample_mean(1-f, 1-y, 1-u, method=method, power=power)
    assert reverse.selected_power == result.selected_power
    assert reverse.candidate_radii == pytest.approx(result.candidate_radii)
    assert reverse.candidate_points == pytest.approx(1 - np.array(result.candidate_points))
    assert reverse.point == pytest.approx(1-result.point)
    assert reverse.interval.low == pytest.approx(1-result.interval.high)
    assert reverse.interval.high == pytest.approx(1-result.interval.low)


def test_deterministic_tie_breaks_toward_smaller_power(monkeypatch):
    # Isolate the public selection rule from the different analytic radii.
    monkeypatch.setattr("judgecal.finite_sample._eb_radius", lambda *args: 0.0)
    result = finite_sample_mean(F_L, Y_L, F_U, method="empirical_bernstein", power=(1, .75, .25))
    assert result.candidate_radii == (0, 0, 0)
    assert result.selected_power == .25


@pytest.mark.parametrize("method", ["hoeffding", "empirical_bernstein"])
def test_intervals_and_point_are_not_clipped_or_given_normal_standard_errors(method):
    result = finite_sample_mean([0, 0], [1, 1], [1, 1], method=method)
    assert result.point == 2
    assert result.interval.high > 2
    assert result.radius > 0
    assert not hasattr(result, "standard_error")
    assert result.estimand == "population mean of the bounded outcome"
    assert json.loads(json.dumps(result.as_dict(), allow_nan=False))["selected_power"] == 1


@pytest.mark.parametrize("method", ["hoeffding", "empirical_bernstein"])
@pytest.mark.parametrize("power", [0, 1, (0, .5, 1)])
def test_smallest_positive_alpha_has_finite_log_domain_bounds(method, power):
    alpha = np.nextafter(0., 1.)
    result = finite_sample_mean(F_L, Y_L, F_U, alpha=alpha, method=method, power=power)
    assert np.isfinite(result.radius)
    assert np.isfinite(result.interval.low) and np.isfinite(result.interval.high)
    assert result.alpha == alpha
    json.dumps(result.as_dict(), allow_nan=False)


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("bad", [
    [], [.2], .5, [[.2, .4]], [np.nan, .2], [np.inf, .2], [-.01, .2], [.2, 1.01],
    [None, .2], ["0.2", .4], [.2 + 0j, .4], [True, .4], [10**1000, 0],
    np.ma.array([.2, .4], mask=[False, True]),
])
def test_invalid_scores_are_not_coerced(position, bad):
    inputs = [F_L, Y_L, F_U]
    inputs[position] = bad
    with pytest.raises(ValueError):
        finite_sample_mean(*inputs)


@pytest.mark.parametrize("bad", [0, 1, -.1, np.nan, np.inf, True, "0.05", Fraction(1,10**1000), Fraction(10**1000-1,10**1000)])
def test_invalid_alpha_including_float_range_loss_is_rejected(bad):
    with pytest.raises(ValueError):
        finite_sample_mean(F_L, Y_L, F_U, alpha=bad)


@pytest.mark.parametrize("bad", ["auto", "0.5", True, -.1, 1.01, np.inf, np.nan, (), (0,0), (0,.5,"auto"), (np.nan,), [0,1]])
def test_invalid_or_unspecified_grid_power_is_rejected(bad):
    with pytest.raises(ValueError):
        finite_sample_mean(F_L, Y_L, F_U, power=bad)


@pytest.mark.parametrize("bad", ["normal", "EB", None, ["hoeffding"]])
def test_unknown_bound_method_is_rejected(bad):
    with pytest.raises(ValueError, match="method"):
        finite_sample_mean(F_L, Y_L, F_U, method=bad)


def test_misaligned_audit_and_group_argument_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        finite_sample_mean(F_L, [.2, .3, .4], F_U)
    with pytest.raises(TypeError, match="labeled_groups"):
        finite_sample_mean(F_L, Y_L, F_U, labeled_groups=[0,1])


def test_bounded_fraction_scores_and_power_are_supported():
    result = finite_sample_mean([Fraction(1,5), Fraction(4,5)], Y_L, F_U, power=Fraction(1,2))
    assert result.point == pytest.approx(.5)
    assert result.selected_power == .5


def test_exact_nontrivial_bernoulli_enumeration_for_human_hoeffding():
    # Enumerate every Binomial(20,.5) count, not a Monte Carlo assertion.
    # Coverage is strictly below 1, so this is not merely a vacuous interval.
    coverage = 0.0
    for k in range(21):
        y = [1.] * k + [0.] * (20-k)
        result = finite_sample_mean([.5]*20, y, [.5,.5], power=0)
        probability = comb(20,k) / 2**20
        coverage += probability * (result.interval.low <= .5 <= result.interval.high)
    assert .95 <= coverage < 1
    assert coverage == pytest.approx(1 - 2 * sum(comb(20,k) for k in range(4)) / 2**20)
