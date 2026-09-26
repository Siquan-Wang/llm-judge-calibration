"""Random-pool prediction has different uncertainty from population inference."""
from fractions import Fraction
from itertools import combinations
import json

import numpy as np
import pytest
from scipy.stats import norm

from judgecal.pool_prediction import (
    AuditResidualMeanResult, PoolMeanPredictionResult, audit_residual_mean,
    predict_heldout_mean,
)
from judgecal.ppi import prediction_powered_mean


F = np.array([.1, .8, .7, .2, .4, .6])
Y = np.array([.9, .2, .4, .8, .6, .3])
U = np.array([.8, .3, .7, .1, .2, .9, .6])


def test_fixed_pool_variance_hand_calculation_and_distinct_result_scope():
    # Residuals at lambda=-.3 are [.8,.5]; sample variance=.045.
    # Pool prediction variance=(1/2+1/4)*.045=.03375.
    result = predict_heldout_mean([0., 1.], [.8, .2], [0., 0., 0., 1.], power=-.3)
    assert isinstance(result, PoolMeanPredictionResult)
    assert result.point == pytest.approx(.575)
    assert result.audit_residual_variance == pytest.approx(.045)
    assert result.prediction_standard_error**2 == pytest.approx(.03375)
    assert result.prediction_interval.low == pytest.approx(.575-norm.isf(.025)*np.sqrt(.03375))
    assert result.prediction_interval.high == pytest.approx(.575+norm.isf(.025)*np.sqrt(.03375))
    assert result.estimand == "random held-out outcome mean"
    assert result.coverage_scope == "marginal over independent iid audit and held-out pool draws"
    assert result.power_bounds == (-1., 1.) and result.power_method == "fixed"
    assert not hasattr(result, "interval") and not hasattr(result, "standard_error")
    assert "conditional fixed-pool guarantee" in " ".join(result.assumptions)
    assert json.loads(json.dumps(result.as_dict(), allow_nan=False))["prediction_interval"]["point"] == result.point


def test_audit_only_pool_variance_includes_unobserved_target_residual_variation():
    pool = predict_heldout_mean(F, Y, U, power=0.)
    population = prediction_powered_mean(F, Y, U, power=0.)
    assert pool.point == population.point == Y.mean()
    assert pool.prediction_standard_error**2 == pytest.approx((1/len(Y)+1/len(U))*np.var(Y, ddof=1))
    assert pool.prediction_standard_error**2-population.standard_error**2 == pytest.approx(np.var(Y, ddof=1)/len(U))


def test_auto_minimizes_audit_residual_only_and_ignores_unlabeled_variance():
    expected = np.cov(Y, F, ddof=1)[0, 1]/np.var(F, ddof=1)
    pool = predict_heldout_mean(F, Y, U, power="auto")
    point = audit_residual_mean(F, Y, U)
    changed = predict_heldout_mean(F, Y, [.1, .1], power="auto")
    assert pool.selected_power == pytest.approx(np.clip(expected, -1., 1.))
    assert pool.selected_power == changed.selected_power == point.selected_power
    assert pool.point == point.point
    assert point.criterion_value == pytest.approx(pool.audit_residual_variance/len(Y))
    population = prediction_powered_mean(F, Y, U, power="auto", power_bounds=(-1., 1.))
    assert abs(population.selected_power) < abs(pool.selected_power)
    candidates = np.linspace(-1, 1, 1001)
    assert pool.audit_residual_variance <= min(np.var(Y-v*F, ddof=1) for v in candidates)+1e-15


def test_default_fixed_one_matches_population_point_without_reusing_its_variance():
    pool = predict_heldout_mean(F, Y, U)
    population = prediction_powered_mean(F, Y, U)
    assert pool.selected_power == 1 and pool.power_method == "fixed"
    assert pool.point == population.point  # Original operation order is preserved.
    assert pool.prediction_standard_error != pytest.approx(population.standard_error)


@pytest.mark.parametrize("power", [-1., -.4, 0., .3, 1.])
def test_both_interval_apis_have_identical_centers_for_identical_fixed_coefficients(power):
    pool = predict_heldout_mean(F, Y, U, power=power)
    population = prediction_powered_mean(F, Y, U, power=power, power_bounds=(-1., 1.))
    assert pool.point == population.point


@pytest.mark.parametrize("inverse", [False, True])
def test_perfect_proxy_recovers_random_pool_and_zero_error(inverse):
    y = np.array([0., 0., 1., 1.])
    hidden = np.array([0., 0., 0., 1., 1., 1., 1., 1.])
    f, u = (1-y, 1-hidden) if inverse else (y, hidden)
    power = -1. if inverse else 1.
    result = predict_heldout_mean(f, y, u, power=power)
    assert result.point == hidden.mean()
    assert result.audit_residual_variance == result.prediction_standard_error == 0.
    assert result.prediction_interval.low == result.prediction_interval.high == result.point
    auto = predict_heldout_mean(f, y, u, power="auto")
    assert auto.selected_power == pytest.approx(power)
    assert auto.point == pytest.approx(hidden.mean())


@pytest.mark.parametrize("power", [-.5, "auto"])
def test_proxy_inversion_and_outcome_reflection(power):
    a = predict_heldout_mean(F, Y, U, power=power)
    reflected_power = "auto" if power == "auto" else -power
    b = predict_heldout_mean(1-F, Y, 1-U, power=reflected_power)
    assert b.selected_power == pytest.approx(-a.selected_power)
    np.testing.assert_allclose([b.point, b.prediction_standard_error, b.prediction_interval.low, b.prediction_interval.high],
                               [a.point, a.prediction_standard_error, a.prediction_interval.low, a.prediction_interval.high], atol=1e-15)
    c = predict_heldout_mean(F, 1-Y, U, power=reflected_power)
    assert c.selected_power == pytest.approx(-a.selected_power)
    assert c.point == pytest.approx(1-a.point)
    assert c.prediction_interval.low == pytest.approx(1-a.prediction_interval.high)
    assert c.prediction_interval.high == pytest.approx(1-a.prediction_interval.low)


def test_constant_audit_proxy_falls_back_zero_even_if_prediction_proxy_varies():
    # Mean([.4]*3) is not representably .4 on common NumPy builds.
    # Exact constancy, rather than a spurious rounding variance, controls fallback.
    f, y, u = [.4]*3, [.1, .3, .8], [0., 1., 0., 1.]
    assert predict_heldout_mean(f, y, u, power="auto").selected_power == 0.
    assert audit_residual_mean(f, y, u).selected_power == 0.
    groups = {"labeled_groups": [0, 0, 1], "unlabeled_groups": [2, 2, 3, 3]}
    assert audit_residual_mean(f, y, u, **groups).selected_power == 0.


def test_zero_covariance_and_degenerate_variance_remain_explicit():
    zero = predict_heldout_mean([0., 0., 1., 1.], [0., 1., 0., 1.], [0., 1.], power="auto")
    assert zero.selected_power == 0.
    degenerate = predict_heldout_mean([0., 1.], [.5, .5], [0., 1.], power="auto")
    assert degenerate.selected_power == 0.
    assert degenerate.prediction_standard_error == 0.
    assert "zero sample variance is not certainty" in " ".join(degenerate.assumptions)
    untruncated = predict_heldout_mean([0., 0.], [1., 1.], [1., 1.])
    assert untruncated.point == untruncated.prediction_interval.low == untruncated.prediction_interval.high == 2.


def test_constant_decimal_residual_has_zero_variance_without_thresholding_real_variation():
    constant = predict_heldout_mean([0.]*7, [.1]*7, [0.]*3)
    assert constant.audit_residual_variance == constant.prediction_standard_error == 0.
    assert constant.prediction_interval.low == constant.prediction_interval.high == constant.point
    varying = predict_heldout_mean([0., 0.], [.1, np.nextafter(.1, 1.)], [0., 0.])
    assert varying.audit_residual_variance > 0.
    assert varying.prediction_standard_error > 0.


def test_point_only_unequal_cluster_slope_and_no_interval_fields():
    groups = np.array([0, 0, 1, 2, 2, 2])
    heldout_groups = np.array([3, 4, 4, 5, 5, 5, 5])
    totals_f = np.array([sum((F-F.mean())[groups == g]) for g in set(groups)])
    totals_y = np.array([sum((Y-Y.mean())[groups == g]) for g in set(groups)])
    slope = np.clip(np.dot(totals_f, totals_y)/np.dot(totals_f, totals_f), -1., 1.)
    result = audit_residual_mean(F, Y, U, labeled_groups=groups, unlabeled_groups=heldout_groups)
    assert isinstance(result, AuditResidualMeanResult)
    assert result.selected_power == pytest.approx(slope)
    assert result.point == pytest.approx(Y.mean()+slope*(U.mean()-F.mean()))
    residual_sums = totals_y-slope*totals_f
    assert result.criterion_value == pytest.approx(3/2*np.dot(residual_sums, residual_sums)/len(Y)**2)
    assert result.n_labeled_groups == result.n_unlabeled_groups == 3
    assert result.variance_method == "one-way-cluster-robust"
    assert result.power_method == "auto-audit-residual-cluster-sandwich"
    for name in ("interval", "prediction_interval", "standard_error", "prediction_standard_error", "estimated_variance_ratio"):
        assert not hasattr(result, name) and name not in result.as_dict()
    changed = audit_residual_mean(F, Y, [.4, .7], labeled_groups=groups, unlabeled_groups=[8, 9])
    assert changed.selected_power == result.selected_power
    assert changed.criterion_value == result.criterion_value
    reverse = audit_residual_mean(1-F, Y, 1-U, labeled_groups=groups, unlabeled_groups=heldout_groups)
    assert reverse.selected_power == pytest.approx(-result.selected_power)
    assert reverse.point == pytest.approx(result.point)


@pytest.mark.parametrize("power", [-1., 0., .4, 1.])
def test_exhaustive_disjoint_partition_residual_covariance_and_variance_identity(power):
    # All C(8,2)*C(6,3)=560 distinct joint partitions of a fixed finite corpus.
    # This validates fixed-coefficient design algebra, not adaptive normal coverage.
    y = np.array([0., .2, .5, .9, 1., .4, .7, .1])
    f = np.array([.8, .1, .4, .3, 1., .6, .2, .9])
    residual = y-power*f
    M, n, N = 8, 2, 3
    l_means, u_means, estimated_variances, errors = [], [], [], []
    for audit_tuple in combinations(range(M), n):
        a = np.array(audit_tuple)
        for heldout_tuple in combinations(sorted(set(range(M))-set(audit_tuple)), N):
            u = np.array(heldout_tuple)
            result = predict_heldout_mean(f[a], y[a], f[u], power=power)
            error = result.point-y[u].mean()
            assert error == pytest.approx(residual[a].mean()-residual[u].mean(), abs=1e-15)
            l_means.append(residual[a].mean())
            u_means.append(residual[u].mean())
            errors.append(error)
            estimated_variances.append(result.prediction_standard_error**2)
    assert len(errors) == 560
    population_variance = np.var(residual, ddof=1)
    np.testing.assert_allclose(np.var(l_means), (1/n-1/M)*population_variance, atol=2e-16)
    np.testing.assert_allclose(np.var(u_means), (1/N-1/M)*population_variance, atol=2e-16)
    covariance = np.mean((l_means-np.mean(l_means))*(u_means-np.mean(u_means)))
    assert covariance == pytest.approx(-population_variance/M, abs=2e-16)
    target = (1/n+1/N)*population_variance
    assert np.var(errors) == pytest.approx(target, abs=2e-16)
    assert np.mean(estimated_variances) == pytest.approx(target, abs=2e-16)
    assert np.mean(errors) == pytest.approx(0., abs=2e-16)


@pytest.mark.parametrize("function", [predict_heldout_mean, audit_residual_mean])
@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("bad", [[], [.5], [[.2, .3]], [None, .1], [True, .1],
                                 [".1", .2], [np.nan, .2], [np.inf, .3], [-.1, .4],
                                 [1.1, .3], [1j, .3], [10**1000, 0],
                                 np.ma.array([.1, .3], mask=[False, True])])
def test_strict_bounded_score_validation(function, position, bad):
    values = [F, Y, U]
    values[position] = bad
    with pytest.raises(ValueError):
        function(*values)


@pytest.mark.parametrize("function", [predict_heldout_mean, audit_residual_mean])
def test_audit_length_mismatch_and_bound_contract(function):
    with pytest.raises(ValueError, match="same length"):
        function(F[:2], Y, U)
    for bad in [None, [-1., 1.], (-1.,), (0., 0.), (-2., 1.), (.1, 1.),
                (-1., -.2), (np.nan, 1.), (-1., True), (-Fraction(1, 10**1000), 0.)]:
        with pytest.raises(ValueError, match="power_bounds"):
            function(F, Y, U, power_bounds=bad)


@pytest.mark.parametrize("kwargs", [
    {"alpha": 0.}, {"alpha": 1.}, {"alpha": True}, {"alpha": np.nan},
    {"alpha": Fraction(1, 10**1000)}, {"alpha": 10**1000},
    {"power": None}, {"power": True}, {"power": "AUTO"}, {"power": -1.01},
    {"power": np.inf}, {"power": -.5, "power_bounds": (-.3, .3)},
])
def test_prediction_option_validation(kwargs):
    with pytest.raises(ValueError):
        predict_heldout_mean(F, Y, U, **kwargs)


def test_real_bound_boundary_and_extreme_alpha_supported():
    result = predict_heldout_mean(F, Y, U, power=-Fraction(1, 3),
                                  power_bounds=(-Fraction(1, 3), Fraction(2, 3)),
                                  alpha=np.nextafter(0., 1.))
    assert result.selected_power == result.power_bounds[0] == -1/3
    assert np.isfinite(result.prediction_interval.low) and np.isfinite(result.prediction_interval.high)
    with pytest.raises(TypeError, match="labeled_groups"):
        predict_heldout_mean(F, Y, U, labeled_groups=[0]*len(F))


@pytest.mark.parametrize("kwargs", [
    {"labeled_groups": [0, 0, 1, 2, 2, 2]},
    {"unlabeled_groups": [3, 3, 4, 4, 5, 5, 5]},
    {"labeled_groups": [0]*6, "unlabeled_groups": [1, 1, 2, 2, 2, 2, 2]},
    {"labeled_groups": [0, 0, 1, 2, 2, 2], "unlabeled_groups": [2, 3, 3, 3, 3, 3, 3]},
    {"labeled_groups": [0, 0, 1, 2, 2, None], "unlabeled_groups": [3, 3, 4, 4, 5, 5, 5]},
    {"labeled_groups": [0, 0, 1, 2, 2, 2], "unlabeled_groups": [3, 3, 4, 4, 5, 5, np.inf]},
    {"labeled_groups": [0, 0, 1], "unlabeled_groups": [3, 3, 4, 4, 5, 5, 5]},
    {"labeled_groups": np.ma.array([0, 0, 1, 2, 2, 2], mask=[0, 0, 0, 0, 0, 1]),
     "unlabeled_groups": [3, 3, 4, 4, 5, 5, 5]},
])
def test_point_only_group_validation(kwargs):
    with pytest.raises(ValueError):
        audit_residual_mean(F, Y, U, **kwargs)
