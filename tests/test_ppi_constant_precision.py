"""Representably constant decimal scores must not fabricate proxy signal."""
import numpy as np
import pytest

from judgecal.ppi import _covariance_of_means, _variance_of_mean, prediction_powered_mean


@pytest.mark.parametrize("count,value", [(3, .4), (7, .1), (11, .3)])
@pytest.mark.parametrize("bounds", [(0., 1.), (-1., 1.)])
@pytest.mark.parametrize("grouped", [False, True])
def test_constant_decimal_proxy_has_zero_power_and_variance(count, value, bounds, grouped):
    outcomes = np.resize([.1, .3, .8], count)
    groups = ({"labeled_groups": np.arange(count), "unlabeled_groups": np.arange(count+1)+100}
              if grouped else {})
    result = prediction_powered_mean([value]*count, outcomes, [value]*(count+1),
                                     power="auto", power_bounds=bounds, **groups)
    human = prediction_powered_mean([value]*count, outcomes, [value]*(count+1),
                                    power=0., power_bounds=bounds, **groups)
    assert result.selected_power == 0
    assert result.point == human.point == outcomes.mean()
    assert result.standard_error == human.standard_error
    assert result.estimated_variance_ratio == 1


def test_exact_constants_have_zero_iid_and_unequal_cluster_moments():
    constant = np.full(7, .1)
    varying = np.array([.1, .3, .8, .2, .4, .5, .7])
    for groups in (None, np.array([0, 0, 0, 1, 2, 2, 3])):
        assert _variance_of_mean(constant, groups) == 0
        assert _covariance_of_means(constant, varying, groups) == 0
        assert _covariance_of_means(varying, constant, groups) == 0


def test_small_but_real_score_variation_is_not_thresholded_to_zero():
    scores = np.array([.4, np.nextafter(.4, 1.)])
    assert _variance_of_mean(scores, None) > 0
    assert _covariance_of_means(scores, scores, None) > 0
