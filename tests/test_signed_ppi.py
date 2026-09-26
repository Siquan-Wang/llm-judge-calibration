"""Signed mean coefficients: independent formulas and compatibility contracts."""
from fractions import Fraction
import inspect
import json

import numpy as np
import pytest

from judgecal.ppi import prediction_powered_mean, prediction_powered_win_rate


def signed(f, y, u, **kwargs):
    return prediction_powered_mean(f, y, u, power_bounds=(-1., 1.), **kwargs)


def arrays():
    return (np.array([.1, .8, .7, .2, .4, .6]),
            np.array([.9, .2, .4, .8, .6, .3]),
            np.array([.8, .3, .7, .1, .2, .9, .6]))


def grouping(clustered):
    return ({"labeled_groups": [0, 0, 1, 2, 2, 2],
             "unlabeled_groups": [3, 4, 4, 5, 5, 5, 5]} if clustered else {})


def covariance(x, y, groups=None):
    """Independent test formula, using explicit cluster summation."""
    if groups is None:
        return sum((x-x.mean())*(y-y.mean())) / (len(x)*(len(x)-1))
    distinct = set(groups)
    totals = [(sum((x-x.mean())[np.asarray(groups) == g]),
               sum((y-y.mean())[np.asarray(groups) == g])) for g in distinct]
    return len(distinct)/(len(distinct)-1)*sum(a*b for a, b in totals)/len(x)**2


def test_negative_covariance_hand_calculated_optimum_and_variance():
    # V_L(Y)=V_L(F)=1/4, C_L(Y,F)=-1/4, V_U(F)=1/16.
    # lambda=-4/5, point=1/2 - (4/5)*(1/4-1/2)=7/10,
    # Q(lambda)=1/4 - (1/4)^2/(5/16)=1/20.
    result = signed([0., 1.], [1., 0.], [0., 0., 0., 1.], power="auto")
    assert result.selected_power == pytest.approx(-4/5)
    assert result.point == pytest.approx(7/10)
    assert result.standard_error**2 == pytest.approx(1/20)
    assert result.estimated_variance_ratio == pytest.approx(1/5)
    assert result.power_bounds == (-1., 1.)
    assert json.loads(json.dumps(result.as_dict()))["power_bounds"] == [-1., 1.]
    assert "[-1,1]" in " ".join(result.assumptions)
    assert "https://arxiv.org/pdf/2311.01453v2" in result.references
    positive = prediction_powered_mean([0., 1.], [1., 0.], [0., 0., 0., 1.], power="auto")
    assert positive.selected_power == 0 and positive.point == .5


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("power", [-1., -.37, 0., "auto"])
def test_negative_coefficients_equal_positive_coefficients_on_complemented_proxy(clustered, power):
    f, y, u = arrays()
    kwargs = grouping(clustered)
    direct = signed(f, y, u, power=power, **kwargs)
    opposite_power = "auto" if power == "auto" else -power
    inverse = prediction_powered_mean(1-f, y, 1-u, power=opposite_power, **kwargs)
    assert direct.selected_power == pytest.approx(-inverse.selected_power)
    assert direct.point == pytest.approx(inverse.point, abs=2e-15)
    assert direct.standard_error == pytest.approx(inverse.standard_error, abs=2e-15)
    assert direct.interval.low == pytest.approx(inverse.interval.low, abs=2e-15)
    assert direct.interval.high == pytest.approx(inverse.interval.high, abs=2e-15)
    assert direct.outcome_mean == inverse.outcome_mean


@pytest.mark.parametrize("clustered", [False, True])
def test_symmetric_auto_range_reflects_coefficient_when_proxy_is_complemented(clustered):
    f, y, u = arrays()
    a = signed(f, y, u, power="auto", **grouping(clustered))
    b = signed(1-f, y, 1-u, power="auto", **grouping(clustered))
    assert b.selected_power == pytest.approx(-a.selected_power)
    np.testing.assert_allclose([b.point, b.standard_error, b.interval.low, b.interval.high],
                               [a.point, a.standard_error, a.interval.low, a.interval.high], atol=2e-15)


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("power", [-.7, "auto"])
def test_complementing_outcomes_reflects_target_and_interval(clustered, power):
    f, y, u = arrays()
    first = signed(f, y, u, power=power, **grouping(clustered))
    other_power = "auto" if power == "auto" else -power
    other = signed(f, 1-y, u, power=other_power, **grouping(clustered))
    assert other.selected_power == pytest.approx(-first.selected_power)
    assert other.point == pytest.approx(1-first.point)
    assert other.interval.low == pytest.approx(1-first.interval.high)
    assert other.interval.high == pytest.approx(1-first.interval.low)
    assert other.standard_error == pytest.approx(first.standard_error)


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("bounds", [(-1., 1.), (-.2, .4), (-.8, 0.), (0., .4)])
def test_auto_matches_explicit_quadratic_and_never_exceeds_audit_only_estimated_variance(clustered, bounds):
    f, y, u = arrays()
    kwargs = grouping(clustered)
    ga, gu = kwargs.get("labeled_groups"), kwargs.get("unlabeled_groups")
    a = covariance(f, f, ga) + covariance(u, u, gu)
    b = covariance(y, f, ga)
    c = covariance(y, y, ga)
    expected = float(np.clip(b/a, *bounds))
    result = prediction_powered_mean(f, y, u, power="auto", power_bounds=bounds, **kwargs)
    assert result.selected_power == pytest.approx(expected)
    assert result.point == pytest.approx(y.mean()+expected*(u.mean()-f.mean()))
    assert result.standard_error**2 == pytest.approx(c-2*expected*b+expected**2*a)
    assert result.standard_error**2 <= c+1e-15
    candidates = np.linspace(*bounds, 1001)
    assert result.standard_error**2 <= min(c-2*candidates*b+candidates**2*a)+1e-15
    signed_result = signed(f, y, u, power="auto", **kwargs)
    positive = prediction_powered_mean(f, y, u, power="auto", **kwargs)
    assert signed_result.standard_error <= positive.standard_error+1e-15


def test_signed_cluster_auto_preserves_rows_groups_and_observation_weighting():
    f, y, u = arrays()
    original = signed(f, y, u, power="auto", **grouping(True))
    audit_order, prediction_order = np.array([5, 1, 3, 0, 2, 4]), np.array([6, 4, 2, 0, 1, 3, 5])
    ga = np.array(["a", "a", "b", "c", "c", "c"])
    gu = np.array(["x", "y", "y", "z", "z", "z", "z"])
    permuted = signed(f[audit_order], y[audit_order], u[prediction_order], power="auto",
                      labeled_groups=ga[audit_order], unlabeled_groups=gu[prediction_order])
    assert original.selected_power == pytest.approx(permuted.selected_power)
    assert original.point == pytest.approx(permuted.point)
    assert original.standard_error == pytest.approx(permuted.standard_error)
    human = signed(f, y, u, power=0, **grouping(True))
    assert human.point == y.mean()
    assert human.point != pytest.approx(np.mean([y[:2].mean(), y[2], y[3:].mean()]))


@pytest.mark.parametrize("bounds", [(-1., 1.), (-1., 0.), (0., 1.)])
def test_constant_proxy_zero_denominator_selects_audit_only(bounds):
    result = prediction_powered_mean([.5]*4, [0., .2, .8, 1.], [.7]*5,
                                      power="auto", power_bounds=bounds)
    assert result.selected_power == 0
    assert result.point == .5
    assert result.estimated_variance_ratio == 1.
    # Same constant outcomes and proxy give zero width, not certainty metadata.
    degenerate = prediction_powered_mean([0.]*4, [.3]*4, [1.]*5,
                                          power="auto", power_bounds=bounds)
    assert degenerate.selected_power == 0
    assert degenerate.estimated_variance_ratio is None
    assert degenerate.interval.low == degenerate.interval.high == .3


def test_zero_covariance_tie_selects_zero_and_narrow_bound_clips():
    zero = signed([0., 0., 1., 1.], [0., 1., 0., 1.], [0., 1.], power="auto")
    assert zero.selected_power == 0
    bound = prediction_powered_mean([0., 1.], [1., 0.], [0., 0., 0., 1.],
                                    power="auto", power_bounds=(-.1, .3))
    assert bound.selected_power == -.1
    fixed = prediction_powered_mean([0., 1.], [1., 0.], [0., 0., 0., 1.],
                                    power=-.1, power_bounds=(-.1, .3))
    assert bound.point == fixed.point and bound.standard_error == fixed.standard_error


def test_perfect_inverse_fixed_minus_one_and_population_oracle_identities():
    y = np.array([0., 0., 1., 1.])
    hidden = np.array([0., 0., 0., 0., 1., 1., 1., 1., 1., 1.])
    f, u = 1-y, 1-hidden
    fixed = signed(f, y, u, power=-1.)
    assert fixed.point == pytest.approx(hidden.mean())
    assert fixed.standard_error**2 == pytest.approx(np.var(hidden, ddof=1)/len(hidden))
    oracle = -len(hidden)/(len(y)+len(hidden))
    oracle_result = signed(f, y, u, power=oracle)
    assert oracle_result.point == pytest.approx(np.concatenate([y, hidden]).mean())
    # Hidden outcomes appear only in test expectations, never inference inputs.


def test_signed_fixed_correction_remains_untruncated():
    result = signed([1., 1.], [1., 1.], [0., 0.], power=-1.)
    assert result.point == result.interval.low == result.interval.high == 2.


@pytest.mark.parametrize("bounds", [
    None, [-1., 1.], np.array([-1., 1.]), (-1.,), (-1., 0., 1.), "-1,1",
    (0., 0.), (1., -1.), (.1, 1.), (-1., -.1), (-1.01, 1.), (-1., 1.01),
    (False, 1.), (-1., True), (-1+0j, 1.), ("-1", 1.), (np.nan, 1.),
    (-1., np.inf), (-10**1000, 1.), (-1., 10**1000),
    (-Fraction(1, 10**1000), 0.), (0., Fraction(1, 10**1000)),
])
def test_invalid_or_unrepresentable_bounds_rejected(bounds):
    with pytest.raises(ValueError, match="power_bounds"):
        prediction_powered_mean(*arrays(), power="auto", power_bounds=bounds)


@pytest.mark.parametrize("power", [-1.01, 1.01, np.nan, np.inf, True, "AUTO", None, 1j])
def test_invalid_signed_power_rejected(power):
    with pytest.raises(ValueError, match="power"):
        signed(*arrays(), power=power)


def test_fixed_power_must_lie_within_requested_interval_and_bounds_normalize_reals():
    with pytest.raises(ValueError, match="power"):
        prediction_powered_mean(*arrays(), power=-.4, power_bounds=(-.3, .7))
    with pytest.raises(ValueError, match="power"):
        prediction_powered_mean(*arrays(), power=.8, power_bounds=(-.3, .7))
    result = prediction_powered_mean(*arrays(), power=Fraction(-1, 5),
                                     power_bounds=(Fraction(-1, 3), np.float32(.75)))
    assert result.power_bounds == (-1/3, .75)
    assert result.selected_power == -.2
    assert all(type(v) is float for v in result.power_bounds)
    boundary = prediction_powered_mean(*arrays(), power=-Fraction(1, 3),
                                       power_bounds=(-Fraction(1, 3), Fraction(2, 3)))
    assert boundary.selected_power == boundary.power_bounds[0] == -1/3


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("power", [0., .3, 1., "auto"])
def test_default_inference_and_categorical_serialization_contract_unchanged(clustered, power):
    f, y, u = arrays()
    kwargs = grouping(clustered)
    default = prediction_powered_mean(f, y, u, power=power, **kwargs)
    explicit = prediction_powered_mean(f, y, u, power=power, power_bounds=(0., 1.), **kwargs)
    assert default.as_dict() == explicit.as_dict()
    if power == "auto":
        assert default.assumptions[5] == "Power minimizes estimated variance over [0,1]; no finite-sample MSE or coverage guarantee."
    else:
        assert default.assumptions[5] == f"Fixed lambda={power:g}; no tuning or guaranteed efficiency improvement."
    assert "power_bounds" not in inspect.signature(prediction_powered_win_rate).parameters
    categorical = prediction_powered_win_rate(["A", "B"], ["B", "A"], ["A", "B"], power=power)
    assert "power_bounds" not in categorical.as_dict()
    assert not hasattr(categorical, "power_bounds")
    assert "power_bounds" in default.as_dict()


def test_negative_power_requires_numeric_explicit_opt_in():
    with pytest.raises(ValueError, match="power"):
        prediction_powered_mean(*arrays(), power=-.5)
    with pytest.raises(ValueError, match="power"):
        prediction_powered_win_rate(["A", "B"], ["B", "A"], ["A", "B"], power=-.5)
    with pytest.raises(TypeError, match="power_bounds"):
        prediction_powered_win_rate(["A", "B"], ["B", "A"], ["A", "B"], power_bounds=(-1., 1.))
