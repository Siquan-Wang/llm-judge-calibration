"""Independent deterministic checks of pilot isolation and its budget cost.

Finite enumeration checks the implementation on a known law; it is not a
substitute for the conditional-expectation argument or a coverage theorem.
"""
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from judgecal.pilot_study import evaluate_draw, pilot_power
from judgecal.ppi import prediction_powered_mean


def test_fractional_regression_slope_is_clipped_after_pool_size_factor():
    # Var(F)=1/8, Cov(Y,F)=3/20, slope=6/5. N/(m+N)=1/2.
    result = pilot_power([.25, .75], [.1, .7], 2, 2)
    assert result["variance"] == pytest.approx(1 / 8)
    assert result["covariance"] == pytest.approx(3 / 20)
    assert result["selected_power"] == pytest.approx(3 / 5)
    assert result["selected_power"] != pytest.approx(.5)
    assert result["constant_proxy"] is False


@pytest.mark.parametrize("value", [.1, .3, .7])
def test_exact_decimal_constant_pilot_has_exact_zero_moments(value):
    result = pilot_power(np.full(7, value), np.linspace(0, 1, 7), 16, 200)
    assert result["constant_proxy"] is True
    assert result["variance"] == 0
    assert result["covariance"] == 0
    assert result["selected_power"] == 0
    assert (result["n_pilot"], result["n_inference"], result["n_unlabeled"]) == (7, 16, 200)


def test_exact_constant_outcome_has_zero_coefficient_without_constant_proxy():
    result = pilot_power(np.linspace(.1, .9, 7), np.full(7, .1), 16, 200)
    assert result["constant_proxy"] is False
    assert result["variance"] > 0
    assert result["covariance"] == 0
    assert result["selected_power"] == 0


def test_small_nonzero_proxy_spread_is_not_treated_as_constant():
    result = pilot_power([.5 - 1e-10, .5 + 1e-10], [.2, .8], 2, 2)
    assert result["constant_proxy"] is False
    assert 0 < result["variance"] < 1e-18
    assert result["covariance"] > 0
    assert result["selected_power"] == 1


def test_inverse_pilot_uses_declared_positive_constraint():
    result = pilot_power([0., 1., 0., 1.], [1., 0., 1., 0.], 16, 200)
    assert result["covariance"] < 0
    assert result["variance"] > 0
    assert result["selected_power"] == 0


def test_exact_binary_zero_covariance_keeps_full_alpha_zero_power_eb_branch():
    # n=10, sum(Y)=2, sum(F)=5 and sum(YF)=1, so the integer cross-moment
    # numerator 10*1-2*5 is zero. A tiny positive roundoff coefficient would
    # incorrectly move the interval onto EB's wider split-alpha branch.
    y_p = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
    f_p = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 0], dtype=float)
    result = pilot_power(f_p, y_p, 10, 200)
    assert result["covariance"] == 0
    assert result["selected_power"] == 0
    assert result["constant_proxy"] is False
    y_i = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 1], dtype=float)
    f_i = np.array([0, 1] * 5, dtype=float)
    records = _by_method(evaluate_draw(np.r_[y_p, y_i], np.r_[f_p, f_i], np.tile([0., 1.], 100)))
    eb = records["pilot50_eb"]
    log = np.log(4 / .05)
    radius = np.sqrt(2 * np.var(y_i, ddof=1) * log / 10) + 7 * log / (3 * 9)
    assert eb["selected_power"] == 0
    assert eb["point"] == y_i.mean()
    assert eb["interval_low"] == pytest.approx(y_i.mean() - radius)
    assert eb["interval_high"] == pytest.approx(y_i.mean() + radius)


def test_small_real_fractional_covariance_is_not_rounded_to_zero():
    y = np.array([.5, .5 + 1e-14])
    result = pilot_power([0., 1.], y, 10, 200)
    expected = (y[1] - y[0]) * 200 / 210
    assert result["covariance"] > 0
    assert result["selected_power"] > 0
    assert result["selected_power"] == pytest.approx(expected, rel=1e-12, abs=0.)


@pytest.mark.parametrize(
    "f_p,y_p,expected_power",
    [
        ([0., 0.], [0., 1.], 0.),
        ([0., 1.], [1., 0.], 0.),
        ([0., 0., 1., 1.], [0., 1., 1., 1.], .25),
        ([0., 1.], [0., 1.], .5),
    ],
)
def test_conditional_mean_and_variance_identity_on_all_64_inference_states(
    f_p, y_p, expected_power
):
    # This law is independent of the pilot and has p=3/10, sensitivity=4/5,
    # specificity=7/10. Every conditioned pilot has positive probability under
    # this law. Its joint-state probabilities and q=9/20 are exact.
    states = [
        (0., 0., Fraction(49, 100)),
        (0., 1., Fraction(21, 100)),
        (1., 0., Fraction(6, 100)),
        (1., 1., Fraction(24, 100)),
    ]
    proxy_states = [(0., Fraction(11, 20)), (1., Fraction(9, 20))]
    fitted = pilot_power(f_p, y_p, 2, 2)
    power = fitted["selected_power"]
    assert power == pytest.approx(expected_power)
    mass = Fraction(0)
    mean = mse = mean_reported_variance = 0.
    configurations = 0
    for residual_states in product(states, repeat=2):
        y_i = [state[0] for state in residual_states]
        f_i = [state[1] for state in residual_states]
        residual_probability = residual_states[0][2] * residual_states[1][2]
        for prediction_states in product(proxy_states, repeat=2):
            probability = residual_probability * prediction_states[0][1] * prediction_states[1][1]
            f_u = [state[0] for state in prediction_states]
            result = prediction_powered_mean(f_i, y_i, f_u, power=power)
            mass += probability
            mean += float(probability) * result.point
            mse += float(probability) * (result.point - .3) ** 2
            mean_reported_variance += float(probability) * result.standard_error ** 2
            configurations += 1
    # Var(Y)=.21, Var(F)=.2475, Cov(Y,F)=.105; m=N=2.
    expected_variance = .105 - .105 * power + .2475 * power ** 2
    assert configurations == 64
    assert mass == 1
    assert mean == pytest.approx(.3, abs=2e-15)
    assert mse == pytest.approx(expected_variance, abs=2e-15)
    assert mean_reported_variance == pytest.approx(expected_variance, abs=2e-15)


def _labeled_fixture():
    # Both prefixes are nonconstant, with a positive, non-perfect association.
    y_l = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 1] * 2, dtype=float)
    f_l = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1] * 2, dtype=float)
    f_u = np.tile([0., 1., 1., 1., 0.], 40)
    return y_l, f_l, f_u


def _by_method(records):
    result = {row["method"]: row for row in records}
    assert len(result) == len(records) == 8
    return result


@pytest.mark.parametrize("pilot_count,method_prefix", [(4, "pilot20"), (10, "pilot50")])
@pytest.mark.parametrize("mutation", ["residual_outcomes", "residual_proxies", "prediction_pool"])
def test_final_pool_mutations_do_not_change_pilot_tuning(pilot_count, method_prefix, mutation):
    y_l, f_l, f_u = _labeled_fixture()
    before = _by_method(evaluate_draw(y_l, f_l, f_u))
    mutated_y, mutated_f, mutated_u = y_l.copy(), f_l.copy(), f_u.copy()
    if mutation == "residual_outcomes":
        mutated_y[pilot_count:] = 1 - mutated_y[pilot_count:]
    elif mutation == "residual_proxies":
        mutated_f[pilot_count:] = .2
    else:
        mutated_u[:] = .125
    after = _by_method(evaluate_draw(mutated_y, mutated_f, mutated_u))
    for family in ("normal", "eb"):
        key = f"{method_prefix}_{family}"
        for field in ("selected_power", "pilot_covariance", "pilot_variance", "pilot_constant"):
            assert after[key][field] == before[key][field]
    # A separate positive control prevents a selector that ignores every input
    # from satisfying the isolation assertions trivially.
    mutated_y[:pilot_count] = 1 - f_l[:pilot_count]
    changed_pilot = _by_method(evaluate_draw(mutated_y, f_l, f_u))
    assert before[f"{method_prefix}_normal"]["selected_power"] > 0
    assert changed_pilot[f"{method_prefix}_normal"]["selected_power"] == 0


def test_pilot_cost_is_included_and_only_residual_rows_enter_final_inference():
    y_l, f_l, f_u = _labeled_fixture()
    results = _by_method(evaluate_draw(y_l, f_l, f_u))
    for row in results.values():
        assert row["total_budget"] == row["total_labels_used"] == 20
        assert row["n_unlabeled"] == 200
        assert row["proxy_scores_supplied"] == 220
        expected_used = 0 if row["method"] in ("human_normal", "human_eb") else 220
        assert row["unique_prediction_scores_used"] == expected_used
        assert row["n_pilot"] + row["n_labeled"] == 20
    for prefix, count in (("pilot20", 4), ("pilot50", 10)):
        normal, eb = results[f"{prefix}_normal"], results[f"{prefix}_eb"]
        residual_count = 20 - count
        # Calculate this without the pilot helper or inference API. Using B
        # instead of m in the shrinkage factor would change these coefficients.
        f_centered = f_l[:count] - f_l[:count].mean()
        y_centered = y_l[:count] - y_l[:count].mean()
        slope = np.dot(f_centered, y_centered) / np.dot(f_centered, f_centered)
        power = np.clip(slope * 200 / (residual_count + 200), 0, 1)
        point = y_l[count:].mean() + power * (f_u.mean() - f_l[count:].mean())
        variance = (
            np.var(y_l[count:] - power * f_l[count:], ddof=1) / residual_count
            + power ** 2 * np.var(f_u, ddof=1) / 200
        )
        assert normal["n_pilot"] == eb["n_pilot"] == count
        assert normal["n_labeled"] == eb["n_labeled"] == residual_count
        assert normal["selected_power"] == pytest.approx(power)
        assert normal["point"] == pytest.approx(point)
        assert normal["estimated_variance"] == pytest.approx(variance)
        assert normal["standard_error"] ** 2 == pytest.approx(variance)
        assert eb["selected_power"] == normal["selected_power"]
        assert eb["point"] == normal["point"]
    for method in ("human_normal", "human_eb", "same_audit_normal", "same_audit_grid_eb"):
        assert results[method]["n_labeled"] == 20
        assert results[method]["n_pilot"] == 0
    assert results["human_normal"]["point"] == y_l.mean()
    assert results["human_eb"]["point"] == results["human_normal"]["point"]


def test_pilot_and_same_audit_formulas_agree_when_proxy_variances_match():
    f_l = np.array([.25, .75])
    y_l = np.array([.1, .7])
    f_u = f_l[::-1].copy()
    pilot = pilot_power(f_l, y_l, 2, 2)
    same_audit = prediction_powered_mean(f_l, y_l, f_u, power="auto")
    assert pilot["selected_power"] == pytest.approx(.6)
    assert pilot["selected_power"] == pytest.approx(same_audit.selected_power)
    assert same_audit.point == pytest.approx(.4)
