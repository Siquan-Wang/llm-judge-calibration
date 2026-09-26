"""Numerical invariants of the original penalized Platt objective."""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import expit

from judgecal import platt_scaling


@pytest.mark.parametrize("offset", [-1e8, -1e4, 1e4, 1e8])
@pytest.mark.parametrize("labels", [[0, 1, 0, 1], [1, 0, 1, 0]])
def test_score_translation_preserves_fit_and_original_coefficient_meaning(offset, labels):
    scores = np.arange(4, dtype=float)
    base = platt_scaling(scores, labels)
    shifted = platt_scaling(scores + offset, labels)
    grid = np.linspace(-2, 5, 29)
    np.testing.assert_allclose(shifted(grid + offset), base(grid), atol=1e-10, rtol=0)
    a, b = base.params_
    shifted_a, shifted_b = shifted.params_
    assert shifted_a == pytest.approx(a, abs=1e-12)
    assert shifted_b == pytest.approx(b - a * offset, rel=1e-12)
    # The old large-offset fit silently returned .5 for every score.
    assert np.ptp(shifted(scores + offset)) > .5


@pytest.mark.parametrize("regularization", [10.0, 1e4])
def test_internal_scaling_preserves_the_raw_slope_penalty(regularization):
    scores = np.array([10., 20., 30., 40.])
    labels = np.array([0, 1, 0, 1])
    centered = scores - 25
    # By symmetry the optimum centered intercept is zero. Solve the raw
    # slope score equation independently; strict convexity identifies it.
    derivative = lambda a: np.dot(centered, expit(a * centered) - labels) + regularization * a
    expected_a = brentq(derivative, -1, 1, xtol=1e-14)
    fitted = platt_scaling(scores, labels, regularization=regularization)
    actual_a, actual_b = fitted.params_
    assert actual_a == pytest.approx(expected_a, abs=1e-7)
    assert actual_b == pytest.approx(-25 * expected_a, abs=3e-6)
    np.testing.assert_allclose(fitted(scores), expit(expected_a * centered), atol=2e-6, rtol=0)
    # A coefficient penalty accidentally left in internal units fails this
    # comparison by orders of magnitude for these score ranges.


@pytest.mark.parametrize("factor", [1e-12, 1e-6, 1e4, 1e100])
def test_change_of_score_units_with_equivalent_regularization_preserves_fit(factor):
    scores = np.array([-3., -1., 0., 2., 4., 7.])
    labels = np.array([0, 0, 1, 0, 1, 1])
    base = platt_scaling(scores, labels, regularization=10)
    scaled = platt_scaling(scores * factor, labels, regularization=10 * factor**2)
    np.testing.assert_allclose(scaled(scores * factor), base(scores), atol=1e-10, rtol=0)
    assert scaled.params_[0] == pytest.approx(base.params_[0] / factor)
    assert scaled.params_[1] == pytest.approx(base.params_[1])


def test_tiny_score_spread_with_large_original_penalty_can_be_intercept_only():
    # The unregularized score effect is tiny compared with this original-unit
    # slope penalty; an effectively constant fit is correct in this case.
    scores = np.array([-3., -1., 0., 2., 4., 7.]) * 1e-100
    fitted = platt_scaling(scores, [0, 0, 1, 0, 1, 1], regularization=1e100)
    np.testing.assert_allclose(fitted(scores), .5, atol=1e-12)
    assert np.all(np.isfinite(fitted.params_))


def test_reported_optimizer_success_must_pass_objective_and_gradient_checks(monkeypatch):
    from types import SimpleNamespace
    from judgecal import calibration
    monkeypatch.setattr(calibration.optimize, "minimize", lambda *args, **kwargs:
                        SimpleNamespace(success=True, x=np.array([0., 0.]), message="success"))
    with pytest.raises(RuntimeError, match="first-order/objective"):
        platt_scaling([0, 1, 2, 3], [0, 1, 0, 1])


def test_large_finite_scores_avoid_training_overflow_and_improve_log_loss():
    scores = np.array([-1e300, -.5e300, .5e300, 1e300])
    labels = np.array([0, 1, 0, 1])
    with np.errstate(over="raise", invalid="raise"):
        fitted = platt_scaling(scores, labels)
        probabilities = fitted(scores)
        extrapolated = fitted([-np.finfo(float).max, np.finfo(float).max])
    assert np.all(np.isfinite(fitted.params_))
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities > 0) & (probabilities < 1))
    assert np.all(np.diff(probabilities) > 0)
    loss = -np.sum(labels * np.log(probabilities) + (1 - labels) * np.log1p(-probabilities))
    assert loss < 4 * np.log(2) - .1
    np.testing.assert_allclose(extrapolated, [0, 1])


def test_representable_differences_near_large_offset_remain_informative():
    offset = 1e300
    scores = np.array([-2., -1., 1., 2.]) * np.spacing(offset)
    shifted_scores = scores + offset
    assert len(np.unique(shifted_scores)) == 4
    labels = [0, 1, 0, 1]
    with np.errstate(over="raise", invalid="raise"):
        base = platt_scaling(scores, labels)
        shifted = platt_scaling(shifted_scores, labels)
    np.testing.assert_allclose(shifted(shifted_scores), base(scores), atol=1e-12, rtol=0)
    assert np.ptp(shifted(shifted_scores)) > .3
    assert np.all(np.isfinite(shifted.params_))


def test_constant_huge_score_is_intercept_only_even_when_prediction_difference_overflows():
    score = np.finfo(float).max
    with np.errstate(over="raise", invalid="raise"):
        fitted = platt_scaling([score] * 4, [0, 0, 0, 1])
        predicted = fitted([-score, 0, score])
    assert fitted.params_[0] == 0
    assert fitted.params_[1] == pytest.approx(np.log(1 / 3))
    np.testing.assert_allclose(predicted, [.25, .25, .25], atol=1e-15)
    assert fitted(score) == pytest.approx(.25)
    assert fitted([]).shape == (0,)


def test_separated_fit_saturates_at_finite_extrapolation_without_overflow_error():
    fitted = platt_scaling([-2, -1, 1, 2], [0, 0, 1, 1])
    with np.errstate(over="raise", invalid="raise"):
        predicted = fitted([-np.finfo(float).max, np.finfo(float).max])
    np.testing.assert_allclose(predicted, [0, 1])


@pytest.mark.parametrize("scores", [[np.inf], [-np.inf], [np.nan], [[1, 2]], ["bad"]])
def test_centered_predictor_still_rejects_invalid_inputs(scores):
    fitted = platt_scaling([1e8, 1e8 + 1, 1e8 + 2, 1e8 + 3], [0, 1, 0, 1])
    with pytest.raises(ValueError, match="prediction scores"):
        fitted(scores)
