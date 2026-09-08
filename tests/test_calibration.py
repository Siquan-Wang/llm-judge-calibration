import numpy as np
import pytest

from judgecal import (
    compare_models,
    isotonic_calibration,
    judge_confusion,
    platt_scaling,
    rogan_gladen_correction,
)
from judgecal.data import make_synthetic


def test_judge_confusion_exact():
    human = ["A", "A", "A", "A", "B", "B", "B", "B"]
    judge = ["A", "A", "A", "B", "B", "B", "A", "A"]
    conf = judge_confusion(judge, human)
    assert conf.sensitivity == pytest.approx(0.75)
    assert conf.specificity == pytest.approx(0.5)
    assert conf.n == 8


def test_rogan_gladen_inverts_measurement_model():
    p_true, se, sp = 0.6, 0.85, 0.8
    p_obs = se * p_true + (1 - sp) * (1 - p_true)
    assert rogan_gladen_correction(p_obs, se, sp) == pytest.approx(p_true)


def test_rogan_gladen_uninformative_judge_returns_nan():
    assert np.isnan(rogan_gladen_correction(0.5, 0.5, 0.5))


def test_correction_recovers_true_win_rate_end_to_end():
    # asymmetric judge errors: expected observed rate 0.95*0.6 + 0.4*0.4 = 0.73
    df = make_synthetic(
        n=20000,
        true_win_rate=0.6,
        judge_sensitivity=0.95,
        judge_specificity=0.60,
        tie_rate=0.0,
        random_state=42,
    )
    calibration = make_synthetic(
        n=4000, true_win_rate=0.6, judge_sensitivity=0.95,
        judge_specificity=0.60, tie_rate=0.0, random_state=43,
    )
    conf = judge_confusion(calibration["judge_winner"], calibration["human_winner"])
    result = compare_models(df["judge_winner"], confusion=conf, random_state=0)
    # raw judge win-rate is visibly biased away from 0.6
    assert abs(result["win_rate"] - 0.6) > 0.05
    # corrected estimate recovers the truth
    assert result["corrected_win_rate"] == pytest.approx(0.6, abs=0.03)
    assert result["corrected_ci"] is None  # summary rates do not supply uncertainty


def test_platt_scaling_recovers_logistic_params():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    p = 1 / (1 + np.exp(-(2.0 * x - 1.0)))
    y = (rng.random(5000) < p).astype(int)
    cal = platt_scaling(x, y)
    a, b = cal.params_
    assert a == pytest.approx(2.0, abs=0.2)
    assert b == pytest.approx(-1.0, abs=0.2)
    probs = cal(np.array([-2.0, 0.0, 2.0]))
    assert np.all((probs >= 0) & (probs <= 1))
    assert np.all(np.diff(probs) > 0)


def test_isotonic_is_monotone_and_fits_steps():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    cal = isotonic_calibration(x, y)
    fitted = cal(x)
    assert np.all(np.diff(fitted) >= -1e-12)
    assert cal(1.0) == pytest.approx(0.0)
    assert cal(6.0) == pytest.approx(1.0)


def test_isotonic_pools_equal_scores_and_is_permutation_invariant():
    assert isotonic_calibration([0, 0], [0, 1])(0) == pytest.approx(0.5)
    assert isotonic_calibration([0, 0], [1, 0])(0) == pytest.approx(0.5)
    x = np.array([0, 0, 1, 1, 2])
    y = np.array([1, 0, 0, 1, 1])
    order = np.array([4, 2, 0, 3, 1])
    grid = np.linspace(-1, 3, 17)
    assert isotonic_calibration(x, y)(grid) == pytest.approx(isotonic_calibration(x[order], y[order])(grid))


@pytest.mark.parametrize("rates", [(np.nan, .8, .8), (.5, np.nan, .8), (.5, .8, np.inf), (-.1, .8, .8), (.5, 1.1, .8)])
def test_rogan_gladen_rejects_invalid_rates_instead_of_clipping_nan(rates):
    with pytest.raises(ValueError):
        rogan_gladen_correction(*rates)


def test_rogan_gladen_exposes_unclipped_diagnostic():
    assert rogan_gladen_correction(.1, .8, .8) == 0
    assert rogan_gladen_correction(.1, .8, .8, clip=False) < 0


def test_confusion_validates_lengths_labels_and_reports_missing_classes():
    with pytest.raises(ValueError):
        judge_confusion(["A"], ["A", "B"])
    with pytest.raises(ValueError):
        judge_confusion(["unknown"], ["A"])
    with pytest.raises(ValueError):
        judge_confusion(["A"], ["A"], positive="A", negative="A")
    conf = judge_confusion(["A", "tie"], ["A", "B"])
    assert conf.n_dropped == 1
    assert conf.n_positive == 1
    assert conf.n_negative == 0
    assert np.isnan(conf.specificity)


@pytest.mark.parametrize("function", [platt_scaling, isotonic_calibration])
@pytest.mark.parametrize("x,y", [([], []), ([0], [0, 1]), ([[0]], [0]), ([np.nan], [0]), ([0], [2])])
def test_calibrators_reject_invalid_training_data(function, x, y):
    with pytest.raises(ValueError):
        function(x, y)


def test_platt_separation_is_finite_and_prediction_uses_stable_sigmoid():
    with pytest.raises(ValueError):
        platt_scaling([0, 1], [1, 1])
    cal = platt_scaling([-2, -1, 1, 2], [0, 0, 1, 1])
    assert np.all(np.isfinite(cal.params_))
    with np.errstate(over="raise"):
        assert cal([-1e6, 1e6]) == pytest.approx([0, 1])
    with pytest.raises(ValueError):
        cal([np.inf])


def test_platt_optimizer_failure_is_not_returned_as_a_fitted_model(monkeypatch):
    from types import SimpleNamespace
    from judgecal import calibration
    monkeypatch.setattr(calibration.optimize, "minimize", lambda *a, **k: SimpleNamespace(success=False, x=np.array([1., 0.]), message="failed"))
    with pytest.raises(RuntimeError, match="optimization failed"):
        platt_scaling([0, 1], [0, 1])


@pytest.mark.parametrize("regularization", [0, -1, np.nan, "bad", True])
def test_platt_rejects_invalid_regularization(regularization):
    with pytest.raises(ValueError):
        platt_scaling([0, 1], [0, 1], regularization=regularization)
