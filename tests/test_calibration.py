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
    conf = judge_confusion(df["judge_winner"], df["human_winner"])
    result = compare_models(df["judge_winner"], confusion=conf, random_state=0)
    # raw judge win-rate is visibly biased away from 0.6
    assert abs(result["win_rate"] - 0.6) > 0.05
    # corrected estimate recovers the truth
    assert result["corrected_win_rate"] == pytest.approx(0.6, abs=0.02)


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
