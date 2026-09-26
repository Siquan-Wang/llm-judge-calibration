"""Bounded numeric outcomes share PPI inference, without categorical semantics."""
import json
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from judgecal import PPIMeanResult, prediction_powered_mean, prediction_powered_win_rate


F_L = np.array([.2, .8])
Y_L = np.array([.4, .6])
F_U = np.array([.1, .3, .7, .9])


def test_fractional_outcome_hand_calculated_mean_and_variance():
    result = prediction_powered_mean(F_L, Y_L, F_U)
    # Audit residuals [.2,-.2] give variance-of-mean .04.
    # Prediction-only variance-of-mean is 1/30, total 11/150.
    assert isinstance(result, PPIMeanResult)
    assert result.point == pytest.approx(.5)
    assert result.prediction_mean == pytest.approx(.5)
    assert result.outcome_mean == pytest.approx(.5)
    assert result.residual_correction == pytest.approx(0)
    assert result.standard_error**2 == pytest.approx(11 / 150)
    assert result.n_labeled == 2 and result.n_unlabeled == 4
    assert result.estimand == "population mean of the bounded outcome"
    assert not hasattr(result, "target")
    assert not hasattr(result, "raw_rate")
    assert not hasattr(result, "human_only_rate")
    assert "human preference" not in " ".join(result.assumptions)
    assert json.loads(json.dumps(result.as_dict(), allow_nan=False))["interval"]["point"] == result.point


def test_fractional_auto_power_matches_analytic_minimizer():
    result = prediction_powered_mean(F_L, Y_L, F_U, power="auto")
    # Cov(mean(Y),mean(F_L))=.03; denominator=.09+1/30=37/300.
    # lambda=9/37; minimized variance=.01 - .03**2/(37/300)=1/370.
    assert result.selected_power == pytest.approx(9 / 37)
    assert result.standard_error**2 == pytest.approx(1 / 370)
    assert result.estimated_variance_ratio == pytest.approx(10 / 37)
    assert result.power_method == "auto-per-pool-sample-variance"


@pytest.mark.parametrize("power", [0, .3, 1, "auto"])
@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("target", ["A", "B"])
def test_categorical_reduction_preserves_identical_inference(power, clustered, target):
    judge = ["A", "tie", "B", "A"]
    human = ["tie", "A", "B", "B"]
    evaluation = ["A", "B", "tie", "A", "tie"]
    mapping = {target: 1., "B" if target == "A" else "A": 0., "tie": .5}
    numeric = [[mapping[label] for label in values] for values in (judge, human, evaluation)]
    kwargs = {}
    if clustered:
        kwargs = {"labeled_groups": [0, 0, 1, 2], "unlabeled_groups": [3, 3, 4, 4, 5]}
    categorical = prediction_powered_win_rate(judge, human, evaluation, target=target, power=power, **kwargs)
    generic = prediction_powered_mean(*numeric, power=power, **kwargs)
    assert generic.prediction_mean == categorical.raw_rate
    assert generic.outcome_mean == categorical.human_only_rate
    for field in (
        "point", "residual_correction", "standard_error", "interval", "method",
        "n_labeled", "n_unlabeled", "n_labeled_groups", "n_unlabeled_groups",
        "selected_power", "power_method", "estimated_variance_ratio", "references",
    ):
        assert getattr(generic, field) == getattr(categorical, field)


@pytest.mark.parametrize("power", [0, .35, 1, "auto"])
@pytest.mark.parametrize("clustered", [False, True])
def test_score_complement_reverses_estimate_and_interval(power, clustered):
    f = np.array([.2, .8, .6, .05])
    y = np.array([.4, .6, .8, .3])
    u = np.array([.1, .3, .7, .9, .6])
    kwargs = {}
    if clustered:
        kwargs = {"labeled_groups": [0, 0, 1, 2], "unlabeled_groups": [3, 3, 4, 4, 5]}
    result = prediction_powered_mean(f, y, u, power=power, **kwargs)
    reverse = prediction_powered_mean(1 - f, 1 - y, 1 - u, power=power, **kwargs)
    assert reverse.selected_power == pytest.approx(result.selected_power)
    assert reverse.point == pytest.approx(1 - result.point)
    assert reverse.residual_correction == pytest.approx(-result.residual_correction)
    assert reverse.standard_error == pytest.approx(result.standard_error)
    assert reverse.interval.low == pytest.approx(1 - result.interval.high)
    assert reverse.interval.high == pytest.approx(1 - result.interval.low)


def test_fractional_outcomes_are_not_thresholded_or_reweighted():
    # Numeric .2/.7/.8/.9 outcomes define their own row-weighted average.
    y = [.2, .7, .8, .9]
    result = prediction_powered_mean([.4] * 4, y, [.3, .6], power=0)
    assert result.outcome_mean == pytest.approx(.65)
    assert result.point == pytest.approx(.65)
    assert result.point != np.mean(np.array(y) > .5)
    assert result.estimated_variance_ratio == 1


def test_cluster_replication_keeps_independent_information_fixed():
    original = prediction_powered_mean(F_L, Y_L, F_U, power="auto")
    repeated = prediction_powered_mean(
        np.repeat(F_L, 5), np.repeat(Y_L, 5), np.repeat(F_U, 5), power="auto",
        labeled_groups=np.repeat(["l0", "l1"], 5),
        unlabeled_groups=np.repeat(["u0", "u1", "u2", "u3"], 5),
    )
    assert repeated.point == pytest.approx(original.point)
    assert repeated.selected_power == pytest.approx(original.selected_power)
    assert repeated.standard_error == pytest.approx(original.standard_error)
    assert repeated.n_labeled_groups == 2 and repeated.n_unlabeled_groups == 4


def test_unequal_cluster_sizes_keep_row_weighted_fractional_mean():
    result = prediction_powered_mean(
        [.2, .8, .4], [.1, .3, .8], [.4, .5, .7], power=0,
        labeled_groups=[0, 0, 1], unlabeled_groups=[2, 2, 3],
    )
    assert result.point == pytest.approx(.4)
    assert result.point != pytest.approx((.2 + .8) / 2)
    # Audit centered cluster sums [-.4,.4], G/(G-1)=2, n=3.
    assert result.standard_error**2 == pytest.approx(.64 / 9)


def test_extreme_correction_and_degenerate_diagnostics_are_not_clipped():
    result = prediction_powered_mean([0, 0], [1, 1], [1, 1])
    assert result.point == result.interval.low == result.interval.high == 2
    assert result.estimated_variance_ratio is None
    tuned = prediction_powered_mean([.4, .4], [.2, .8], [.4, .4], power="auto")
    assert tuned.selected_power == 0
    assert tuned.point == pytest.approx(.5)
    assert tuned.estimated_variance_ratio == 1


def test_supported_real_numeric_containers_keep_scores():
    result = prediction_powered_mean(
        pd.Series([Fraction(1, 5), Fraction(4, 5)]), (.4, .6), np.array(F_U, dtype=np.float32),
    )
    assert result.point == pytest.approx(.5)
    assert result.standard_error**2 == pytest.approx(11 / 150)


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("bad", [
    [], [.2], .5, "01", [[.2, .4]], [np.nan, .2], [np.inf, .2],
    [-.01, .2], [.2, 1.01], [None, .2], ["0.2", .4], [.2 + 0j, .4],
    [True, .4], np.array([False, True]), [pd.NA, .2],
    [10**1000, 0], [-10**1000, .4],
    np.ma.array([.2, .4], mask=[False, True]),
])
def test_invalid_numeric_inputs_are_rejected_without_coercion(position, bad):
    arrays = [F_L, Y_L, F_U]
    arrays[position] = bad
    with pytest.raises(ValueError):
        prediction_powered_mean(*arrays)


def test_misaligned_audit_arrays_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        prediction_powered_mean([.1, .2], [.3, .4, .5], [.2, .5])


@pytest.mark.parametrize("kwargs", [
    {"labeled_groups": [0, 1]},
    {"unlabeled_groups": [2, 3, 4, 5]},
    {"labeled_groups": [0, 1], "unlabeled_groups": [1, 2, 3, 4]},
    {"labeled_groups": [0, 0], "unlabeled_groups": [2, 3, 4, 5]},
    {"labeled_groups": [0, None], "unlabeled_groups": [2, 3, 4, 5]},
    {"labeled_groups": [0, 1], "unlabeled_groups": [2, np.inf, 4, 5]},
    {"labeled_groups": np.ma.array([0, 1], mask=[False, True]), "unlabeled_groups": [2, 3, 4, 5]},
])
def test_numeric_api_preserves_group_validation(kwargs):
    with pytest.raises(ValueError):
        prediction_powered_mean(F_L, Y_L, F_U, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"alpha": 0}, {"alpha": 1}, {"alpha": np.nan}, {"alpha": True},
    {"power": -.1}, {"power": 1.1}, {"power": np.nan}, {"power": "AUTO"},
    {"power": True}, {"power": None},
    {"alpha": 10**1000}, {"power": 10**1000},
    {"alpha": Fraction(1, 10**1000)}, {"alpha": Fraction(10**1000 - 1, 10**1000)},
])
def test_numeric_api_preserves_inference_option_validation(kwargs):
    with pytest.raises(ValueError):
        prediction_powered_mean(F_L, Y_L, F_U, **kwargs)


def test_smallest_positive_alpha_is_finite_for_numeric_scores():
    result = prediction_powered_mean(F_L, Y_L, F_U, alpha=np.nextafter(0., 1.))
    assert np.isfinite(result.interval.low)
    assert np.isfinite(result.interval.high)


def test_categorical_wrapper_also_preserves_masks_as_missing_data():
    with pytest.raises(ValueError, match="masked"):
        prediction_powered_win_rate(
            np.ma.array(["A", "B"], mask=[False, True]), ["A", "B"], ["A", "B"],
        )
    with pytest.raises(ValueError, match="masked"):
        prediction_powered_win_rate(
            ["A", "B"], ["A", "B"], ["A", "B"],
            labeled_groups=np.ma.array([0, 1], mask=[False, True]), unlabeled_groups=[2, 3],
        )
