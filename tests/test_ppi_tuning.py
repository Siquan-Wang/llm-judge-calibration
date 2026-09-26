"""Analytical checks for bounded PPI++ mean power tuning.

These check estimator algebra and invariances, not a guarantee of realized
MSE improvement or finite-sample coverage after data-dependent tuning.
"""
import json

import numpy as np
import pytest

from judgecal import prediction_powered_win_rate


JUDGE_L = ["A", "A", "B", "B"]
HUMAN_L = ["A", "tie", "B", "tie"]
JUDGE_U = ["A", "B", "tie", "tie"]


def test_explicit_power_one_matches_backward_compatible_default():
    default = prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U)
    fixed = prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, power=1.0)
    assert fixed == default
    assert fixed.method == "ppi-mean-normal-asymptotic"
    assert fixed.selected_power == 1.0
    assert fixed.power_method == "fixed"
    assert fixed.point == fixed.raw_rate + fixed.residual_correction


@pytest.mark.parametrize("clustered", [False, True])
def test_zero_power_recovers_human_only_point_and_variance(clustered):
    kwargs = {}
    if clustered:
        kwargs = {
            "labeled_groups": ["l0", "l0", "l1", "l1"],
            "unlabeled_groups": ["u0", "u0", "u1", "u1"],
        }
    result = prediction_powered_win_rate(
        JUDGE_L, HUMAN_L, JUDGE_U, power=0.0, **kwargs
    )
    assert result.point == result.human_only_rate == .5
    # Human scores [1,.5,0,.5]; cluster centered sums [.5,-.5].
    expected_variance = 1 / 16 if clustered else 1 / 24
    assert result.standard_error**2 == pytest.approx(expected_variance)
    assert result.estimated_variance_ratio == pytest.approx(1.0)
    assert result.selected_power == 0


def test_auto_matches_hand_calculated_covariance_minimizer_with_ties():
    result = prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, power="auto")
    # Cov(mean(Y_L),mean(F_L))=1/24. Prediction variances=1/12 + 1/24.
    # lambda=(1/24)/(1/8)=1/3; Q=1/24 -(1/24)**2/(1/8)=1/36.
    assert result.selected_power == pytest.approx(1 / 3)
    assert result.standard_error**2 == pytest.approx(1 / 36)
    assert result.estimated_variance_ratio == pytest.approx(2 / 3)
    assert result.point == .5
    assert result.power_method == "auto-per-pool-sample-variance"
    assert result.method == "ppi-plus-plus-mean-normal-asymptotic"
    assert "https://arxiv.org/abs/2311.01453" in result.references


def test_perfect_judge_tuning_combines_two_independent_equal_pools():
    labels = ["A", "B", "A", "B"]
    result = prediction_powered_win_rate(labels, labels, labels, power="auto")
    # Equal prediction variances in equal pools give oracle lambda=1/2.
    assert result.selected_power == pytest.approx(.5)
    assert result.standard_error**2 == pytest.approx(1 / 24)
    assert result.estimated_variance_ratio == pytest.approx(.5)


def test_nonunit_power_preserves_raw_residual_diagnostic():
    result = prediction_powered_win_rate(
        ["A", "A", "A", "B"], ["A", "B", "B", "B"], ["B"] * 4,
        power=.25,
    )
    assert result.raw_rate == 0
    assert result.residual_correction == -.5
    assert result.point == pytest.approx(.25 + .25 * (0 - .75))
    assert result.point != result.raw_rate + result.residual_correction


@pytest.mark.parametrize("human", [
    ["A", "B", "A", "B"],  # Zero covariance with the judge.
    ["B", "B", "A", "A"],  # Negative covariance with the judge.
])
def test_weak_or_negative_judge_selects_human_only_endpoint(human):
    result = prediction_powered_win_rate(JUDGE_L, human, JUDGE_U, power="auto")
    assert result.selected_power == 0
    assert result.point == result.human_only_rate
    assert result.estimated_variance_ratio == pytest.approx(1)


def test_zero_prediction_variance_has_explicit_human_only_fallback():
    result = prediction_powered_win_rate(
        ["A"] * 4, ["A", "B", "A", "B"], ["A"] * 4, power="auto"
    )
    assert result.selected_power == 0
    assert result.point == .5
    assert result.standard_error**2 == pytest.approx(1 / 12)
    assert result.estimated_variance_ratio == pytest.approx(1)


def test_auto_clips_unconstrained_minimizer_at_one():
    # Human scores spread twice as far as judge scores; an unconstrained
    # minimizer is >1 because the judge-only pool is large.
    result = prediction_powered_win_rate(
        ["tie", "B"], ["A", "B"], ["tie", "B"] * 50, power="auto"
    )
    assert result.selected_power == 1


@pytest.mark.parametrize("power", [0, .5, 1, "auto"])
def test_zero_human_variance_ratio_is_none_and_serializable(power):
    result = prediction_powered_win_rate(
        JUDGE_L, ["A"] * 4, JUDGE_U, power=power
    )
    assert result.estimated_variance_ratio is None
    serialized = json.loads(json.dumps(result.as_dict(), allow_nan=False))
    assert serialized["estimated_variance_ratio"] is None


def test_unequal_cluster_sandwich_minimizer_uses_paired_cluster_covariance():
    judge_l = ["A", "B", "A", "B", "B"]
    human_l = ["A", "tie", "A", "B", "tie"]
    judge_u = ["A", "tie", "B", "A", "B", "B"]
    labeled_groups = [0, 0, 1, 2, 2]
    unlabeled_groups = [3, 3, 4, 4, 4, 5]
    y = np.array([1, .5, 1, 0, .5])
    f_l = np.array([1, 0, 1, 0, 0])
    f_u = np.array([1, .5, 0, 1, 0, 0])

    def cluster_sums(values, groups):
        return np.array([
            (values - values.mean())[np.array(groups) == group].sum()
            for group in sorted(set(groups))
        ])

    # Independently assemble the three-cluster covariance and variance.
    sy = cluster_sums(y, labeled_groups)
    sl = cluster_sums(f_l, labeled_groups)
    su = cluster_sums(f_u, unlabeled_groups)
    covariance = 1.5 * (sy @ sl) / 25
    denominator = 1.5 * (sl @ sl) / 25 + 1.5 * (su @ su) / 36
    expected_power = np.clip(covariance / denominator, 0, 1)
    result = prediction_powered_win_rate(
        judge_l, human_l, judge_u,
        labeled_groups=labeled_groups, unlabeled_groups=unlabeled_groups,
        power="auto",
    )
    assert result.selected_power == pytest.approx(expected_power)
    assert result.point == pytest.approx(y.mean() + expected_power * (f_u.mean() - f_l.mean()))
    assert result.standard_error**2 == pytest.approx(
        1.5 * ((sy - expected_power * sl) @ (sy - expected_power * sl)) / 25
        + expected_power**2 * 1.5 * (su @ su) / 36
    )
    assert result.power_method == "auto-per-pool-cluster-sandwich"


@pytest.mark.parametrize("clustered", [False, True])
def test_auto_minimizes_its_estimated_variance_over_fixed_power_grid(clustered):
    kwargs = {}
    if clustered:
        kwargs = {
            "labeled_groups": ["l0", "l0", "l1", "l2"],
            "unlabeled_groups": ["u0", "u1", "u2", "u2"],
        }
    result = prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, power="auto", **kwargs)
    for power in np.linspace(0, 1, 31):
        fixed = prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, power=power, **kwargs)
        assert result.standard_error <= fixed.standard_error + 1e-14
    assert result.estimated_variance_ratio <= 1 + 1e-14


@pytest.mark.parametrize("clustered", [False, True])
def test_auto_target_reversal_preserves_power_and_reverses_interval(clustered):
    kwargs = {}
    if clustered:
        kwargs = {
            "labeled_groups": ["l0", "l0", "l1", "l2"],
            "unlabeled_groups": ["u0", "u1", "u2", "u2"],
        }
    a = prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, power="auto", **kwargs)
    b = prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, target="B", power="auto", **kwargs)
    assert a.selected_power == pytest.approx(b.selected_power)
    assert a.standard_error == pytest.approx(b.standard_error)
    assert a.estimated_variance_ratio == pytest.approx(b.estimated_variance_ratio)
    assert b.point == pytest.approx(1 - a.point)
    assert b.interval.low == pytest.approx(1 - a.interval.high)
    assert b.interval.high == pytest.approx(1 - a.interval.low)


def test_cluster_replication_and_group_id_relabeling_preserve_tuning():
    original = prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, power="auto")
    repeated = prediction_powered_win_rate(
        np.repeat(JUDGE_L, 5), np.repeat(HUMAN_L, 5), np.repeat(JUDGE_U, 5),
        labeled_groups=np.repeat(["a", "c", "b", "d"], 5),
        unlabeled_groups=np.repeat([101, 105, 103, 104], 5), power="auto",
    )
    assert repeated.selected_power == pytest.approx(original.selected_power)
    assert repeated.point == pytest.approx(original.point)
    assert repeated.standard_error == pytest.approx(original.standard_error)
    assert repeated.estimated_variance_ratio == pytest.approx(original.estimated_variance_ratio)


@pytest.mark.parametrize("power", ["auto", 0, .5])
def test_power_options_keep_disjoint_group_validation(power):
    with pytest.raises(ValueError, match="disjoint"):
        prediction_powered_win_rate(
            JUDGE_L, HUMAN_L, JUDGE_U,
            labeled_groups=[0, 0, 1, 1], unlabeled_groups=[1, 1, 2, 2], power=power,
        )


@pytest.mark.parametrize("power", [
    -.1, 1.1, np.inf, -np.inf, np.nan, True, np.bool_(False), None,
    "AUTO", "1.0", [], np.array([.5]), {},
])
def test_invalid_power_is_rejected(power):
    with pytest.raises(ValueError, match="power"):
        prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, power=power)
