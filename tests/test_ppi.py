"""Analytical and seeded population checks for the fixed-weight PPI mean."""
import json

import numpy as np
import pytest
from scipy import stats

from judgecal.intervals import Interval
from judgecal.ppi import prediction_powered_win_rate


JUDGE_L = ["A", "B", "A", "B"]
HUMAN_L = ["A", "A", "B", "B"]
JUDGE_U = ["A", "A", "B", "tie"]


def test_hand_calculated_iid_estimate_variance_and_interval():
    result = prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U)
    # Residuals [0, 1, -1, 0]: variance of mean = 1/6.
    # Judge-only [1, 1, 0, .5]: variance of mean = 11/192.
    expected_se = np.sqrt(43 / 192)
    assert result.point == pytest.approx(5 / 8)
    assert result.raw_rate == pytest.approx(5 / 8)
    assert result.human_only_rate == pytest.approx(1 / 2)
    assert result.residual_correction == 0
    assert result.standard_error == pytest.approx(expected_se)
    assert isinstance(result.interval, Interval)
    assert result.interval.point == result.point
    assert result.interval.low == pytest.approx(5 / 8 - stats.norm.ppf(.975) * expected_se)
    assert result.interval.high == pytest.approx(5 / 8 + stats.norm.ppf(.975) * expected_se)
    assert result.n_labeled == result.n_unlabeled == 4
    assert result.n_labeled_groups is result.n_unlabeled_groups is None
    assert result.variance_method == "iid-sample-variance"
    assert "asymptotic" in result.method
    assert json.loads(json.dumps(result.as_dict()))["interval"]["point"] == result.point


def test_correction_and_bounds_are_not_clipped():
    result = prediction_powered_win_rate(["B", "B"], ["A", "A"], ["A", "A"])
    assert result.point == result.interval.low == result.interval.high == 2.0
    assert result.raw_rate == result.human_only_rate == result.residual_correction == 1.0
    assert result.standard_error == 0


def test_ties_are_half_wins_on_both_pools():
    result = prediction_powered_win_rate(
        ["A", "tie", "B"], ["tie", "tie", "A"], ["tie", "B", "tie"]
    )
    assert result.raw_rate == pytest.approx(1 / 3)
    assert result.human_only_rate == pytest.approx(2 / 3)
    assert result.residual_correction == pytest.approx(1 / 6)
    assert result.point == pytest.approx(1 / 2)


@pytest.mark.parametrize("clustered", [False, True])
def test_target_reversal_reverses_point_and_interval(clustered):
    groups = {}
    if clustered:
        groups = {"labeled_groups": ["l0", "l0", "l1", "l1"],
                  "unlabeled_groups": ["u0", "u1", "u1", "u1"]}
    a = prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, **groups)
    b = prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, target="B", **groups)
    assert b.point == pytest.approx(1 - a.point)
    assert b.raw_rate == pytest.approx(1 - a.raw_rate)
    assert b.human_only_rate == pytest.approx(1 - a.human_only_rate)
    assert b.residual_correction == pytest.approx(-a.residual_correction)
    assert b.standard_error == pytest.approx(a.standard_error)
    assert b.interval.low == pytest.approx(1 - a.interval.high)
    assert b.interval.high == pytest.approx(1 - a.interval.low)


def test_cluster_replication_does_not_create_independent_information():
    original = prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U)
    repeated = [np.repeat(values, 5) for values in (JUDGE_L, HUMAN_L, JUDGE_U)]
    clustered = prediction_powered_win_rate(
        *repeated,
        labeled_groups=np.repeat(["l0", "l1", "l2", "l3"], 5),
        unlabeled_groups=np.repeat(["u0", "u1", "u2", "u3"], 5),
    )
    iid = prediction_powered_win_rate(*repeated)
    assert clustered.point == original.point == iid.point
    assert clustered.standard_error == pytest.approx(original.standard_error)
    assert iid.standard_error < clustered.standard_error
    assert clustered.n_labeled == clustered.n_unlabeled == 20
    assert clustered.n_labeled_groups == clustered.n_unlabeled_groups == 4
    assert clustered.variance_method == "one-way-cluster-robust"


def test_hand_calculated_unequal_cluster_sizes_keep_observation_weighting():
    result = prediction_powered_win_rate(
        ["B", "B", "A"], ["A", "B", "B"], ["A", "A", "B"],
        labeled_groups=["l0", "l0", "l1"],
        unlabeled_groups=["u0", "u0", "u1"],
    )
    # Audit residual cluster sums: [1,-1], variance = 4/9.
    # Judge-only centered cluster sums: [2/3,-2/3], variance = 16/81.
    assert result.point == pytest.approx(2 / 3)
    assert result.standard_error == pytest.approx(np.sqrt(52 / 81))


def test_higher_confidence_produces_wider_interval():
    a = prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, alpha=.01)
    b = prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, alpha=.1)
    assert a.interval.low < b.interval.low < b.interval.high < a.interval.high


@pytest.mark.parametrize("constant", [False, True])
def test_smallest_positive_alpha_does_not_underflow(constant):
    inputs = [["A", "A"]] * 3 if constant else [JUDGE_L, HUMAN_L, JUDGE_U]
    result = prediction_powered_win_rate(*inputs, alpha=np.nextafter(0.0, 1.0))
    assert np.isfinite(result.interval.low)
    assert np.isfinite(result.interval.high)
    assert result.interval.low <= result.point <= result.interval.high


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, np.nan, np.inf, "0.05", None, True])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, alpha=alpha)


@pytest.mark.parametrize("target", ["tie", "a", "C", 1, None, ["A"]])
def test_invalid_target_is_rejected(target):
    with pytest.raises(ValueError, match="target"):
        prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, target=target)


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("bad", [[], ["A"], [["A", "B"], ["A", "B"]],
                                  ["A", "unknown"], [1, 0], ["A", None],
                                  ["A", np.nan], "AB"])
def test_invalid_label_arrays_are_rejected(position, bad):
    inputs = [JUDGE_L, HUMAN_L, JUDGE_U]
    inputs[position] = bad
    with pytest.raises(ValueError):
        prediction_powered_win_rate(*inputs)


def test_misaligned_audit_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        prediction_powered_win_rate(JUDGE_L, HUMAN_L[:2], JUDGE_U)


@pytest.mark.parametrize("kwargs", [
    {"labeled_groups": ["l0", "l1", "l2", "l3"]},
    {"unlabeled_groups": ["u0", "u1", "u2", "u3"]},
])
def test_both_group_arrays_are_required(kwargs):
    with pytest.raises(ValueError, match="supplied together"):
        prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, **kwargs)


@pytest.mark.parametrize("bad", [
    [], ["l0", "l1"], ["l0"] * 4,
    [["l0", "l1"], ["l2", "l3"]],
    ["l0", "l1", None, "l3"], ["l0", "l1", np.nan, "l3"],
    ["l0", "l1", np.inf, "l3"], ["l0", "l1", {}, "l3"],
])
@pytest.mark.parametrize("side", ["labeled_groups", "unlabeled_groups"])
def test_invalid_group_arrays_are_rejected(bad, side):
    groups = {"labeled_groups": ["l0", "l1", "l2", "l3"],
              "unlabeled_groups": ["u0", "u1", "u2", "u3"]}
    groups[side] = bad
    with pytest.raises(ValueError):
        prediction_powered_win_rate(JUDGE_L, HUMAN_L, JUDGE_U, **groups)


def test_groups_overlapping_between_pools_are_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        prediction_powered_win_rate(
            JUDGE_L, HUMAN_L, JUDGE_U,
            labeled_groups=[1, 1, 2, 2], unlabeled_groups=[2, 2, 3, 3],
        )


def test_frozen_monte_carlo_corrects_judge_bias_with_reasonable_coverage():
    # Independent draws from a known population, not a finite-corpus reference.
    # P(Y=A)=.4; sensitivity=.9, specificity=.65 -> P(F=A)=.57.
    rng = np.random.default_rng(20260907)
    true_rate = .4
    judge_rate = .57
    estimates, raw_rates, covered = [], [], []
    for _ in range(800):
        human = rng.random(300) < true_rate
        judge = rng.random(300) < np.where(human, .9, .35)
        unlabeled = rng.random(1200) < judge_rate
        result = prediction_powered_win_rate(
            np.where(judge, "A", "B"), np.where(human, "A", "B"),
            np.where(unlabeled, "A", "B"),
        )
        estimates.append(result.point)
        raw_rates.append(result.raw_rate)
        covered.append(result.interval.low <= true_rate <= result.interval.high)
    estimates = np.asarray(estimates)
    raw_rates = np.asarray(raw_rates)
    # Frozen tolerances are sampling sanity checks, not a coverage guarantee.
    assert abs(estimates.mean() - true_rate) < .008
    assert .16 < raw_rates.mean() - true_rate < .18
    assert .91 < np.mean(covered) < .985
    assert np.mean((estimates - true_rate)**2) < np.mean((raw_rates - true_rate)**2) / 9
