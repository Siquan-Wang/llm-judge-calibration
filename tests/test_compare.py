import numpy as np
import pytest

from judgecal import compare_models, judge_confusion
from judgecal.calibration import JudgeConfusion
from judgecal.compare import paired_agreement_gap
from judgecal.data import make_synthetic


def test_target_b_correction_respects_asymmetric_confusion_orientation():
    conf = JudgeConfusion(.9, .6, .75, 100)
    labels = ["A"] * 58 + ["B"] * 42
    a = compare_models(labels, confusion=conf)
    b = compare_models(labels, target="B", confusion=conf)
    assert a["corrected_win_rate"] == pytest.approx(.36)
    assert b["corrected_win_rate"] == pytest.approx(.64)
    assert a["corrected_win_rate"] + b["corrected_win_rate"] == pytest.approx(1)
    reversed_conf = JudgeConfusion(.6, .9, .75, 100, positive="B", negative="A")
    assert compare_models(labels, confusion=reversed_conf)["corrected_win_rate"] == pytest.approx(.36)


def test_raw_keys_stay_raw_and_summary_confusion_has_no_corrected_ci():
    result = compare_models(["A"] * 70 + ["B"] * 30, confusion=JudgeConfusion(.9, .5, .7, 100))
    assert result["win_rate"] == result["raw_win_rate"] == .7
    assert result["ci"] == result["raw_ci"]
    assert result["significant"] == result["raw_significant"]
    assert result["corrected_win_rate"] == pytest.approx(.5)
    assert result["corrected_ci"] is None
    assert result["corrected_significant"] is None


def test_rg_rejects_ties_instead_of_silently_switching_estimands():
    conf = JudgeConfusion(.8, .8, .8, 20)
    assert compare_models(["A", "tie"])["win_rate"] == .75
    with pytest.raises(ValueError, match="binary"):
        compare_models(["A", "tie"], confusion=conf)
    with pytest.raises(ValueError, match="binary"):
        compare_models(["A", "B"], calibration_judge=["A", "tie"], calibration_human=["A", "B"])
    with pytest.raises(ValueError, match="dropped ties"):
        compare_models(["A", "B"], confusion=judge_confusion(["A", "B", "tie"], ["A", "B", "A"]))


def test_invalid_and_unidentified_confusion_never_turn_into_zero_estimates():
    with pytest.raises(ValueError):
        compare_models(["A", "B"], confusion=JudgeConfusion(1., np.nan, 1., 2))
    result = compare_models(["A", "B"], confusion=JudgeConfusion(.5, .5, .5, 100))
    assert np.isnan(result["corrected_win_rate"])
    assert result["corrected_ci"] is None
    assert result["correction_diagnostics"]["status"] == "unidentifiable"


def test_held_out_joint_correction_has_separate_uncertainty_and_recovers_truth():
    calibration = make_synthetic(n=2000, true_win_rate=.6, judge_sensitivity=.9, judge_specificity=.7, tie_rate=0, random_state=11)
    evaluation = make_synthetic(n=4000, true_win_rate=.6, judge_sensitivity=.9, judge_specificity=.7, tie_rate=0, random_state=12)
    result = compare_models(evaluation.judge_winner, calibration_judge=calibration.judge_winner,
                            calibration_human=calibration.human_winner, n_boot=300, random_state=7)
    assert result["corrected_win_rate"] == pytest.approx(.6, abs=.04)
    ci = result["corrected_ci"]
    assert ci is not None
    assert ci["low"] < .6 < ci["high"]
    assert ci["method"] == "joint-calibration-bootstrap"
    assert result["correction_diagnostics"]["valid_draws"] == 300
    assert result["correction_diagnostics"]["invalid_draws"] == 0


def test_joint_bootstrap_reflects_calibration_sample_uncertainty():
    human = np.array(["A"] * 100 + ["B"] * 100)
    judge = np.array(["A"] * 80 + ["B"] * 20 + ["B"] * 70 + ["A"] * 30)
    evaluation = ["A"] * 6000 + ["B"] * 4000
    small = compare_models(evaluation, calibration_judge=judge, calibration_human=human, n_boot=300, random_state=3)
    large = compare_models(evaluation, calibration_judge=np.tile(judge, 20), calibration_human=np.tile(human, 20), n_boot=300, random_state=3)
    width = lambda result: result["corrected_ci"]["high"] - result["corrected_ci"]["low"]
    assert width(small) > 1.5 * width(large)


def test_boundary_and_unstable_calibration_abstain_from_interval_claims():
    boundary = compare_models(["A", "B"], calibration_judge=["A", "B"], calibration_human=["A", "B"], n_boot=50)
    assert boundary["corrected_ci"] is None
    assert boundary["correction_diagnostics"]["status"] == "boundary-rate-bootstrap-unavailable"
    unstable = compare_models(["A", "B"] * 10,
                               calibration_judge=["A", "A", "B", "A", "B", "B"],
                               calibration_human=["A", "A", "A", "B", "B", "B"],
                               n_boot=100, random_state=0)
    assert unstable["corrected_ci"] is None
    assert unstable["correction_diagnostics"]["invalid_draws"] > 0
    assert unstable["correction_diagnostics"]["status"] == "unstable-bootstrap"


def test_paired_gap_interval_preserves_pairing_and_returns_gap_uncertainty():
    human = ["A"] * 20
    a = ["A"] * 10 + ["B"] * 10
    same = paired_agreement_gap(a, a, human, n_boot=100, random_state=0)
    opposite = paired_agreement_gap(a, a[::-1], human, n_boot=100, random_state=0)
    assert same["gap"] == opposite["gap"] == 0
    assert same["gap_ci"]["method"] == "hoeffding"
    assert opposite["gap_ci"]["method"] == "paired-bootstrap"
    assert opposite["gap_ci"]["low"] < 0 < opposite["gap_ci"]["high"]
    assert not same["gap_significant"]


def test_paired_gap_single_observation_and_input_validation():
    one = paired_agreement_gap(["A"], ["B"], ["A"], n_boot=10)
    assert one["gap"] == 1
    assert one["gap_ci"]["low"] < 0
    assert not one["gap_significant"]
    with pytest.raises(ValueError):
        paired_agreement_gap(["A"], ["B", "B"], ["A"])
    with pytest.raises(ValueError):
        compare_models(["A"], calibration_judge=["A"])
    with pytest.raises(ValueError):
        compare_models(["A"], confusion=JudgeConfusion(.8, .8, .8, 10), calibration_judge=["A"], calibration_human=["A"])


def test_one_raw_win_is_not_significant():
    result = compare_models(["A"], n_boot=20)
    assert result["ci"]["low"] < .5
    assert not result["significant"]
