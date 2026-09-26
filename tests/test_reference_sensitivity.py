"""Distinct human references must share splits, costs and leakage boundaries."""
import numpy as np
import pandas as pd
import pytest

from judgecal.benchmark import fixed_evaluation_label_budget, reference_definition_sensitivity


def fixture_frame():
    rng = np.random.default_rng(61)
    cases = [(2, 1, 0, "A"), (1, 2, 0, "B"), (1, 1, 0, "tie"),
             (1, 0, 1, "tie"), (0, 1, 1, "tie"), (0, 0, 3, "tie"),
             (4, 0, 1, "A"), (0, 4, 1, "B")]
    votes = [cases[i] for i in rng.integers(0, len(cases), 80)]
    frame = pd.DataFrame({
        "question_id": np.repeat(np.arange(40), 2), "turn": np.tile([1, 2], 40),
        "model_a": "alpha", "model_b": "beta",
        "gpt4_winner": rng.choice(["A", "B", "tie"], 80),
        "human_winner": [case[3] for case in votes],
        "n_human_a": [case[0] for case in votes],
        "n_human_b": [case[1] for case in votes],
        "n_human_ties": [case[2] for case in votes],
    })
    frame["n_human_votes"] = frame[["n_human_a", "n_human_b", "n_human_ties"]].sum(axis=1)
    return frame


def test_plurality_numerics_are_exactly_preserved_and_input_is_unchanged():
    frame = fixture_frame()
    original = frame.copy(deep=True)
    baseline, baseline_summary = fixed_evaluation_label_budget(frame, seeds=[0, 3])
    trials, summary = reference_definition_sensitivity(frame, seeds=[0, 3])
    plurality = trials.loc[trials.reference_definition == "plurality", baseline.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline, plurality)
    pd.testing.assert_frame_equal(
        baseline_summary,
        summary.loc[summary.reference_definition == "plurality", baseline_summary.columns].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(frame, original)
    assert len(trials) == 2 * len(baseline)
    assert len(summary) == 2 * len(baseline_summary)
    assert set(trials.reference_definition) == {"plurality", "mean_vote"}


def test_splits_costs_and_raw_judge_are_identical_across_definitions():
    trials, _ = reference_definition_sensitivity(fixture_frame(), seeds=[0, 3])
    keys = ["model_a", "model_b", "labeled_fraction", "seed", "method"]
    majority = trials[trials.reference_definition == "plurality"].set_index(keys)
    mean_vote = trials[trials.reference_definition == "mean_vote"].set_index(keys)
    for column in ("split_sha256", "n_labeled", "n_evaluation", "n_unused", "labeled_questions",
                   "evaluation_questions", "unused_questions", "human_comparisons_used", "human_votes_used"):
        pd.testing.assert_series_equal(majority[column], mean_vote[column])
    np.testing.assert_array_equal(majority.xs("raw_judge", level="method").estimate,
                                  mean_vote.xs("raw_judge", level="method").estimate)
    assert all("heldout_human_reference" not in m for m in trials.attrs["split_manifest"])
    for _, group in trials.groupby(["reference_definition", "seed"]):
        assert group.heldout_human_reference.nunique() == 1
        assert group.loc[group.method == "raw_judge", "estimate"].nunique() == 1


def test_mean_vote_reference_weights_comparisons_equally_not_individual_votes():
    frame = fixture_frame()
    trials, _ = reference_definition_sensitivity(frame, fractions=[.4], seeds=[4])
    split = trials.attrs["split_manifest"][0]
    audit = frame[frame.question_id.isin(split["labeled_question_ids"])]
    evaluation = frame[frame.question_id.isin(split["evaluation_question_ids"])]
    expected_audit = ((audit.n_human_a + .5 * audit.n_human_ties) / audit.n_human_votes).mean()
    expected_target = ((evaluation.n_human_a + .5 * evaluation.n_human_ties) / evaluation.n_human_votes).mean()
    pooled_votes = (evaluation.n_human_a.sum() + .5 * evaluation.n_human_ties.sum()) / evaluation.n_human_votes.sum()
    mean_vote = trials[trials.reference_definition == "mean_vote"]
    assert mean_vote.heldout_human_reference.eq(expected_target).all()
    assert mean_vote.loc[mean_vote.method == "human_only", "estimate"].iloc[0] == pytest.approx(expected_audit)
    assert not np.isclose(expected_target, pooled_votes)
    assert any((frame.n_human_a == 1) & (frame.n_human_ties == 1) & (frame.n_human_b == 0))


def set_unanimous(frame, ids, target):
    changed = frame.copy()
    mask = changed.question_id.isin(ids)
    changed.loc[mask, ["n_human_a", "n_human_b", "n_human_ties"]] = 0
    changed.loc[mask, "n_human_a" if target == "A" else "n_human_b"] = changed.loc[mask, "n_human_votes"]
    changed.loc[mask, "human_winner"] = target
    return changed


def test_hidden_vote_counts_and_plurality_labels_cannot_change_inference():
    frame = fixture_frame()
    before, _ = reference_definition_sensitivity(frame, seeds=[4])
    hidden = before.attrs["split_manifest"][0]["evaluation_question_ids"]
    modified = set_unanimous(frame, hidden, "A")
    after, _ = reference_definition_sensitivity(modified, seeds=[4])
    for column in ("estimate", "selected_power", "interval_low", "interval_high", "standard_error",
                   "estimated_variance_ratio", "human_votes_used", "split_sha256"):
        pd.testing.assert_series_equal(before[column], after[column])
    assert not np.array_equal(before.heldout_human_reference, after.heldout_human_reference)
    assert before.attrs["split_manifest"] == after.attrs["split_manifest"]


def test_unused_votes_labels_and_predictions_do_not_change_results():
    frame = fixture_frame()
    before, before_summary = reference_definition_sensitivity(frame, fractions=[.2], seeds=[4])
    unused = before.attrs["split_manifest"][0]["unused_question_ids"]
    modified = set_unanimous(frame, unused, "B")
    modified.loc[modified.question_id.isin(unused), "gpt4_winner"] = "A"
    after, after_summary = reference_definition_sensitivity(modified, fractions=[.2], seeds=[4])
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(before_summary, after_summary)


def test_unanimous_votes_make_the_two_references_and_estimators_identical():
    frame = fixture_frame()
    a = frame.human_winner.eq("A")
    b = frame.human_winner.eq("B")
    frame["n_human_a"] = np.where(a, frame.n_human_votes, 0)
    frame["n_human_b"] = np.where(b, frame.n_human_votes, 0)
    frame["n_human_ties"] = np.where(~a & ~b, frame.n_human_votes, 0)
    trials, _ = reference_definition_sensitivity(frame, seeds=[0])
    majority = trials[trials.reference_definition == "plurality"].reset_index(drop=True)
    mean_vote = trials[trials.reference_definition == "mean_vote"].reset_index(drop=True)
    for column in ("estimate", "heldout_human_reference", "selected_power", "interval_low", "interval_high"):
        pd.testing.assert_series_equal(majority[column], mean_vote[column])


def test_row_permutation_and_multiple_pairs_preserve_results():
    frame = fixture_frame()
    frame = pd.concat([frame, frame.assign(model_b="gamma")], ignore_index=True)
    before, before_summary = reference_definition_sensitivity(frame, seeds=[1])
    after, after_summary = reference_definition_sensitivity(frame.sample(frac=1, random_state=2), seeds=[1])
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(before_summary, after_summary)
    assert len(before) == 2 * 2 * 3 * 4


@pytest.mark.parametrize("column,value", [
    ("n_human_a", -1), ("n_human_b", 1.5), ("n_human_ties", True),
    ("n_human_a", np.inf), ("n_human_b", np.nan), ("n_human_a", "1"),
    ("n_human_votes", 0), ("n_human_votes", 999), ("n_human_votes", 2.0),
    ("human_winner", "unknown"), ("human_winner", pd.NA),
])
def test_invalid_counts_or_labels_are_rejected(column, value):
    frame = fixture_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        reference_definition_sensitivity(frame, seeds=[0])


def test_inconsistent_plurality_is_rejected_instead_of_changing_the_reference():
    frame = fixture_frame()
    frame.loc[0, "human_winner"] = "B" if frame.loc[0, "human_winner"] != "B" else "A"
    with pytest.raises(ValueError, match="plurality implied"):
        reference_definition_sensitivity(frame, seeds=[0])


@pytest.mark.parametrize("column", ["n_human_a", "n_human_b", "n_human_ties", "n_human_votes"])
def test_missing_vote_counts_fail_closed(column):
    with pytest.raises(ValueError, match="missing reference-definition"):
        reference_definition_sensitivity(fixture_frame().drop(columns=column), seeds=[0])
