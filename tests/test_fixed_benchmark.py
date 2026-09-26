import numpy as np
import pandas as pd
import pytest

from judgecal.benchmark import fixed_evaluation_label_budget


def fixture_frame():
    """Two correlated turns per question, with recorded annotation costs."""
    rng = np.random.default_rng(21)
    return pd.DataFrame({
        "question_id": np.repeat(np.arange(40), 2),
        "turn": np.tile([1, 2], 40),
        "model_a": "alpha", "model_b": "beta",
        "human_winner": rng.choice(["A", "B", "tie"], 80),
        "gpt4_winner": rng.choice(["A", "B", "tie"], 80),
        "n_human_votes": rng.integers(1, 5, 80),
    })


def test_fixed_target_nested_audits_and_complete_pool_manifest():
    frame = fixture_frame()
    trials, summary = fixed_evaluation_label_budget(frame, seeds=[0, 1])
    assert len(trials) == 3 * 2 * 4
    assert len(summary) == 3 * 4
    assert set(trials.method) == {"raw_judge", "human_only", "ppi", "ppi_tuned"}
    manifests = trials.attrs["split_manifest"]
    assert len(manifests) == 6
    for seed in (0, 1):
        splits = sorted((x for x in manifests if x["seed"] == seed),
                        key=lambda x: x["labeled_fraction"])
        previous_audit = set()
        fixed_evaluation = set(splits[0]["evaluation_question_ids"])
        for split in splits:
            audit = set(split["labeled_question_ids"])
            evaluation = set(split["evaluation_question_ids"])
            unused = set(split["unused_question_ids"])
            assert previous_audit <= audit
            assert evaluation == fixed_evaluation
            assert len(evaluation) == 10
            assert not audit & evaluation
            assert not audit & unused
            assert not evaluation & unused
            assert audit | evaluation | unused == set(frame.question_id)
            assert split["n_labeled"] == 2 * split["labeled_questions"]
            assert split["n_evaluation"] == 2 * split["evaluation_questions"]
            assert split["n_unused"] == 2 * split["unused_questions"]
            assert split["n_labeled"] + split["n_evaluation"] + split["n_unused"] == len(frame)
            previous_audit = audit
        each_seed = trials.loc[trials.seed == seed]
        assert each_seed.heldout_human_reference.nunique() == 1
        assert each_seed.loc[each_seed.method == "raw_judge", "estimate"].nunique() == 1


def test_hidden_evaluation_labels_never_change_inference_or_tuning():
    frame = fixture_frame()
    before, _ = fixed_evaluation_label_budget(frame, seeds=[7])
    hidden = before.attrs["split_manifest"][0]["evaluation_question_ids"]
    modified = frame.copy()
    modified.loc[modified.question_id.isin(hidden), "human_winner"] = "A"
    after, _ = fixed_evaluation_label_budget(modified, seeds=[7])
    for column in ("estimate", "selected_power", "interval_low", "interval_high",
                   "standard_error", "estimated_variance_ratio"):
        pd.testing.assert_series_equal(before[column], after[column])
    assert not np.array_equal(before.heldout_human_reference, after.heldout_human_reference)


def test_unused_labels_and_predictions_never_change_any_result():
    frame = fixture_frame()
    before, before_summary = fixed_evaluation_label_budget(frame, fractions=[0.2], seeds=[7])
    unused = before.attrs["split_manifest"][0]["unused_question_ids"]
    assert unused
    modified = frame.copy()
    modified.loc[modified.question_id.isin(unused), ["human_winner", "gpt4_winner"]] = "B"
    after, after_summary = fixed_evaluation_label_budget(modified, fractions=[0.2], seeds=[7])
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(before_summary, after_summary)


def test_all_methods_use_same_audit_and_record_annotation_costs():
    frame = fixture_frame()
    trials, _ = fixed_evaluation_label_budget(frame, fractions=[0.4], seeds=[2])
    split = trials.attrs["split_manifest"][0]
    audit = frame.loc[frame.question_id.isin(split["labeled_question_ids"])]
    assert trials.split_sha256.nunique() == 1
    raw = trials.loc[trials.method == "raw_judge"].iloc[0]
    assert raw.human_comparisons_used == raw.human_votes_used == 0
    assert pd.isna(raw.interval_width)
    assert pd.isna(raw.selected_power)
    corrected = trials.loc[trials.method != "raw_judge"]
    assert corrected.human_comparisons_used.eq(len(audit)).all()
    assert corrected.human_votes_used.eq(audit.n_human_votes.sum()).all()
    np.testing.assert_allclose(corrected.interval_width,
                               corrected.interval_high - corrected.interval_low)
    assert corrected.standard_error.ge(0).all()
    assert trials.loc[trials.method == "human_only", "selected_power"].iloc[0] == 0
    assert trials.loc[trials.method == "ppi", "selected_power"].iloc[0] == 1
    assert trials.loc[trials.method == "ppi_tuned", "selected_power"].between(0, 1).all()


def test_missing_vote_counts_are_reported_as_unknown_rather_than_one_vote():
    trials, _ = fixed_evaluation_label_budget(
        fixture_frame().drop(columns="n_human_votes"), fractions=[0.2], seeds=[0],
    )
    assert trials.loc[trials.method != "raw_judge", "human_votes_used"].isna().all()
    assert trials.loc[trials.method == "raw_judge", "human_votes_used"].eq(0).all()


def test_reproducibility_is_independent_of_input_order_and_pairs_are_complete():
    frame = fixture_frame()
    frame = pd.concat([frame, frame.assign(model_b="gamma")], ignore_index=True)
    first, summary = fixed_evaluation_label_budget(frame, seeds=[2, 8])
    second, second_summary = fixed_evaluation_label_budget(
        frame.sample(frac=1, random_state=19), seeds=[2, 8],
    )
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(summary, second_summary)
    assert first.attrs["split_manifest"] == second.attrs["split_manifest"]
    assert len(first) == 2 * 3 * 2 * 4
    assert summary.trials.eq(4).all()


@pytest.mark.parametrize("kwargs", [
    {"fractions": []}, {"fractions": [0.2, 0.2]}, {"fractions": [True]},
    {"fractions": ["0.2"]}, {"fractions": [np.nan]}, {"fractions": [0.8]},
    {"fractions": [0.01]}, {"evaluation_fraction": 0.01},
    {"evaluation_fraction": 0}, {"evaluation_fraction": np.inf},
    {"evaluation_fraction": True}, {"seeds": []}, {"seeds": [2, 2]},
    {"seeds": [True]}, {"seeds": [-1]}, {"seeds": [1.5]},
    {"alpha": 0}, {"alpha": True}, {"min_pair_questions": 2},
    {"min_pair_questions": True}, {"min_pair_questions": 41},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        fixed_evaluation_label_budget(fixture_frame(), **kwargs)


@pytest.mark.parametrize("column,value", [
    ("question_id", -1), ("question_id", "1"), ("model_a", " beta"),
    ("model_b", "alpha"), ("human_winner", "unknown"),
    ("gpt4_winner", None), ("n_human_votes", 0),
    ("n_human_votes", 1.5), ("n_human_votes", True), ("turn", None),
])
def test_invalid_data_rejected(column, value):
    frame = fixture_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        fixed_evaluation_label_budget(frame, seeds=[0])


def test_repeated_votes_cannot_be_mistaken_for_independent_comparisons():
    frame = fixture_frame()
    with pytest.raises(ValueError, match="aggregate repeated"):
        fixed_evaluation_label_budget(pd.concat([frame, frame.iloc[:1]]), seeds=[0])
