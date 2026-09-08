import numpy as np
import pandas as pd
import pytest

from judgecal.benchmark import evaluate_label_budget, question_disjoint_split


def fixture_frame():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "question_id": np.repeat(np.arange(40), 2), "turn": np.tile([1, 2], 40),
        "model_a": "model1", "model_b": "model2",
        "human_winner": rng.choice(["A", "B", "tie"], 80),
        "gpt4_winner": rng.choice(["A", "B", "tie"], 80),
    })


def test_questions_not_rows_are_partitioned():
    frame = fixture_frame()
    a, u = question_disjoint_split(frame, 0.4, 8)
    assert set(a.question_id).isdisjoint(u.question_id)
    assert len(a) + len(u) == len(frame)
    assert a.question_id.nunique() == 16
    assert a.groupby("question_id").size().eq(2).all()
    pd.testing.assert_frame_equal(a, question_disjoint_split(frame, 0.4, 8)[0])


def test_hidden_human_labels_cannot_change_estimates():
    frame = fixture_frame()
    _, hidden = question_disjoint_split(frame, 0.4, 3)
    before, _ = evaluate_label_budget(frame, fractions=[0.4], seeds=[3])
    altered = frame.copy()
    altered.loc[hidden.index, "human_winner"] = "A"
    after, _ = evaluate_label_budget(altered, fractions=[0.4], seeds=[3])
    np.testing.assert_array_equal(before.estimate, after.estimate)
    assert not np.array_equal(before.heldout_human_reference, after.heldout_human_reference)


def test_every_pair_budget_and_seed_reported_and_reproducible():
    frame = fixture_frame()
    second = frame.assign(model_b="model3")
    frame = pd.concat([frame, second], ignore_index=True)
    a, summary = evaluate_label_budget(frame, fractions=[0.2, 0.6], seeds=[0, 1, 2])
    b, _ = evaluate_label_budget(frame.sample(frac=1, random_state=123), fractions=[0.2, 0.6], seeds=[0, 1, 2])
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 2 * 2 * 3 * 3
    assert len(summary) == 6
    assert len(a.attrs["split_manifest"]) == 12


def test_duplicate_annotations_are_not_independent_observations():
    frame = fixture_frame()
    with pytest.raises(ValueError, match="aggregate repeated"):
        evaluate_label_budget(pd.concat([frame, frame.iloc[:1]]))


@pytest.mark.parametrize("ids", [np.arange(40) / 10 + 0.1, [str(i) for i in range(40)], np.arange(40) - 1])
def test_manifest_rejects_noninteger_or_negative_ids(ids):
    frame = fixture_frame().assign(question_id=np.repeat(ids, 2))
    with pytest.raises(ValueError, match="nonnegative integers"):
        evaluate_label_budget(frame, fractions=[0.4], seeds=[0])


@pytest.mark.parametrize("fraction", [0, 1, np.nan, 0.01, 0.99])
def test_invalid_or_too_small_split(fraction):
    with pytest.raises(ValueError):
        question_disjoint_split(fixture_frame(), fraction)
