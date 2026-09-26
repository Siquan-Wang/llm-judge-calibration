"""Shared cached-judge audit splits, label isolation and explicit invalid parses."""
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from judgecal.judge_audit import judge_accuracy_audit
from judgecal.ppi import prediction_powered_mean


def panel():
    rng = np.random.default_rng(518)
    rows = []
    for group in range(40):
        # Unequal clusters catch accidental instruction rather than comparison weighting.
        for copy in range(2 if group % 3 == 0 else 1):
            gold = 1 + ((group + copy) % 2)
            subset = "Natural" if group < 20 else ("Neighbor", "GPTInst", "GPTOut", "Manual")[group % 4]
            for judge in ("judge_a", "judge_b"):
                rows.append({"sample_id": f"s{group:02d}_{copy}", "instruction_id": f"i{group:02d}",
                             "subset": subset, "judge": judge, "reference_label": gold,
                             "forward_prediction": int(rng.choice([0, 1, 2], p=[.1, .4, .5])),
                             "reverse_prediction": int(rng.choice([0, 1, 2], p=[.1, .5, .4]))})
    return pd.DataFrame(rows)


def run(frame=None, **kwargs):
    return judge_accuracy_audit(panel() if frame is None else frame, seeds=[7], **kwargs)


def test_shared_complete_instruction_splits_nested_audits_and_fixed_targets():
    frame = panel()
    trials, summary = judge_accuracy_audit(frame, seeds=[7, 13])
    assert len(trials) == 3 * 2 * 3 * 2 * 4
    assert len(summary) == 3 * 2 * 3 * 4
    assert summary.trials.eq(2).all()
    assert not any("coverage" in name or "mcse" in name for name in summary)
    manifests = trials.attrs["split_manifest"]
    assert len(manifests) == 3 * 2 * 3
    for cohort in ("All", "Natural", "Adversarial"):
        cohort_frame = (frame if cohort == "All" else
                        frame.loc[frame.subset.eq("Natural") == (cohort == "Natural")])
        for seed in (7, 13):
            splits = [m for m in manifests if m["cohort"] == cohort and m["seed"] == seed]
            previous = set()
            evaluation = splits[0]["evaluation_instruction_ids"]
            for split in splits:
                audit, heldout, unused = [set(split[f"{name}_instruction_ids"])
                                          for name in ("labeled", "evaluation", "unused")]
                assert previous <= audit
                assert split["evaluation_instruction_ids"] == evaluation
                assert not audit & heldout and not audit & unused and not heldout & unused
                assert audit | heldout | unused == set(cohort_frame.instruction_id)
                assert len(audit) == int(len(audit | heldout | unused) * split["labeled_fraction"])
                for name in ("labeled", "evaluation", "unused"):
                    expected = set(cohort_frame.loc[
                        cohort_frame.instruction_id.isin(split[f"{name}_instruction_ids"]), "sample_id"])
                    assert expected == set(split[f"{name}_sample_ids"])
                    assert split[f"n_{name}"] == len(expected)
                selected = trials.loc[(trials.cohort == cohort) & (trials.seed == seed)
                                      & (trials.labeled_fraction == split["labeled_fraction"])]
                assert selected.split_sha256.nunique() == 1
                assert selected.split_sha256.iloc[0] == split["split_sha256"]
                assert not any("reference" in key or "prediction" in key for key in split)
                previous = audit
            for _, rows in trials.loc[(trials.cohort == cohort) & (trials.seed == seed)].groupby("judge"):
                assert rows.heldout_accuracy_reference.nunique() == 1
                assert rows.loc[rows.method == "raw_proxy", "estimate"].nunique() == 1


def test_hidden_gold_never_changes_proxy_inference_or_tuning():
    frame = panel()
    before, _ = run(frame, cohorts=["All"])
    hidden = before.attrs["split_manifest"][0]["evaluation_sample_ids"]
    mutated = frame.copy()
    mask = mutated.sample_id.isin(hidden)
    mutated.loc[mask, "reference_label"] = 3 - mutated.loc[mask, "reference_label"]
    after, _ = run(mutated, cohorts=["All"])
    columns = ["estimate", "selected_power", "interval_low", "interval_high", "interval_width",
               "standard_error", "estimated_variance_ratio", "human_labels_used", "cached_judgments_used"]
    pd.testing.assert_frame_equal(before[columns], after[columns])
    assert before.attrs["split_manifest"] == after.attrs["split_manifest"]
    assert not before.heldout_accuracy_reference.equals(after.heldout_accuracy_reference)


def test_unused_gold_and_cached_predictions_never_change_results():
    frame = panel()
    before, summary = run(frame, fractions=[.2], cohorts=["All"])
    unused = before.attrs["split_manifest"][0]["unused_sample_ids"]
    mutated = frame.copy()
    mask = mutated.sample_id.isin(unused)
    mutated.loc[mask, "reference_label"] = 3 - mutated.loc[mask, "reference_label"]
    mutated.loc[mask, ["forward_prediction", "reverse_prediction"]] = 0
    after, after_summary = run(mutated, fractions=[.2], cohorts=["All"])
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(summary, after_summary)


def test_direct_grouped_inference_costs_and_observation_weighting():
    frame = panel()
    trials, _ = run(frame, fractions=[.4], cohorts=["All"])
    manifest = trials.attrs["split_manifest"][0]
    for judge, rows in trials.groupby("judge"):
        source = frame.loc[frame.judge == judge].sort_values(["instruction_id", "sample_id"])
        audit = source.loc[source.sample_id.isin(manifest["labeled_sample_ids"])]
        evaluation = source.loc[source.sample_id.isin(manifest["evaluation_sample_ids"])]
        proxy = lambda data: ((data.forward_prediction != 0) & (data.reverse_prediction != 0)
                              & (data.forward_prediction == data.reverse_prediction)).to_numpy(dtype=float)
        correct = (audit.forward_prediction == audit.reference_label).to_numpy(dtype=float)
        reference = float((evaluation.forward_prediction == evaluation.reference_label).mean())
        assert rows.heldout_accuracy_reference.eq(reference).all()
        for row in rows.itertuples():
            assert row.heldout_reference_labels_scored == row.heldout_forward_judgments_scored == len(evaluation)
            if row.method == "raw_proxy":
                assert row.estimate == proxy(evaluation).mean()
                assert row.human_labels_used == 0
                assert row.cached_judgments_used == 2 * len(evaluation)
                assert pd.isna(row.interval_width)
                continue
            power = {"human_only": 0., "ppi": 1., "ppi_tuned": "auto"}[row.method]
            result = prediction_powered_mean(proxy(audit), correct, proxy(evaluation),
                                             labeled_groups=audit.instruction_id,
                                             unlabeled_groups=evaluation.instruction_id, power=power)
            assert row.estimate == result.point
            assert row.interval_low == result.interval.low
            assert row.interval_high == result.interval.high
            assert row.selected_power == result.selected_power
            assert row.variance_method == "one-way-cluster-robust"
            assert row.human_labels_used == len(audit)
            expected_cache = len(audit) if row.method == "human_only" else 2 * (len(audit) + len(evaluation))
            assert row.cached_judgments_used == expected_cache


def test_invalid_predictions_retained_and_never_equal_valid_gold():
    frame = panel()
    frame[["forward_prediction", "reverse_prediction"]] = 0
    trials, _ = run(frame, cohorts=["All"])
    assert trials.heldout_accuracy_reference.eq(0).all()
    assert trials.estimate.eq(0).all()
    assert trials.total_samples.eq(frame.sample_id.nunique()).all()
    # Reverse invalidity destroys the proxy but not forward correctness.
    frame["forward_prediction"] = frame["reference_label"]
    trials, _ = run(frame, cohorts=["All"])
    assert trials.heldout_accuracy_reference.eq(1).all()
    assert trials.loc[trials.method == "raw_proxy", "estimate"].eq(0).all()
    assert trials.loc[trials.method != "raw_proxy", "estimate"].eq(1).all()


def test_input_order_invariance_and_summary_recomputation():
    frame = panel()
    trials, summary = run(frame)
    shuffled, shuffled_summary = run(frame.sample(frac=1, random_state=91))
    pd.testing.assert_frame_equal(trials, shuffled)
    pd.testing.assert_frame_equal(summary, shuffled_summary)
    assert trials.attrs == shuffled.attrs
    groups = ["cohort", "judge", "labeled_fraction", "method"]
    actual = trials.groupby(groups).absolute_error.mean()
    np.testing.assert_array_equal(actual, summary.set_index(groups).mean_absolute_error)


@pytest.mark.parametrize("column,value", [
    ("reference_label", 0), ("reference_label", 3), ("reference_label", True),
    ("forward_prediction", None), ("forward_prediction", "None"),
    ("forward_prediction", "1"), ("forward_prediction", 1.0),
    ("forward_prediction", -1), ("forward_prediction", 3), ("forward_prediction", True),
    ("reverse_prediction", "unknown"), ("reverse_prediction", np.nan),
    ("subset", "adversarial"), ("sample_id", 1), ("instruction_id", " i00"), ("judge", ""),
])
def test_unknown_parse_and_invalid_metadata_rejected(column, value):
    frame = panel()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        run(frame)


def test_incomplete_duplicate_or_disagreeing_panels_are_rejected():
    frame = panel()
    with pytest.raises(ValueError, match="complete sample panel"):
        run(frame.iloc[1:])
    with pytest.raises(ValueError, match="exactly once"):
        run(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
    changed = frame.copy()
    changed.loc[0, "reference_label"] = 3 - changed.loc[0, "reference_label"]
    with pytest.raises(ValueError, match="agree across judges"):
        run(changed)
    with pytest.raises(ValueError, match="missing cached-audit"):
        run(frame.drop(columns="reverse_prediction"))
    with pytest.raises(ValueError, match="nonempty"):
        run(frame.iloc[:0])


@pytest.mark.parametrize("kwargs", [
    {"fractions": []}, {"fractions": [.2, .2]}, {"fractions": [Fraction(1, 5), .2]},
    {"fractions": [True]}, {"fractions": ["0.2"]}, {"fractions": [.01]},
    {"fractions": [.8]}, {"fractions": [np.nan]}, {"seeds": []},
    {"seeds": [1, 1]}, {"seeds": [True]}, {"seeds": [-1]},
    {"evaluation_fraction": .01}, {"evaluation_fraction": 0},
    {"alpha": Fraction(1, 10**1000)}, {"alpha": 10**1000},
    {"alpha": True}, {"alpha": 1}, {"cohorts": []},
    {"cohorts": ["All", "All"]}, {"cohorts": ["Unknown"]},
    {"min_cohort_instructions": 3}, {"min_cohort_instructions": True},
    {"min_cohort_instructions": 41},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        judge_accuracy_audit(panel(), **kwargs)


def test_absent_requested_cohort_is_not_silently_omitted():
    frame = panel().query("subset == 'Natural'")
    with pytest.raises(ValueError, match="cohort Adversarial"):
        run(frame)
