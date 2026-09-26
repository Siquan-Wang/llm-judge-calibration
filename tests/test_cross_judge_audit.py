"""Cross-model proxy identity, shared prompt groups and honest cached costs."""
import json

import numpy as np
import pandas as pd
import pytest

from judgecal import cross_judge_audit as audit
from judgecal.judge_audit import judge_accuracy_audit
from judgecal.pool_prediction import audit_residual_mean
from judgecal.ppi import prediction_powered_mean
from judgecal.rewardbench import REWARDBENCH_JUDGES


def panel():
    rng = np.random.default_rng(147)
    rows = []
    for group in range(40):
        for copy in range(2 if group % 3 == 0 else 1):
            # Same prompt can occur in multiple real subsets and must not split.
            subset = (("alpacaeval-easy", "alpacaeval-hard") if group < 20 else
                      ("llmbar-natural", "llmbar-adver-neighbor"))[copy]
            choices = rng.integers(1, 3, size=2)
            reference = 1 + (group + copy) % 2
            for index, judge in enumerate(REWARDBENCH_JUDGES):
                rows.append({
                    "sample_id": f"s{group:02d}_{copy}", "instruction_id": f"i{group:02d}",
                    "source_id": group, "subset": subset, "judge": judge,
                    "reference_choice": reference, "canonical_choice": int(choices[index]),
                    "reference_correct": int(choices[index] == reference),
                    "agreement_proxy": int(choices[0] == choices[1]),
                    "auxiliary_judge": REWARDBENCH_JUDGES[1-index],
                })
    return pd.DataFrame(rows)


def run(labels=None, **kwargs):
    return audit.cross_judge_accuracy_audit(panel() if labels is None else labels, seeds=[7], **kwargs)


def refresh_reference_correct(frame):
    frame["reference_correct"] = frame.canonical_choice.eq(frame.reference_choice).astype(int)


def refresh_agreement(frame):
    choices = frame.pivot(index="sample_id", columns="judge", values="canonical_choice")
    agreement = choices.iloc[:, 0].eq(choices.iloc[:, 1]).astype(int)
    frame["agreement_proxy"] = frame.sample_id.map(agreement)


def test_shared_splits_complete_prompts_nested_budgets_and_true_cohorts():
    labels = panel()
    trials, summary = run(labels)
    assert len(trials) == 3 * 3 * 2 * 6 and len(summary) == len(trials)
    assert set(trials.method) == {"raw_proxy", "human_only", "ppi", "ppi_tuned", "ppi_signed", "audit_residual"}
    assert set(trials.cohort) == {"NonLLMBar", "All", "LLMBar"}
    assert not any("coverage" in name or "mcse" in name for name in summary)
    assert trials.target_judge.ne(trials.auxiliary_judge).all()
    assert "judge" not in trials and "human_labels_used" not in trials
    assert "mean_reference_labels_used" in summary
    manifests = trials.attrs["split_manifest"]
    assert len(manifests) == 9
    for cohort in ("NonLLMBar", "All", "LLMBar"):
        selected = labels if cohort == "All" else labels.loc[labels.subset.str.startswith("llmbar-") == (cohort == "LLMBar")]
        splits = [m for m in manifests if m["cohort"] == cohort]
        previous = set()
        for split in splits:
            training, evaluation, unused = [set(split[f"{part}_instruction_ids"])
                                            for part in ("labeled", "evaluation", "unused")]
            assert previous <= training
            assert evaluation == set(splits[0]["evaluation_instruction_ids"])
            assert not training & evaluation and not training & unused and not evaluation & unused
            assert training | evaluation | unused == set(selected.instruction_id)
            assert len(evaluation) == int(.25 * selected.instruction_id.nunique())
            assert len(training) == int(split["labeled_fraction"] * selected.instruction_id.nunique())
            for part in ("labeled", "evaluation", "unused"):
                expected = set(selected.loc[selected.instruction_id.isin(split[f"{part}_instruction_ids"]), "sample_id"])
                assert expected == set(split[f"{part}_sample_ids"])
            rows = trials.loc[(trials.cohort == cohort) & (trials.labeled_fraction == split["labeled_fraction"])]
            assert rows.split_sha256.nunique() == 1
            assert len(split["target_judges"]) == 2 and "judges" not in split
            previous = training
        raw = trials.loc[(trials.cohort == cohort) & trials.method.eq("raw_proxy")]
        assert raw.estimate.nunique() == 1  # Same proxy and fixed target for both directions/all budgets.
    for item in trials.attrs["cohort_metadata"]:
        assert sum(item["subset_counts"].values()) == item["samples"]
    public = json.dumps(trials.attrs) + " ".join(trials.columns) + " ".join(summary.columns)
    for forbidden in ("forward", "reverse", "Natural", "order agreement"):
        assert forbidden not in public


def test_hidden_and_unused_reference_isolation_and_gold_reversal_invariance():
    labels = panel()
    before, _ = run(labels, cohorts=["All"], fractions=[.2])
    split = before.attrs["split_manifest"][0]
    changed = labels.copy()
    hidden = changed.sample_id.isin(split["evaluation_sample_ids"])
    changed.loc[hidden, "reference_choice"] = 3-changed.loc[hidden, "reference_choice"]
    refresh_reference_correct(changed)
    after, _ = run(changed, cohorts=["All"], fractions=[.2])
    inference = ["estimate", "selected_power", "interval_low", "interval_high", "standard_error",
                 "audit_residual_criterion", "reference_labels_used", "cached_judgments_used"]
    pd.testing.assert_frame_equal(before[inference], after[inference])
    assert before.attrs == after.attrs
    assert not before.heldout_accuracy_reference.equals(after.heldout_accuracy_reference)
    unused = labels.copy()
    mask = unused.sample_id.isin(split["unused_sample_ids"])
    unused.loc[mask, "reference_choice"] = 3-unused.loc[mask, "reference_choice"]
    unused.loc[mask, "canonical_choice"] = 3-unused.loc[mask, "canonical_choice"]
    refresh_reference_correct(unused)
    refresh_agreement(unused)
    unused_result, _ = run(unused, cohorts=["All"], fractions=[.2])
    pd.testing.assert_frame_equal(before, unused_result)
    reversed_gold = labels.copy()
    reversed_gold.reference_choice = 3-reversed_gold.reference_choice
    refresh_reference_correct(reversed_gold)
    full, _ = run(reversed_gold, cohorts=["All"], fractions=[.2])
    assert labels.agreement_proxy.equals(reversed_gold.agreement_proxy)
    assert full.loc[full.method.eq("raw_proxy"), "estimate"].equals(before.loc[before.method.eq("raw_proxy"), "estimate"])


def test_direct_grouped_formulas_point_only_scope_and_shared_logical_costs():
    labels = panel()
    trials, _ = run(labels, cohorts=["All"], fractions=[.4])
    split = trials.attrs["split_manifest"][0]
    for target, rows in trials.groupby("target_judge"):
        source = labels.loc[labels.judge.eq(target)].sort_values(["instruction_id", "sample_id"])
        train = source.loc[source.sample_id.isin(split["labeled_sample_ids"])]
        evaluation = source.loc[source.sample_id.isin(split["evaluation_sample_ids"])]
        f, y, u = train.agreement_proxy, train.reference_correct, evaluation.agreement_proxy
        assert rows.heldout_accuracy_reference.eq(evaluation.reference_correct.mean()).all()
        for row in rows.itertuples():
            assert row.heldout_reference_labels_scored == row.heldout_target_judgments_scored == len(evaluation)
            if row.method == "raw_proxy":
                assert row.estimate == u.mean()
                assert row.reference_labels_used == 0 and row.cached_judgments_used == 2*len(evaluation)
                assert pd.isna(row.interval_width)
                continue
            assert row.reference_labels_used == len(train)
            assert row.cached_judgments_used == (len(train) if row.method == "human_only" else 2*(len(train)+len(evaluation)))
            kwargs = {"labeled_groups": train.instruction_id, "unlabeled_groups": evaluation.instruction_id}
            if row.method == "audit_residual":
                expected = audit_residual_mean(f, y, u, **kwargs)
                assert row.estimate == expected.point and row.selected_power == expected.selected_power
                assert row.audit_residual_criterion == expected.criterion_value
                assert all(pd.isna(getattr(row, name)) for name in ("interval_low", "interval_high", "interval_width", "standard_error", "estimated_variance_ratio"))
            else:
                power = {"human_only": 0., "ppi": 1., "ppi_tuned": "auto", "ppi_signed": "auto"}[row.method]
                expected = prediction_powered_mean(f, y, u, **kwargs, power=power,
                                                   power_bounds=(-1., 1.) if row.method == "ppi_signed" else (0., 1.))
                assert row.estimate == expected.point and row.selected_power == expected.selected_power
                assert row.interval_low == expected.interval.low and row.standard_error == expected.standard_error


def test_private_transport_reuses_legacy_numerics_without_changing_engine_behavior():
    labels = audit._validate_labels(panel())
    transport = audit._transport_panel(labels)
    old_trials, old_summary = judge_accuracy_audit(transport, seeds=[7], cohorts=["All"], include_signed=True, include_pool_tuned=True)
    trials, summary = run(labels, cohorts=["All"])
    renames = {"judge": "target_judge", "human_labels_used": "reference_labels_used",
               "mean_human_labels_used": "mean_reference_labels_used", "heldout_forward_judgments_scored": "heldout_target_judgments_scored"}
    for old, new in ((old_trials, trials), (old_summary, summary)):
        expected = old.rename(columns=renames)
        pd.testing.assert_frame_equal(expected, new.drop(columns="auxiliary_judge"))
    after, after_summary = judge_accuracy_audit(transport, seeds=[7], cohorts=["All"], include_signed=True, include_pool_tuned=True)
    pd.testing.assert_frame_equal(old_trials, after)
    pd.testing.assert_frame_equal(old_summary, after_summary)
    assert old_trials.attrs == after.attrs


def test_shuffled_input_preserves_every_output_and_manifest():
    first, summary = run()
    shuffled, second_summary = run(panel().sample(frac=1, random_state=38))
    pd.testing.assert_frame_equal(first, shuffled)
    pd.testing.assert_frame_equal(summary, second_summary)
    assert first.attrs == shuffled.attrs


def test_direction_swap_and_canonical_orientation_preserve_agreement():
    labels = panel()
    before, summary = run(labels, cohorts=["All"])
    assert labels.source_id.nunique() < labels.sample_id.nunique()
    both_wrong = labels.groupby("sample_id").reference_correct.sum().eq(0)
    assert both_wrong.any()
    assert labels.loc[labels.sample_id.isin(both_wrong[both_wrong].index), "agreement_proxy"].eq(1).all()
    swap = dict(zip(REWARDBENCH_JUDGES, reversed(REWARDBENCH_JUDGES)))
    changed = labels.copy()
    changed.judge = changed.judge.map(swap)
    changed.auxiliary_judge = changed.auxiliary_judge.map(swap)
    after, _ = run(changed, cohorts=["All"])
    after.target_judge = after.target_judge.map(swap)
    after.auxiliary_judge = after.auxiliary_judge.map(swap)
    keys = ["cohort", "seed", "labeled_fraction", "target_judge", "method"]
    pd.testing.assert_frame_equal(before.sort_values(keys).reset_index(drop=True),
                                  after.sort_values(keys).reset_index(drop=True))
    # Relabel both response alternatives without changing either choice relation.
    reoriented = labels.copy()
    reoriented.reference_choice = 3-reoriented.reference_choice
    reoriented.canonical_choice = 3-reoriented.canonical_choice
    again, second_summary = run(reoriented, cohorts=["All"])
    pd.testing.assert_frame_equal(before, again)
    pd.testing.assert_frame_equal(summary, second_summary)
    assert before.attrs == again.attrs


@pytest.mark.parametrize("column,value", [
    ("canonical_choice", 0), ("canonical_choice", .5), ("canonical_choice", True),
    ("reference_choice", 0), ("reference_choice", "1"), ("reference_correct", .5),
    ("agreement_proxy", 2), ("source_id", True), ("source_id", -1),
    ("judge", "other"), ("auxiliary_judge", "other"), ("subset", "other"),
    ("instruction_id", " "), ("sample_id", None),
])
def test_invalid_contract_values_raise_without_row_drops(column, value):
    labels = panel().astype({column: object})
    labels.loc[0, column] = value
    with pytest.raises(ValueError):
        run(labels)


@pytest.mark.parametrize("problem", ["missing_column", "duplicate", "missing_partner", "metadata", "reference", "correctness", "proxy", "self_auxiliary"])
def test_inconsistent_or_incomplete_panel_raises(problem):
    labels = panel()
    if problem == "missing_column": labels = labels.drop(columns="canonical_choice")
    elif problem == "duplicate": labels = pd.concat([labels, labels.iloc[[0]]])
    elif problem == "missing_partner": labels = labels.iloc[1:]
    elif problem == "metadata": labels.loc[0, "instruction_id"] = "different"
    elif problem == "reference": labels.loc[0, "reference_choice"] = 3-labels.loc[0, "reference_choice"]
    elif problem == "correctness": labels.loc[0, "reference_correct"] = 1-labels.loc[0, "reference_correct"]
    elif problem == "proxy": labels.loc[0, "agreement_proxy"] = 1-labels.loc[0, "agreement_proxy"]
    elif problem == "self_auxiliary": labels.loc[0, "auxiliary_judge"] = labels.loc[0, "judge"]
    with pytest.raises(ValueError):
        run(labels)


@pytest.mark.parametrize("kwargs", [
    {"cohorts": []}, {"cohorts": ["All", "All"]}, {"cohorts": ["Natural"]}, {"cohorts": [{}]},
    {"fractions": [.8]}, {"fractions": [True]}, {"evaluation_fraction": 0},
    {"alpha": False}, {"min_cohort_instructions": 100},
])
def test_invalid_controls_raise(kwargs):
    with pytest.raises(ValueError):
        run(**kwargs)
