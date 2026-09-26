"""Structural, analytic, and deterministic checks of the simulation design."""
import json

import numpy as np
import pandas as pd
import pytest

from judgecal.simulation import (
    SIMULATION_SCENARIOS,
    _draw_pool,
    _mean_variance,
    run_simulation_study,
)


@pytest.fixture(scope="module")
def small_study():
    return run_simulation_study(repetitions=3, seed=123)


@pytest.mark.parametrize("kwargs", [
    {"repetitions": 0}, {"repetitions": 1}, {"repetitions": -1},
    {"repetitions": 2.5}, {"repetitions": True}, {"repetitions": "3"},
    {"seed": -1}, {"seed": 1.5}, {"seed": True}, {"seed": None},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        run_simulation_study(**kwargs)


def test_seeded_determinism_and_json_configuration(small_study):
    trials, summary, configuration = small_study
    again_trials, again_summary, again_configuration = run_simulation_study(3, 123)
    pd.testing.assert_frame_equal(trials, again_trials)
    pd.testing.assert_frame_equal(summary, again_summary)
    assert configuration == again_configuration
    assert json.loads(json.dumps(configuration, allow_nan=False)) == configuration


def test_complete_trials_schema_and_shared_budgets(small_study):
    trials, summary, configuration = small_study
    assert len(trials) == 7 * 4 * 3
    assert len(summary) == 7 * 4
    assert not trials.duplicated(["scenario", "repetition", "method"]).any()
    assert set(trials.method) == set(configuration["methods"])
    assert (trials.groupby(["scenario", "repetition"]).size() == 4).all()
    numeric = ["point", "standard_error", "ci_low", "ci_high", "interval_width", "error"]
    assert np.isfinite(trials[numeric].to_numpy()).all()
    assert (trials.ci_high >= trials.ci_low).all()
    assert np.array_equal(trials.covered, (trials.ci_low <= trials.truth) & (trials.truth <= trials.ci_high))
    for _, group in trials.groupby(["scenario", "repetition"]):
        assert group.n_labeled.nunique() == group.n_unlabeled.nunique() == 1
        assert (group[group.method != "raw_judge"].audit_labels_used == group.n_labeled.iloc[0]).all()
        assert (group[group.method == "raw_judge"].audit_labels_used == 0).all()
    assert (trials[trials.method == "human_only"].selected_power == 0.0).all()
    assert (trials[trials.method == "ppi"].selected_power == 1.0).all()


def test_frozen_scenarios_have_known_population_expectations():
    assert len({scenario.name for scenario in SIMULATION_SCENARIOS}) == 7
    for scenario in SIMULATION_SCENARIOS:
        # Enumerate the four outcomes of a human/judge observation. This
        # independently checks the analytic means recorded with the study.
        p, se, sp = scenario.audit_prevalence, scenario.audit_sensitivity, scenario.audit_specificity
        joint = [(1, 1, p * se), (1, 0, p * (1 - se)),
                 (0, 1, (1 - p) * (1 - sp)), (0, 0, (1 - p) * sp)]
        expected_y = sum(y * prob for y, f, prob in joint)
        expected_f = sum(f * prob for y, f, prob in joint)
        assert sum(prob for _, _, prob in joint) == pytest.approx(1)
        assert expected_y == pytest.approx(scenario.audit_prevalence)
        assert expected_f == pytest.approx(scenario.audit_judge_rate)
        config = scenario.configuration()
        assert config["expected_fixed_ppi"] == pytest.approx(scenario.target_judge_rate + expected_y - expected_f)
        if not scenario.validity_scope.startswith("assumption_violation"):
            assert config["expected_fixed_ppi"] == pytest.approx(config["truth"])
            assert config["expected_human_only"] == pytest.approx(config["truth"])


def test_shift_scenarios_isolate_distinct_transfer_failures():
    lookup = {scenario.name: scenario for scenario in SIMULATION_SCENARIOS}
    prevalence = lookup["prevalence_shift"]
    conditional = lookup["conditional_error_shift"]
    assert prevalence.audit_prevalence != prevalence.target_prevalence
    assert prevalence.audit_sensitivity == prevalence.target_sensitivity
    assert prevalence.audit_specificity == prevalence.target_specificity
    assert conditional.audit_prevalence == conditional.target_prevalence
    assert conditional.audit_sensitivity != conditional.target_sensitivity
    assert conditional.audit_specificity != conditional.target_specificity
    assert prevalence.configuration()["expected_fixed_ppi"] == pytest.approx(.645)
    assert conditional.configuration()["expected_fixed_ppi"] == pytest.approx(.53)
    assert lookup["iid_anticorrelated"].audit_sensitivity + lookup["iid_anticorrelated"].audit_specificity < 1


def test_cluster_generation_preserves_units_and_disjoint_pools():
    rng = np.random.default_rng(9)
    y_l, f_l, g_l = _draw_pool(rng, 8, 4, .6, .9, .8, 0)
    y_u, f_u, g_u = _draw_pool(rng, 80, 4, .6, .9, .8, 8)
    assert not (set(g_l) & set(g_u))
    assert len(y_l) == 32 and len(y_u) == 320
    for y, f, g in [(y_l, f_l, g_l), (y_u, f_u, g_u)]:
        for group in np.unique(g):
            assert len(y[g == group]) == 4
            assert len(set(y[g == group])) == len(set(f[g == group])) == 1
        # For equal-sized perfect repeats, the sandwich variance is exactly
        # the independent unit-level sample variance divided by unit count.
        assert _mean_variance(f, g) == pytest.approx(np.var(f[::4], ddof=1) / len(f[::4]))


def test_raw_intervals_are_explicitly_distinguished(small_study):
    trials, summary, _ = small_study
    raw = trials[trials.method == "raw_judge"]
    assert (raw.interval_estimand == "target_judge_positive_rate").all()
    assert raw.selected_power.isna().all()
    assert raw.estimated_variance_ratio.isna().all()
    assert (raw.native_truth != raw.truth).all()
    assert np.array_equal(raw.native_covered, (raw.ci_low <= raw.native_truth) & (raw.native_truth <= raw.ci_high))
    assert summary[summary.method == "raw_judge"].mean_selected_power.isna().all()


def test_human_only_native_audit_target_is_distinct_under_prevalence_shift(small_study):
    trials, summary, _ = small_study
    human = trials[trials.method == "human_only"]
    assert (human.interval_estimand == "audit_human_preference_rate").all()
    shift = human[human.scenario == "prevalence_shift"]
    assert (shift.native_truth == .4).all()
    assert (shift.truth == .75).all()
    assert np.array_equal(shift.native_covered, (shift.ci_low <= .4) & (.4 <= shift.ci_high))
    assert np.array_equal(shift.covered, (shift.ci_low <= .75) & (.75 <= shift.ci_high))
    row = summary[(summary.method == "human_only") & (summary.scenario == "prevalence_shift")].iloc[0]
    assert row.native_truth == .4
    assert row.native_coverage == pytest.approx(shift.native_covered.mean())


def test_summary_statistics_match_retained_draws(small_study):
    trials, summary, _ = small_study
    for row in summary.itertuples():
        group = trials[(trials.scenario == row.scenario) & (trials.method == row.method)]
        assert row.bias == pytest.approx(group.error.mean())
        assert row.rmse == pytest.approx(np.sqrt((group.error ** 2).mean()))
        assert row.coverage == pytest.approx(group.covered.mean())
        assert row.mean_width == pytest.approx(group.interval_width.mean())
        assert row.bias_mcse == pytest.approx(group.error.std(ddof=1) / np.sqrt(3))
        assert row.coverage_mcse == pytest.approx(np.sqrt(row.coverage * (1 - row.coverage) / 3))
