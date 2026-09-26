"""Known-truth design, scope flags, paired widths and Monte Carlo summaries."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from judgecal import finite_sample
from judgecal import finite_study


@pytest.fixture(scope="module")
def small_study():
    return finite_study.run_finite_sample_study(3, 19)


def test_complete_iid_grid_and_explicit_stress_scenarios(small_study):
    trials, summary, config = small_study
    assert len(trials) == 14 * 8 * 3
    assert len(summary) == 14 * 8
    assert len(config["scenarios"]) == 14
    iid = [row for row in config["scenarios"] if row["scenario_family"] == "iid"]
    assert {(row["profile"], row["audit_units"]) for row in iid} == {
        (name, size) for name in ("balanced_strong", "rare_strong",
                                 "balanced_uninformative", "balanced_perfect")
        for size in (20, 200, 1000)
    }
    assert all(row["validity_scope"] == "iid_same_population" for row in iid)
    assert all(row["target_units"] == 10 * row["audit_units"] for row in config["scenarios"])
    assert not trials.duplicated(["scenario", "repetition", "method"]).any()
    assert trials.groupby(["scenario", "repetition"]).size().eq(8).all()
    assert trials.audit_labels_used.eq(trials.n_labeled).all()
    assert json.loads(json.dumps(config, allow_nan=False)) == config
    for row in config["scenarios"]:
        p, se, sp = row["audit_prevalence"], row["audit_sensitivity"], row["audit_specificity"]
        q_l = p * se + (1-p) * (1-sp)
        q_u = row["target_prevalence"] * row["target_sensitivity"] + (1-row["target_prevalence"]) * (1-row["target_specificity"])
        assert row["expected_fixed_ppi"] == pytest.approx(p + q_u - q_l)
        if row["scenario_family"] == "iid":
            assert row["expected_fixed_ppi"] == pytest.approx(row["truth"])


def test_deterministic_draws_are_unchanged_when_stress_cases_are_omitted(small_study):
    trials, summary, config = small_study
    again, again_summary, again_config = finite_study.run_finite_sample_study(3, 19)
    pd.testing.assert_frame_equal(trials, again)
    pd.testing.assert_frame_equal(summary, again_summary)
    assert config == again_config
    iid, iid_summary, iid_config = finite_study.run_finite_sample_study(3, 19, False)
    pd.testing.assert_frame_equal(trials[trials.scenario_family == "iid"].reset_index(drop=True), iid)
    pd.testing.assert_frame_equal(summary[summary.scenario_family == "iid"].reset_index(drop=True), iid_summary)
    assert len(iid_config["scenarios"]) == 12


def test_every_method_receives_identical_generated_arrays(monkeypatch):
    seen = []
    normal_original = finite_study.prediction_powered_mean
    finite_original = finite_sample.finite_sample_mean

    def record(call):
        def wrapped(f_l, y_l, f_u, *args, **kwargs):
            seen.append(tuple(hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in (f_l, y_l, f_u)))
            return call(f_l, y_l, f_u, *args, **kwargs)
        return wrapped

    monkeypatch.setattr(finite_study, "prediction_powered_mean", record(normal_original))
    monkeypatch.setattr(finite_sample, "finite_sample_mean", record(finite_original))
    finite_study.run_finite_sample_study(2, 4)
    assert len(seen) == 14 * 2 * 8
    assert all(len(set(seen[index:index+8])) == 1 for index in range(0, len(seen), 8))


def test_theorem_flags_are_specific_to_finite_guarantees_and_intended_target(small_study):
    trials, _, _ = small_study
    iid = trials[trials.scenario_family == "iid"]
    assert iid.loc[iid.interval_family != "normal", "theorem_applicable"].all()
    assert not iid.loc[iid.interval_family == "normal", "theorem_applicable"].any()
    violations = trials[trials.scenario_family != "iid"]
    assert not violations.theorem_applicable.any()
    dependence = trials[trials.scenario_family == "dependence_violation"]
    assert dependence.n_labeled.eq(200).all()
    assert dependence.audit_units.eq(50).all()
    assert not dependence.native_theorem_applicable.any()
    shift = trials[trials.scenario_family == "shift_violation"]
    human = shift[shift.method.str.startswith("human_")]
    assert human.native_truth.eq(.4).all()
    assert human.truth.eq(.75).all()
    assert human.loc[human.interval_family != "normal", "native_theorem_applicable"].all()
    assert not human.loc[human.interval_family == "normal", "native_theorem_applicable"].any()
    other = shift[~shift.method.str.startswith("human_")]
    assert other.native_truth.isna().all()
    assert other.native_covered.isna().all()
    assert other.native_theorem_applicable.isna().all()
    assert other.native_estimand.eq("not_reported_for_shifted_rectified_functional").all()


def test_finite_candidate_metadata_identifies_the_selected_untruncated_interval(small_study):
    trials, _, _ = small_study
    for row in trials.itertuples():
        assert row.ci_high >= row.ci_low
        assert row.interval_width == pytest.approx(row.ci_high-row.ci_low)
        if row.interval_family == "normal":
            assert row.candidate_count == 0
            assert row.candidate_powers_json == row.candidate_radii_json == row.candidate_points_json == "[]"
            continue
        powers = json.loads(row.candidate_powers_json)
        radii = json.loads(row.candidate_radii_json)
        points = json.loads(row.candidate_points_json)
        assert len(powers) == len(radii) == len(points) == row.candidate_count
        chosen = powers.index(row.selected_power)
        assert row.point == points[chosen]
        assert row.interval_width == pytest.approx(2 * radii[chosen])
        assert radii[chosen] == min(radii)
        if row.method == "ppi_eb_grid":
            assert powers == [0., .25, .5, .75, 1.]
        else:
            assert len(powers) == 1
        assert np.isnan(row.standard_error)
    hoeffding = trials[trials.interval_family == "hoeffding"]
    assert hoeffding.variance_method.eq("bounded-range-concentration").all()


def test_width_comparisons_are_paired_and_domain_diagnostics_are_distinct(small_study):
    trials, summary, _ = small_study
    for (_, _), group in trials.groupby(["scenario", "repetition"]):
        by_method = group.set_index("method")
        for row in group.itertuples():
            baseline = by_method.loc[row.paired_width_reference]
            assert baseline.interval_family == row.interval_family
            assert row.width_difference_vs_human_family == pytest.approx(row.interval_width-baseline.interval_width)
    for row in summary.itertuples():
        group = trials[(trials.scenario == row.scenario) & (trials.method == row.method)]
        difference = group.width_difference_vs_human_family
        assert row.width_difference_vs_human_family == pytest.approx(difference.mean())
        assert row.width_difference_vs_human_family_mcse == pytest.approx(difference.std(ddof=1)/np.sqrt(3))
        assert row.fraction_width_above_one == pytest.approx(group.interval_width.gt(1).mean())
        assert row.fraction_full_domain_covered == pytest.approx(((group.ci_low<=0)&(group.ci_high>=1)).mean())
    expected_domain = (trials.ci_low <= 0) & (trials.ci_high >= 1)
    np.testing.assert_array_equal(trials.full_domain_covered, expected_domain)
    assert ((trials.interval_width > 1) & ~trials.full_domain_covered).any()
    human_summary = summary[summary.method.str.startswith("human_")]
    assert human_summary.width_difference_vs_human_family.eq(0).all()
    assert human_summary.width_difference_vs_human_family_mcse.eq(0).all()


def test_exact_mc_coverage_intervals_retain_boundary_uncertainty(small_study):
    trials, _, _ = small_study
    selected = trials[trials.scenario == trials.scenario.iloc[0]].copy()
    selected["covered"] = True
    full = finite_study._summarize_finite_trials(selected)
    assert full.coverage.eq(1).all()
    assert full.coverage_mcse.eq(0).all()
    np.testing.assert_allclose(full.coverage_mc_low, .025**(1/3))
    assert full.coverage_mc_high.eq(1).all()
    selected["covered"] = False
    empty = finite_study._summarize_finite_trials(selected)
    assert empty.coverage.eq(0).all()
    assert empty.coverage_mc_low.eq(0).all()
    np.testing.assert_allclose(empty.coverage_mc_high, 1-.025**(1/3))


def test_summary_metrics_match_all_retained_draws(small_study):
    trials, summary, _ = small_study
    assert np.isfinite(trials[["point", "ci_low", "ci_high", "interval_width", "error"]]).all().all()
    np.testing.assert_array_equal(trials.covered, (trials.ci_low <= trials.truth) & (trials.truth <= trials.ci_high))
    for row in summary.itertuples():
        group = trials[(trials.scenario == row.scenario) & (trials.method == row.method)]
        assert row.bias == pytest.approx(group.error.mean())
        assert row.rmse == pytest.approx(np.sqrt((group.error**2).mean()))
        assert row.mean_width == pytest.approx(group.interval_width.mean())
        assert row.coverage == pytest.approx(group.covered.mean())
        assert row.zero_width_draws == group.interval_width.eq(0).sum()
        assert row.coverage_mc_low <= row.coverage <= row.coverage_mc_high
        assert row.native_coverage_repetitions == group.native_covered.notna().sum()


@pytest.mark.parametrize("kwargs", [
    {"repetitions": 1}, {"repetitions": -1}, {"repetitions": 1.5},
    {"repetitions": True}, {"seed": -1}, {"seed": 1.5}, {"seed": True},
    {"include_stress_cases": 1}, {"include_stress_cases": None},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        finite_study.run_finite_sample_study(**kwargs)
