"""Budgeted independent pilots: contracts, exact summaries and stable draws."""
from fractions import Fraction
import json

import numpy as np
import pandas as pd
import pytest

from judgecal import pilot_study as study


@pytest.fixture(scope="module")
def small_study():
    return study.independent_pilot_study(3, 77)


def test_pilot_moments_and_factor_are_in_original_score_units():
    f = np.array([.1, .2, .35, .6, .9])
    y = np.array([.9, .1, .4, .5, .8])
    result = study.pilot_power(f, y, 16, 200)
    variance = np.var(f, ddof=1)
    covariance = np.cov(f, y, ddof=1)[0, 1]
    assert result["variance"] == pytest.approx(variance)
    assert result["covariance"] == pytest.approx(covariance)
    assert result["selected_power"] == pytest.approx(np.clip(covariance/((1+16/200)*variance), 0, 1))
    assert result["constant_proxy"] is False
    assert (result["n_pilot"], result["n_inference"], result["n_unlabeled"]) == (5, 16, 200)


def test_exact_binary_zero_covariance_preserves_zero_power_eb_allocation():
    # Independent replay found this balanced-F, imbalanced-Y pilot at the
    # uninformative B=20, repetition=40 cell. Its covariance numerator is
    # 10*1 - 2*5 = 0, which must not acquire a positive rounding residue.
    pilot_y = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    pilot_f = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 0])
    metadata = study.pilot_power(pilot_f, pilot_y, 10, 200)
    assert metadata["covariance"] == 0.
    assert metadata["selected_power"] == 0.
    assert metadata["variance"] == pytest.approx(25/90)
    assert metadata["constant_proxy"] is False
    final_y, final_f, f_u = np.tile([0., 1.], 5), np.tile([1., 0.], 5), np.tile([0., 1.], 100)
    records = {row["method"]: row for row in study.evaluate_draw(
        np.r_[pilot_y, final_y], np.r_[pilot_f, final_f], f_u)}
    expected = study.finite_sample_mean(final_f, final_y, f_u, method="empirical_bernstein", power=0.)
    observed = records["pilot50_eb"]
    assert observed["selected_power"] == 0.
    assert observed["interval_low"] == expected.interval.low
    assert observed["interval_high"] == expected.interval.high
    assert records["pilot50_normal"]["point"] == observed["point"]


def test_joint_score_reflection_preserves_power_and_reflects_point_intervals():
    rng = np.random.default_rng(209)
    y, f, u = rng.random(20), rng.random(20), rng.random(34)
    before, after = study.evaluate_draw(y, f, u), study.evaluate_draw(1-y, 1-f, 1-u)
    for left, right in zip(before, after):
        assert left["method"] == right["method"]
        assert left["selected_power"] == pytest.approx(right["selected_power"], abs=1e-14)
        assert right["point"] == pytest.approx(1-left["point"])
        assert right["interval_low"] == pytest.approx(1-left["interval_high"])
        assert right["interval_high"] == pytest.approx(1-left["interval_low"])


def test_fractional_scores_supported_and_minimum_budget_explicit():
    records = study.evaluate_draw([Fraction(1, 3)]*10, [.2]*10, [.7, .2])
    assert len(records) == 8
    assert all(record["point"] == pytest.approx(1/3) for record in records)
    assert all(record["n_pilot"] in (0, 2, 5) for record in records)
    assert all(record["pilot_constant"] for record in records if record["n_pilot"])
    with pytest.raises(ValueError, match="at least ten"):
        study.evaluate_draw([.2]*9, [.5]*9, [0., 1.])


def test_complete_grid_same_draw_method_panel_and_costs(small_study):
    trials, summary, paired = small_study
    assert trials.shape[0] == 12*3*8
    assert len(summary) == 12*8
    assert len(paired) == 12*48
    assert trials.groupby(["scenario", "repetition"]).size().eq(8).all()
    assert not trials.duplicated(["scenario", "repetition", "method"]).any()
    assert set(trials.profile) == {"balanced_strong", "boundary_strong", "uninformative", "perfect"}
    assert set(trials.total_budget) == {20, 200, 1000}
    assert trials.n_unlabeled.eq(10*trials.total_budget).all()
    assert trials.total_labels_used.eq(trials.total_budget).all()
    assert trials.n_labeled.add(trials.n_pilot).eq(trials.total_budget).all()
    assert trials.proxy_scores_supplied.eq(trials.total_budget+trials.n_unlabeled).all()
    human = trials.method.str.startswith("human_")
    assert trials.loc[human, "unique_prediction_scores_used"].eq(0).all()
    assert trials.loc[~human, "unique_prediction_scores_used"].eq(trials.loc[~human, "proxy_scores_supplied"]).all()
    config = trials.attrs["configuration"]
    assert json.loads(json.dumps(config, allow_nan=False)) == config
    assert config["profile_codes"] == {"balanced_strong": 1, "boundary_strong": 2, "uninformative": 3, "perfect": 4}
    assert config["distinct_point_methods"] == list(study.POINT_METHODS)
    assert config["numpy_version"] == np.__version__


def test_normal_and_finite_versions_share_exact_points_and_powers(small_study):
    trials, _, _ = small_study
    for _, group in trials.groupby(["scenario", "repetition"]):
        indexed = group.set_index("method")
        for prefix in ("human", "pilot20", "pilot50"):
            left, right = indexed.loc[prefix+"_normal"], indexed.loc[prefix+"_eb"]
            assert left.point == right.point
            assert left.selected_power == right.selected_power
            assert left.conditional_variance == right.conditional_variance
    normal = trials.interval_family.eq("normal")
    assert trials.loc[normal, "estimated_variance"].notna().all()
    np.testing.assert_array_equal(trials.loc[normal, "estimated_variance"], trials.loc[normal, "standard_error"]**2)
    assert trials.loc[~normal, "estimated_variance"].isna().all()
    assert trials.theorem_applicable.eq(~normal).all()


def test_scoring_preserves_endpoints_and_known_variance_scope(small_study):
    trials, _, _ = small_study
    np.testing.assert_array_equal(trials.squared_error, (trials.point-trials.truth)**2)
    np.testing.assert_array_equal(trials.width, trials.interval_high-trials.interval_low)
    assert trials.full_domain.eq((trials.interval_low <= 0) & (trials.interval_high >= 1)).all()
    assert trials.zero_width.eq(trials.width.eq(0)).all()
    for row in trials.itertuples():
        assert row.covered == study._coverage_contains(row.interval_low, row.interval_high, row.truth)
    same_audit = trials.method.str.startswith("same_audit")
    assert trials.conditional_variance_applicable.eq(~same_audit).all()
    assert trials.loc[same_audit, "conditional_variance"].isna().all()
    assert trials.loc[~same_audit, "conditional_variance"].ge(0).all()
    human = trials[trials.method.str.startswith("human_")]
    np.testing.assert_allclose(human.conditional_variance, human.truth*(1-human.truth)/human.total_budget)
    uninformative = trials[(trials.profile == "uninformative") & trials.method.str.startswith("pilot")]
    assert (uninformative.conditional_variance >= .25/uninformative.n_labeled).all()


def test_known_law_diagnostic_has_exact_perfect_and_independent_controls():
    assert study._conditional_variance(1., 20, 200, .5, 1., 1.) == .25/200
    assert study._conditional_variance(0., 20, 200, .5, 1., 1.) == .25/20
    power = .3
    expected = .25/20 + power**2*.6*.4*(1/20+1/200)
    assert study._conditional_variance(power, 20, 200, .5, .6, .4) == pytest.approx(expected)


def test_simultaneous_grid_diagnostics_match_selected_candidate(small_study):
    trials, _, _ = small_study
    grid = trials[trials.method == "same_audit_grid_eb"]
    for row in grid.itertuples():
        powers = json.loads(row.candidate_powers_json)
        radii = json.loads(row.candidate_radii_json)
        points = json.loads(row.candidate_points_json)
        assert powers == list(study.POWER_GRID)
        index = powers.index(row.selected_power)
        assert row.point == points[index]
        assert row.width == pytest.approx(2*radii[index])
        assert radii[index] == min(radii)


def test_stable_philox_keys_preserve_prefixes_and_cell_order(monkeypatch, small_study):
    trials, _, _ = small_study
    extended, _, _ = study.independent_pilot_study(4, 77)
    pd.testing.assert_frame_equal(trials.reset_index(drop=True), extended[extended.repetition < 3].reset_index(drop=True))
    monkeypatch.setattr(study, "PROFILES", tuple(reversed(study.PROFILES)))
    reordered, _, _ = study.independent_pilot_study(3, 77)
    keys = ["scenario", "repetition", "method"]
    pd.testing.assert_frame_equal(trials.sort_values(keys).reset_index(drop=True), reordered.sort_values(keys).reset_index(drop=True))
    changed, _, _ = study.independent_pilot_study(3, 78)
    assert not np.array_equal(reordered.point, changed.point)


def test_every_summary_replays_raw_rows_and_exact_coverage_mc(small_study):
    trials, summary, _ = small_study
    for row in summary.itertuples():
        group = trials[(trials.scenario == row.scenario) & (trials.method == row.method)]
        error = group.error.to_numpy()
        loss = group.squared_error.to_numpy()
        assert row.bias == pytest.approx(error.mean())
        assert row.bias_mcse == pytest.approx(error.std(ddof=1)/np.sqrt(3))
        assert row.mse == pytest.approx(loss.mean())
        assert row.mse_mcse == pytest.approx(loss.std(ddof=1)/np.sqrt(3))
        assert row.rmse == pytest.approx(np.sqrt(loss.mean()))
        assert row.mean_width == pytest.approx(group.width.mean())
        assert row.width_mcse == pytest.approx(group.width.std(ddof=1)/np.sqrt(3))
        assert row.coverage == group.covered.mean()
        assert row.coverage_mc_low <= row.coverage <= row.coverage_mc_high
        assert row.zero_power_fraction == group.selected_power.eq(0).mean()
        assert row.constant_pilot_fraction == group.pilot_constant.mean()
    all_covered = trials.copy()
    all_covered["covered"] = True
    summary = study._summarize(all_covered)
    np.testing.assert_allclose(summary.coverage_mc_low, .025**(1/3))
    assert summary.coverage_mc_high.eq(1).all()
    all_covered["covered"] = False
    summary = study._summarize(all_covered)
    assert summary.coverage_mc_low.eq(0).all()
    np.testing.assert_allclose(summary.coverage_mc_high, 1-.025**(1/3))


def test_all_paired_contrasts_retain_covariance_and_unique_risk_rules(small_study):
    trials, _, paired = small_study
    for row in paired.itertuples():
        cell = trials[trials.scenario == row.scenario]
        left = cell[cell.method == row.method].sort_values("repetition")
        right = cell[cell.method == row.baseline].sort_values("repetition")
        if row.metric == "rmse":
            a, b = np.sqrt(left.squared_error.mean()), np.sqrt(right.squared_error.mean())
            expected = a-b
            values = left.squared_error.to_numpy()/(2*a)-right.squared_error.to_numpy()/(2*b)
        else:
            column = {"bias": "error", "mse": "squared_error", "width": "width", "coverage": "covered"}[row.metric]
            values = left[column].to_numpy(dtype=float)-right[column].to_numpy(dtype=float)
            expected = values.mean()
        assert row.mean_difference == pytest.approx(expected)
        assert row.mcse == pytest.approx(values.std(ddof=1)/np.sqrt(3))
        if row.metric in ("width", "coverage"):
            assert left.interval_family.iloc[0] == right.interval_family.iloc[0]
        else:
            assert row.method in study.POINT_METHODS
            assert row.baseline in ("human_normal", "same_audit_normal")


@pytest.mark.parametrize("bad", [True, "0.5", None, complex(.5), np.nan, np.inf, -.1, 1.1])
def test_strict_score_validation(bad):
    with pytest.raises(ValueError):
        study.pilot_power([0., bad], [.2, .8], 16, 200)
    with pytest.raises(ValueError):
        study.evaluate_draw([.2]*20, [.1]*20, [0., bad])


def test_missing_mask_and_shapes_rejected():
    with pytest.raises(ValueError, match="masked"):
        study.pilot_power(np.ma.array([0., 1.], mask=[0, 1]), [.2, .8], 16, 200)
    with pytest.raises(ValueError, match="equal lengths"):
        study.pilot_power([0., 1.], [.2, .5, .8], 16, 200)
    with pytest.raises(ValueError, match="one-dimensional"):
        study.evaluate_draw([[.2]*20], [.1]*20, [0., 1.])
    with pytest.raises(ValueError, match="equal lengths"):
        study.evaluate_draw([.2]*20, [.1]*19, [0., 1.])


@pytest.mark.parametrize("kwargs", [
    {"repetitions": 1}, {"repetitions": True}, {"repetitions": 2.5},
    {"seed": -1}, {"seed": False}, {"seed": 1.2},
    {"alpha": 0}, {"alpha": 1}, {"alpha": True}, {"alpha": np.nan},
    {"alpha": Fraction(1, 10**1000)},
])
def test_invalid_study_options(kwargs):
    with pytest.raises(ValueError):
        study.independent_pilot_study(**kwargs)


@pytest.mark.parametrize("counts", [(1, 2), (2, 1), (True, 2), (2, 2.1), (-1, 2)])
def test_invalid_pilot_design_counts(counts):
    with pytest.raises(ValueError):
        study.pilot_power([0., 1.], [.2, .8], *counts)
