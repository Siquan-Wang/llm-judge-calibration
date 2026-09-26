"""Design, analytic-oracle, paired-inference and dependence checks."""
import json

import numpy as np
import pandas as pd
import pytest

from judgecal import stress_grid


@pytest.fixture(scope="module")
def small_grid():
    return stress_grid.run_stress_grid(repetitions=3, seed=31)


def test_full_cartesian_grid_has_no_omitted_conditions(small_grid):
    trials, summary, config = small_grid
    assert len(trials) == (36 + 3 * 2) * 4 * 3
    assert len(summary) == (36 + 3 * 2) * 4
    iid = [x for x in config["scenarios"] if x["inference_mode"] == "iid"]
    assert len(iid) == 36
    combinations = {(x["truth"], x["judge_profile"], x["audit_units"]) for x in iid}
    assert combinations == {
        (p, profile, n) for p in (.2, .5, .8)
        for profile in ("strong", "moderate", "uninformative", "anti")
        for n in (20, 50, 200)
    }
    assert not trials.duplicated(["scenario", "repetition", "method"]).any()
    assert trials.groupby(["scenario", "repetition"]).size().eq(4).all()
    assert trials.target_units.eq(10 * trials.audit_units).all()
    assert summary.repetitions.eq(3).all()
    assert json.loads(json.dumps(config, allow_nan=False)) == config


def test_determinism_and_optional_ablation_do_not_change_iid_draws(small_grid):
    trials, summary, config = small_grid
    again, again_summary, again_config = stress_grid.run_stress_grid(3, 31)
    pd.testing.assert_frame_equal(trials, again)
    pd.testing.assert_frame_equal(summary, again_summary)
    assert config == again_config
    iid, iid_summary, no_cluster_config = stress_grid.run_stress_grid(3, 31, False)
    pd.testing.assert_frame_equal(trials[trials.inference_mode == "iid"].reset_index(drop=True), iid)
    pd.testing.assert_frame_equal(summary[summary.inference_mode == "iid"].reset_index(drop=True), iid_summary)
    assert len(no_cluster_config["scenarios"]) == 36
    assert not no_cluster_config["include_cluster_ablation"]


def test_oracle_is_analytic_minimizer_of_independently_enumerated_variance(small_grid):
    _, _, config = small_grid
    for scenario in config["scenarios"]:
        if scenario["inference_mode"] != "iid":
            assert scenario["oracle_power"] is None
            continue
        p = scenario["truth"]
        se, sp = scenario["audit_sensitivity"], scenario["audit_specificity"]
        y, f = np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])
        probability = np.array([p * se, p * (1-se), (1-p) * (1-sp), (1-p) * sp])
        q = float(probability @ f)
        human_variance = float(probability @ ((y-p) ** 2))
        judge_variance = float(probability @ ((f-q) ** 2))
        covariance = float(probability @ ((y-p) * (f-q)))
        n, target_n = scenario["audit_units"], scenario["target_units"]

        def objective(power):
            return (human_variance / n + power**2 * judge_variance * (1/n + 1/target_n)
                    - 2 * power * covariance / n)

        oracle = scenario["oracle_power"]
        assert 0 <= oracle <= 1
        assert all(objective(oracle) <= objective(power) + 1e-15 for power in np.linspace(0, 1, 101))
        assert scenario["expected_raw_judge"] == pytest.approx(q)
        assert scenario["expected_human_only"] == pytest.approx(p)
        assert scenario["expected_fixed_ppi"] == pytest.approx(p)
        if scenario["judge_profile"] in ("anti", "uninformative"):
            assert oracle == pytest.approx(0)


def test_oracle_metadata_never_enters_fitted_estimates(monkeypatch, small_grid):
    trials, _, _ = small_grid
    monkeypatch.setattr(stress_grid, "_oracle_power", lambda _: .123456)
    modified, _, _ = stress_grid.run_stress_grid(3, 31)
    for column in ("point", "standard_error", "ci_low", "ci_high", "selected_power"):
        pd.testing.assert_series_equal(trials[column], modified[column])
    assert modified.loc[modified.inference_mode == "iid", "oracle_power"].eq(.123456).all()


def test_cluster_ablation_uses_same_draws_and_exposes_false_independence(small_grid):
    trials, _, config = small_grid
    clustered = trials[trials.inference_mode == "cluster"].set_index(["draw_group", "repetition", "method"])
    naive = trials[trials.inference_mode == "naive_iid"].set_index(["draw_group", "repetition", "method"])
    assert clustered.index.equals(naive.index)
    assert naive.validity_scope.str.startswith("assumption_violation").all()
    assert clustered.variance_method.eq("one-way-cluster-robust").all()
    assert naive.variance_method.eq("iid-sample-variance").all()
    for method in ("raw_judge", "human_only", "ppi"):
        correct = clustered.xs(method, level="method")
        incorrect = naive.xs(method, level="method")
        np.testing.assert_array_equal(correct.point, incorrect.point)
        assert (correct.standard_error >= incorrect.standard_error).all()
    # Exact repeat algebra: row-iid variance is too small by this factor.
    human_correct = clustered.xs("human_only", level="method")
    human_naive = naive.xs("human_only", level="method")
    factors = (4 * human_correct.audit_units - 1) / (human_correct.audit_units - 1)
    np.testing.assert_allclose(human_correct.standard_error**2,
                               human_naive.standard_error**2 * factors)
    clustered_configs = [s for s in config["scenarios"] if s["inference_mode"] != "iid"]
    assert len(clustered_configs) == 6
    assert all(s["turns_per_unit"] == 4 for s in clustered_configs)
    assert all(s["data_have_repeated_clusters"] for s in clustered_configs)
    assert all(s["grouped"] == (s["inference_mode"] == "cluster") for s in clustered_configs)


def test_paired_mcse_uses_matched_losses_not_independent_method_errors(small_grid):
    trials, _, _ = small_grid
    scenario = trials.scenario.iloc[0]
    selected = trials[(trials.scenario == scenario) & trials.method.isin(["human_only", "ppi"])].copy()
    for method, errors in [("human_only", [1., 2., 3.]), ("ppi", [2., 3., 4.])]:
        selected.loc[selected.method == method, "error"] = errors
    result = stress_grid._paired_summary(selected).set_index("method")
    ppi = result.loc["ppi"]
    human = result.loc["human_only"]
    assert ppi.mse_difference_vs_human == pytest.approx(5.)
    assert ppi.mse_difference_vs_human_mcse == pytest.approx(2 / np.sqrt(3))
    assert ppi.mae_difference_vs_human == pytest.approx(1.)
    assert ppi.mae_difference_vs_human_mcse == pytest.approx(0.)
    a, b = np.array([4., 9., 16.]), np.array([1., 4., 9.])
    rmse_a, rmse_b = np.sqrt(a.mean()), np.sqrt(b.mean())
    influence = a / (2 * rmse_a) - b / (2 * rmse_b)
    assert ppi.rmse_difference_vs_human == pytest.approx(rmse_a-rmse_b)
    assert ppi.rmse_difference_vs_human_mcse == pytest.approx(influence.std(ddof=1)/np.sqrt(3))
    naive_mcse = np.sqrt(a.var(ddof=1)/(4 * rmse_a**2 * 3)
                        + b.var(ddof=1)/(4 * rmse_b**2 * 3))
    assert ppi.rmse_difference_vs_human_mcse < naive_mcse
    assert human.mse_difference_vs_human == human.mse_difference_vs_human_mcse == 0
    assert human.rmse_difference_vs_human == human.rmse_difference_vs_human_mcse == 0


def test_zero_rmse_delta_method_is_explicitly_undefined_unless_losses_match(small_grid):
    trials, _, _ = small_grid
    selected = trials[trials.scenario == trials.scenario.iloc[0]].copy()
    selected.loc[selected.method == "human_only", "error"] = 0.
    selected.loc[selected.method == "ppi", "error"] = 1.
    result = stress_grid._paired_summary(selected).set_index("method")
    assert pd.isna(result.loc["ppi", "rmse_difference_vs_human_mcse"])
    assert result.loc["human_only", "rmse_difference_vs_human_mcse"] == 0


def test_retained_results_and_summary_match_their_declared_targets(small_grid):
    trials, summary, _ = small_grid
    assert np.isfinite(trials[["point", "standard_error", "ci_low", "ci_high", "error"]]).all().all()
    assert np.array_equal(trials.covered, (trials.ci_low <= trials.truth) & (trials.truth <= trials.ci_high))
    assert np.array_equal(trials.native_covered,
                          (trials.ci_low <= trials.native_truth) & (trials.native_truth <= trials.ci_high))
    raw = trials[trials.method == "raw_judge"]
    assert raw.audit_labels_used.eq(0).all()
    assert raw.selected_power.isna().all()
    assert trials.loc[trials.method != "raw_judge", "audit_labels_used"].eq(
        trials.loc[trials.method != "raw_judge", "n_labeled"]
    ).all()
    for row in summary.itertuples():
        group = trials[(trials.scenario == row.scenario) & (trials.method == row.method)]
        assert row.bias == pytest.approx(group.error.mean())
        assert row.rmse == pytest.approx(np.sqrt((group.error**2).mean()))
        assert row.coverage == pytest.approx(group.covered.mean())
        if row.method != "raw_judge":
            assert row.power_q50 == pytest.approx(group.selected_power.median())
            assert row.power_zero_fraction == pytest.approx(group.selected_power.eq(0).mean())


@pytest.mark.parametrize("kwargs", [
    {"repetitions": 1}, {"repetitions": -1}, {"repetitions": 2.5},
    {"repetitions": True}, {"seed": -1}, {"seed": 1.5}, {"seed": True},
    {"include_cluster_ablation": 1}, {"include_cluster_ablation": None},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        stress_grid.run_stress_grid(**kwargs)
