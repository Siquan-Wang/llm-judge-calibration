"""Target-specific errors, marginal coverage and paired Monte Carlo summaries."""
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from judgecal import estimand_study as study
from judgecal.intervals import _binomial_exact_interval
from judgecal.simulation import SimulationScenario


def test_complete_grid_targets_configuration_and_reproducibility():
    trials, summary = study.run_estimand_study(repetitions=2, seed=6)
    again, second_summary = study.run_estimand_study(repetitions=2, seed=6)
    pd.testing.assert_frame_equal(trials, again)
    pd.testing.assert_frame_equal(summary, second_summary)
    assert trials.attrs == again.attrs
    assert len(trials) == 36 * 2 * 4 and len(summary) == 36 * 4
    assert trials.groupby(["scenario", "repetition"]).size().eq(4).all()
    assert set(trials.method) == set(study.METHODS)
    assert trials.population_truth.equals(trials.prevalence)
    assert trials.n_unlabeled.eq(trials.audit_ratio * trials.n_labeled).all()
    assert set(trials.audit_ratio) == {.5, 1., 10.}
    assert not trials.pool_truth.equals(trials.population_truth)
    assert trials.loc[trials.method == "pool_tuned", "primary_target"].eq("realized_pool_mean").all()
    assert trials.loc[trials.method.str.endswith("tuned"), "interval_coefficient_handling"].str.contains("selected").all()
    config = trials.attrs["configuration"]
    assert len(config["scenarios"]) == 36 and config["repetitions"] == 2
    for row in config["scenarios"]:
        p, q = row["prevalence"], row["expected_raw_judge"]
        slope = p * (1 - p) * (row["audit_sensitivity"] + row["audit_specificity"] - 1) / (q * (1 - q))
        assert row["oracle_pool_power"] == pytest.approx(slope)
        assert row["oracle_population_power"] == pytest.approx(slope * row["audit_ratio"] / (1 + row["audit_ratio"]))
        # The analytic excess pool risk from using its population oracle is nonnegative.
        n, size = row["audit_units"], row["target_units"]
        def risk(power):
            return (1/n + 1/size) * (p*(1-p) - 2*power*slope*q*(1-q) + power**2*q*(1-q))
        expected_gap = (1/n + 1/size) * q*(1-q) * (row["oracle_population_power"] - slope)**2
        assert risk(row["oracle_population_power"]) - risk(slope) == pytest.approx(expected_gap, abs=1e-16)


def test_shared_inputs_and_chosen_coefficients_match_between_interval_apis(monkeypatch):
    calls = []
    original_population, original_pool = study.prediction_powered_mean, study.predict_heldout_mean
    def wrap(function, kind):
        def captured(f, y, u, **kwargs):
            result = function(f, y, u, **kwargs)
            calls.append((kind, f.copy(), y.copy(), u.copy(), result))
            return result
        return captured
    monkeypatch.setattr(study, "prediction_powered_mean", wrap(original_population, "population"))
    monkeypatch.setattr(study, "predict_heldout_mean", wrap(original_pool, "pool"))
    study.run_estimand_study(repetitions=2)
    assert len(calls) == 36 * 2 * 8
    for start in range(0, len(calls), 8):
        block = calls[start:start+8]
        for call in block:
            for first, other in zip(block[0][1:4], call[1:4]):
                np.testing.assert_array_equal(first, other)
        for call in block:
            matches = [other for other in block if other[0] != call[0]
                       and other[4].selected_power == call[4].selected_power]
            assert matches and all(other[4].point == call[4].point for other in matches)


def test_hidden_pool_outcomes_change_only_scoring_not_any_fitted_quantity(monkeypatch):
    before, _ = study.run_estimand_study(repetitions=2)
    draw = study._draw_pool
    def alter(rng, units, turns, prevalence, sensitivity, specificity, group_offset):
        y, f, g = draw(rng, units, turns, prevalence, sensitivity, specificity, group_offset)
        return (1-y if group_offset else y), f, g
    monkeypatch.setattr(study, "_draw_pool", alter)
    after, _ = study.run_estimand_study(repetitions=2)
    inference = ["point", "selected_power", "power_method", "population_variance", "pool_prediction_variance",
                 "population_ci_low", "population_ci_high", "pool_pi_low", "pool_pi_high",
                 "population_truth", "population_error", "population_covered"]
    pd.testing.assert_frame_equal(before[inference], after[inference])
    assert not before.pool_truth.equals(after.pool_truth)
    assert not before.pool_error.equals(after.pool_error)


def test_both_target_summaries_and_paired_mcse_are_recomputed_on_matched_draws():
    trials, summary = study.run_estimand_study(repetitions=5, seed=21, alpha=.1)
    indexed = summary.set_index(["scenario", "method"])
    for scenario, rows in trials.groupby("scenario"):
        for target in ("population", "pool"):
            errors = rows.pivot(index="repetition", columns="method", values=f"{target}_error")
            for method, group in rows.groupby("method"):
                row = indexed.loc[(scenario, method)]
                losses = errors[method]**2
                assert row[f"{target}_rmse"] == pytest.approx(np.sqrt(losses.mean()))
                assert row[f"{target}_bias"] == pytest.approx(errors[method].mean())
                assert row[f"{target}_mean_width"] == pytest.approx(group[f"{target}_interval_width"].mean())
                assert row[f"{target}_coverage"] == group[f"{target}_covered"].mean()
                ci = _binomial_exact_interval(int(group[f"{target}_covered"].sum()), 5, .05)
                assert row[f"{target}_coverage_mc_low"] == ci.low
                assert row[f"{target}_coverage_mc_high"] == ci.high
                for baseline in ("population_tuned", "pool_tuned"):
                    difference = losses - errors[baseline]**2
                    assert row[f"{target}_mse_difference_vs_{baseline}"] == pytest.approx(difference.mean())
                    assert row[f"{target}_mse_difference_vs_{baseline}_mcse"] == pytest.approx(difference.std(ddof=1)/np.sqrt(5))
    np.testing.assert_allclose(trials.population_error, trials.point - trials.population_truth)
    np.testing.assert_allclose(trials.pool_error, trials.point - trials.pool_truth)
    for target, low, high in (("population", "population_ci_low", "population_ci_high"),
                              ("pool", "pool_pi_low", "pool_pi_high")):
        truth = trials[f"{target}_truth"]
        scale = np.maximum.reduce([np.ones(len(trials)), abs(trials[low]), abs(trials[high]), abs(truth)])
        tolerance = 8 * np.finfo(float).eps * scale
        expected = (trials[low] - tolerance <= truth) & (truth <= trials[high] + tolerance)
        assert trials[f"{target}_covered"].equals(expected)


def test_perfect_proxy_demonstrates_different_point_error_targets(monkeypatch):
    scenario = SimulationScenario("perfect", .5, .5, 1., 1., 1., 1., 20, 10, 1, False, "iid_same_population")
    monkeypatch.setattr(study, "_conditions", lambda: iter([(scenario, "positive", .5)]))
    def draw(rng, units, turns, prevalence, sensitivity, specificity, group_offset):
        y = np.array([0., 1.] * (units//2))
        if group_offset:
            y[:2] = 1.  # Realized U mean .6 differs from population .5.
        return y, y.copy(), np.arange(group_offset, group_offset+units)
    monkeypatch.setattr(study, "_draw_pool", draw)
    trials, _ = study.run_estimand_study(repetitions=2)
    fixed = trials.loc[trials.method == "fixed_ppi"]
    assert fixed.pool_error.eq(0).all() and fixed.pool_prediction_variance.eq(0).all()
    assert fixed.population_error.gt(0).all() and fixed.population_variance.gt(0).all()
    human = trials.loc[trials.method == "human_only"]
    assert human.population_error.eq(0).all() and human.pool_error.lt(0).all()
    assert fixed.oracle_pool_power.eq(1).all()
    assert fixed.oracle_population_power.eq(1/3).all()


def test_coverage_rounding_guard_preserves_inverse_zero_width_endpoints(monkeypatch):
    scenario = SimulationScenario("inverse_rounding", .5, .5, 0., 0., 0., 0., 20, 10, 1, False, "iid_same_population")
    monkeypatch.setattr(study, "_conditions", lambda: iter([(scenario, "inverse", .5)]))
    def draw(rng, units, turns, prevalence, sensitivity, specificity, group_offset):
        positives = 6 if group_offset else 4
        y = np.r_[np.ones(positives), np.zeros(units-positives)]
        return y, 1-y, np.arange(group_offset, group_offset+units)
    monkeypatch.setattr(study, "_draw_pool", draw)
    trials, summary = study.run_estimand_study(repetitions=2)
    pool = trials.loc[trials.method == "pool_tuned"]
    assert pool.selected_power.eq(-1).all()
    assert pool.pool_prediction_variance.eq(0).all() and pool.pool_interval_width.eq(0).all()
    assert pool.pool_pi_low.eq(pool.point).all() and pool.pool_pi_high.eq(pool.point).all()
    assert pool.point.eq(.6000000000000001).all() and pool.pool_truth.eq(.6).all()
    assert pool.pool_error.gt(0).all()  # Retain the actual floating-point error.
    assert pool.pool_covered.all()
    assert summary.loc[summary.method == "pool_tuned", "pool_coverage"].eq(1).all()
    assert "Coverage scoring only" in trials.attrs["configuration"]["coverage_numerical_convention"]["scope"]


def test_coverage_guard_is_symmetric_but_does_not_hide_real_misses():
    epsilon = np.finfo(float).eps
    assert study._coverage_contains(.5, .5, .5-epsilon)
    assert study._coverage_contains(.5, .5, .5+epsilon)
    assert not study._coverage_contains(.5, .5, .5-16*epsilon)
    assert not study._coverage_contains(.5, .5, .5+16*epsilon)
    assert not study._coverage_contains(.5, .5, .500001)
    # The scale factor matters for untruncated bounds outside [0,1].
    assert study._coverage_contains(4., 4., 4.+4*epsilon)
    assert not study._coverage_contains(4., 4., 4.+64*epsilon)


@pytest.mark.parametrize("kwargs", [
    {"repetitions": 1}, {"repetitions": True}, {"repetitions": 2.5},
    {"seed": -1}, {"seed": True}, {"seed": 1.5}, {"alpha": 0},
    {"alpha": 1}, {"alpha": True}, {"alpha": "0.05"}, {"alpha": np.nan},
    {"alpha": Fraction(1, 10**1000)}, {"alpha": 10**1000},
])
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        study.run_estimand_study(**kwargs)
