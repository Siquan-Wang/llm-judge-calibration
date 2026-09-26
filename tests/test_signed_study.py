"""Paired known-truth signed-power experiments retain every draw and limitation."""
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from judgecal import signed_study as study
from judgecal.intervals import _binomial_exact_interval


def test_complete_grid_config_oracles_and_deterministic_outputs():
    trials, summary = study.signed_power_study(repetitions=3, seed=17)
    again, summary_again = study.signed_power_study(repetitions=3, seed=17)
    pd.testing.assert_frame_equal(trials, again)
    pd.testing.assert_frame_equal(summary, summary_again)
    assert trials.attrs == again.attrs
    assert len(trials) == 18 * 4 * 3 and len(summary) == 18 * 4
    assert set(trials.method) == set(study.METHODS)
    assert trials.groupby(["scenario", "repetition"]).size().eq(4).all()
    assert trials.truth.equals(trials.prevalence)
    assert trials.n_unlabeled.eq(10 * trials.n_labeled).all()
    config = trials.attrs["configuration"]
    assert len(config["scenarios"]) == 18
    assert config["repetitions"] == 3 and config["seed"] == 17
    assert config["power_bounds"]["ppi_signed"] == [-1., 1.]
    for row in config["scenarios"]:
        p, q = row["prevalence"], row["expected_raw_judge"]
        cov = p * (1 - p) * (row["audit_sensitivity"] + row["audit_specificity"] - 1)
        assert row["oracle_power"] == pytest.approx(cov / (1.1 * q * (1 - q)))
        assert abs(row["oracle_power"]) <= 10 / 11 + 1e-14
        assert row["expected_human_only"] == pytest.approx(p)
        assert row["expected_fixed_ppi"] == pytest.approx(p)
    assert trials.loc[trials.profile == "positive", "oracle_power"].gt(0).all()
    assert trials.loc[trials.profile == "inverse", "oracle_power"].lt(0).all()
    assert trials.loc[trials.profile == "uninformative", "oracle_power"].eq(0).all()


def test_all_methods_share_draws_and_oracle_is_not_passed_to_estimator(monkeypatch):
    original = study.prediction_powered_mean
    calls = []

    def capture(f, y, u, **kwargs):
        calls.append((f.copy(), y.copy(), u.copy(), kwargs))
        return original(f, y, u, **kwargs)

    monkeypatch.setattr(study, "prediction_powered_mean", capture)
    study.signed_power_study(repetitions=2)
    assert len(calls) == 18 * 2 * 4
    for start in range(0, len(calls), 4):
        reference = calls[start]
        for call in calls[start:start + 4]:
            for expected, actual in zip(reference[:3], call[:3]):
                np.testing.assert_array_equal(expected, actual)
        assert [c[3]["power"] for c in calls[start:start + 4]] == [0., 1., "auto", "auto"]
        assert calls[start + 3][3]["power_bounds"] == (-1., 1.)
        assert all("oracle" not in key for call in calls[start:start + 4] for key in call[3])


def test_prediction_pool_outcomes_are_not_used_as_targets_or_tuning(monkeypatch):
    before, summary = study.signed_power_study(repetitions=2)
    draw = study._draw_pool

    def change_hidden_outcomes(rng, units, turns, prevalence, sensitivity, specificity, group_offset):
        y, f, g = draw(rng, units, turns, prevalence, sensitivity, specificity, group_offset)
        return (1 - y if group_offset else y), f, g

    monkeypatch.setattr(study, "_draw_pool", change_hidden_outcomes)
    after, summary_after = study.signed_power_study(repetitions=2)
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(summary, summary_after)


def test_paired_loss_mcse_exact_coverage_and_sign_diagnostics():
    trials, summary = study.signed_power_study(repetitions=7, seed=51, alpha=.1)
    lookup = summary.set_index(["scenario", "method"])
    assert trials.alpha.eq(.1).all()
    for scenario, rows in trials.groupby("scenario"):
        errors = rows.pivot(index="repetition", columns="method", values="error")
        for method, group in rows.groupby("method"):
            group = group.sort_values("repetition")
            row = lookup.loc[(scenario, method)]
            for name, baseline in (("human", "human_only"), ("positive", "ppi_tuned")):
                differences = errors[method]**2 - errors[baseline]**2
                np.testing.assert_allclose(group[f"squared_error_difference_vs_{name}"], differences, atol=1e-16)
                assert row[f"mse_difference_vs_{name}"] == pytest.approx(differences.mean())
                assert row[f"mse_difference_vs_{name}_mcse"] == pytest.approx(differences.std(ddof=1) / np.sqrt(7))
            ci = _binomial_exact_interval(int(group.covered.sum()), len(group), .05)
            assert row.coverage_mc_low == ci.low and row.coverage_mc_high == ci.high
            if row.coverage == 1:
                assert row.coverage_mc_low < 1
            powers = group.selected_power
            assert row.power_negative_fraction == powers.lt(0).mean()
            assert row.power_lower_boundary_fraction == powers.eq(group.power_lower).mean()
            assert row.power_upper_boundary_fraction == powers.eq(group.power_upper).mean()
            assert row.point_outside_unit_interval_fraction == ((group.point < 0) | (group.point > 1)).mean()
            assert row.mean_width == pytest.approx((group.ci_high - group.ci_low).mean())
        # Estimated-Q dominance is algebraic; realized squared-error dominance is not required.
        se = rows.pivot(index="repetition", columns="method", values="standard_error")
        assert np.all(se.ppi_signed <= se.ppi_tuned + 1e-14)
        assert np.all(se.ppi_tuned <= se.human_only + 1e-14)


@pytest.mark.parametrize("kwargs", [
    {"repetitions": 1}, {"repetitions": True}, {"repetitions": 2.5},
    {"seed": -1}, {"seed": True}, {"seed": 1.5},
    {"alpha": 0}, {"alpha": 1}, {"alpha": True}, {"alpha": np.nan},
    {"alpha": "0.05"}, {"alpha": Fraction(1, 10**1000)}, {"alpha": 10**1000},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        study.signed_power_study(**kwargs)
