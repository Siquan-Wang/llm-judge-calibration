"""Separate population-mean inference from prediction of a random pool mean.

The point formula is shared, but its optimal coefficient and uncertainty
depend on the target. This is a retrospective estimand sensitivity study,
not a correction to the existing population API or a new inference theorem.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd

from .intervals import _binomial_exact_interval
from .pool_prediction import predict_heldout_mean
from .ppi import prediction_powered_mean
from .simulation import SimulationScenario, _draw_pool


PREVALENCES = (.5, .95)
PROFILES = (("positive", .95, .90), ("inverse", .05, .10), ("uninformative", .60, .40))
AUDIT_SIZES = (20, 200)
RATIOS = (.5, 1., 10.)
METHODS = ("human_only", "fixed_ppi", "population_tuned", "pool_tuned")
POINT_RULES = {"human_only": "fixed_zero", "fixed_ppi": "fixed_one",
               "population_tuned": "population_variance_tuned",
               "pool_tuned": "audit_residual_variance_tuned"}
POPULATION_SCOPE = "asymptotic confidence interval for the population outcome mean"
POOL_SCOPE = "asymptotic marginal prediction interval for the random held-out mean"


def _coverage_contains(low: float, high: float, truth: float) -> bool:
    """Score numerical membership without modifying an inference endpoint."""
    tolerance = 8 * np.finfo(float).eps * max(1., abs(low), abs(high), abs(truth))
    return bool(low - tolerance <= truth <= high + tolerance)


def _conditions():
    for prevalence in PREVALENCES:
        for profile, sensitivity, specificity in PROFILES:
            for count in AUDIT_SIZES:
                for ratio in RATIOS:
                    yield SimulationScenario(
                        f"iid_p{round(100 * prevalence):03d}_{profile}_n{count:04d}_r{round(10 * ratio):03d}",
                        prevalence, prevalence, sensitivity, specificity, sensitivity, specificity,
                        count, int(round(count * ratio)), 1, False, "iid_same_population",
                    ), profile, ratio


def _oracles(scenario: SimulationScenario) -> tuple[float, float]:
    p, q = scenario.target_prevalence, scenario.target_judge_rate
    covariance = p * (1 - p) * (scenario.target_sensitivity + scenario.target_specificity - 1)
    slope = covariance / (q * (1 - q)) if 0 < q < 1 else 0.
    population = slope * scenario.target_units / (scenario.audit_units + scenario.target_units)
    return float(np.clip(population, -1., 1.)), float(np.clip(slope, -1., 1.))


def _mcse(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _summarize_estimand_trials(trials: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (scenario, method), group in trials.groupby(["scenario", "method"], sort=False):
        powers = group.selected_power
        record = {
            "scenario": scenario, "method": method, "profile": group.profile.iloc[0],
            "prevalence": float(group.prevalence.iloc[0]),
            "audit_ratio": float(group.audit_ratio.iloc[0]),
            "n_labeled": int(group.n_labeled.iloc[0]), "n_unlabeled": int(group.n_unlabeled.iloc[0]),
            "repetitions": len(group), "alpha": float(group.alpha.iloc[0]),
            "point_rule": group.point_rule.iloc[0], "primary_target": group.primary_target.iloc[0],
            "interval_coefficient_handling": group.interval_coefficient_handling.iloc[0],
            "population_interval_scope": POPULATION_SCOPE, "pool_interval_scope": POOL_SCOPE,
            "oracle_population_power": float(group.oracle_population_power.iloc[0]),
            "oracle_pool_power": float(group.oracle_pool_power.iloc[0]),
            "mean_selected_power": float(powers.mean()), "power_lower": -1., "power_upper": 1.,
            "power_negative_fraction": float(powers.lt(0).mean()),
            "power_zero_fraction": float(powers.eq(0).mean()),
            "power_positive_fraction": float(powers.gt(0).mean()),
            "power_lower_boundary_fraction": float(powers.eq(-1.).mean()),
            "power_upper_boundary_fraction": float(powers.eq(1.).mean()),
            "point_outside_unit_interval_fraction": float(group.point_outside_unit_interval.mean()),
        }
        for target in ("population", "pool"):
            errors = group[f"{target}_error"].to_numpy()
            mse = float(np.mean(errors**2))
            rmse = float(np.sqrt(mse))
            width = group[f"{target}_interval_width"].to_numpy()
            covered = group[f"{target}_covered"]
            coverage = float(covered.mean())
            mc_interval = _binomial_exact_interval(int(covered.sum()), len(group), .05)
            record.update({
                f"{target}_bias": float(errors.mean()), f"{target}_bias_mcse": _mcse(errors),
                f"{target}_mse": mse, f"{target}_mse_mcse": _mcse(errors**2),
                f"{target}_rmse": rmse,
                f"{target}_rmse_mcse": _mcse(errors**2) / (2 * rmse) if rmse else 0.,
                f"{target}_mean_width": float(width.mean()), f"{target}_mean_width_mcse": _mcse(width),
                f"{target}_coverage": coverage,
                f"{target}_coverage_mcse": float(np.sqrt(coverage * (1 - coverage) / len(group))),
                f"{target}_coverage_mc_low": mc_interval.low, f"{target}_coverage_mc_high": mc_interval.high,
                f"{target}_zero_width_draws": int(np.sum(width == 0)),
            })
            for baseline in ("population_tuned", "pool_tuned"):
                differences = group[f"{target}_squared_error_difference_vs_{baseline}"].to_numpy()
                record[f"{target}_mse_difference_vs_{baseline}"] = float(differences.mean())
                record[f"{target}_mse_difference_vs_{baseline}_mcse"] = _mcse(differences)
        records.append(record)
    return pd.DataFrame(records)


def run_estimand_study(
    repetitions: int = 1000, seed: int = 2030, alpha: float = .05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return 36 shared-draw scenarios with two explicitly matched targets.

    Four coefficient rules use the same audit outcomes/proxies and separate
    prediction-pool proxies. Target-pool outcomes are held aside until all
    points, coefficients and intervals have been computed. Each point is
    scored against both the analytic population prevalence and the realized
    pool outcome mean. These losses are different and must be compared only
    within a named target, not ranked across unlike estimands.

    For every point, the population mean API supplies its matched confidence
    interval and the iid pool API supplies its matched marginal prediction
    interval. Passing a selected coefficient to an alternate fixed-power API
    does not erase selection: metadata explicitly retains same-audit tuning.
    Both intervals are plug-in asymptotic normal constructions, not exact or
    distribution-free guarantees. No mismatched cross-target coverage is used.

    The grid crosses p=.5/.95, positive/inverse/uninformative binary proxies,
    audit n=20/200, and N/n=.5/1/10. ``audit_ratio`` always means N/n, not n/N.
    All coefficients use [-1,1]. Diagnostic oracle coefficients never enter
    estimation. All failures, zero widths and out-of-domain points remain.

    Summaries have one row per scenario/method with separate population_ and
    pool_ metrics and paired squared-loss contrasts. Coverage Monte Carlo
    intervals are exact pointwise 95% binomial intervals; MCSEs describe
    independent simulation replications, not deployment uncertainty. Full
    configuration is attached as trials.attrs['configuration'] after summary.
    """
    for name, value, minimum in (("repetitions", repetitions, 2), ("seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real)
            or not 0 < alpha < 1 or not np.isfinite(float(alpha)) or not 0 < float(alpha) < 1):
        raise ValueError("alpha must be a finite real number strictly between zero and one")
    repetitions, seed, alpha = int(repetitions), int(seed), float(alpha)
    conditions = list(_conditions())
    streams = np.random.SeedSequence(seed).spawn(len(conditions))
    records, scenarios = [], []
    for (scenario, profile, ratio), stream in zip(conditions, streams):
        rng = np.random.default_rng(stream)
        oracle_population, oracle_pool = _oracles(scenario)
        scenarios.append({**scenario.configuration(), "profile": profile,
                          "prevalence": scenario.target_prevalence, "audit_ratio": ratio,
                          "oracle_population_power": oracle_population, "oracle_pool_power": oracle_pool,
                          "oracle_role": "Analytic diagnostic only, never passed to a fitted method"})
        for repetition in range(repetitions):
            y_l, f_l, _ = _draw_pool(rng, scenario.audit_units, 1, scenario.audit_prevalence,
                                    scenario.audit_sensitivity, scenario.audit_specificity, 0)
            y_u, f_u, _ = _draw_pool(rng, scenario.target_units, 1, scenario.target_prevalence,
                                    scenario.target_sensitivity, scenario.target_specificity, scenario.audit_units)
            args = {"alpha": alpha, "power_bounds": (-1., 1.)}
            population_results = {
                method: prediction_powered_mean(f_l, y_l, f_u, power=power, **args)
                for method, power in (("human_only", 0.), ("fixed_ppi", 1.), ("population_tuned", "auto"))
            }
            pool_native = predict_heldout_mean(f_l, y_l, f_u, power="auto", **args)
            population_results["pool_tuned"] = prediction_powered_mean(
                f_l, y_l, f_u, power=pool_native.selected_power, **args)
            pool_results = {
                method: predict_heldout_mean(f_l, y_l, f_u, power=result.selected_power, **args)
                for method, result in population_results.items() if method != "pool_tuned"
            }
            pool_results["pool_tuned"] = pool_native
            # Only after every method is formed do held-out outcomes enter scoring.
            pool_truth = float(y_u.mean())
            population_truth = scenario.target_prevalence
            losses = {
                target: {method: (result.point - truth)**2 for method, result in population_results.items()}
                for target, truth in (("population", population_truth), ("pool", pool_truth))
            }
            for method in METHODS:
                population, pool = population_results[method], pool_results[method]
                native = pool_native if method == "pool_tuned" else population
                population_interval, pool_interval = population.interval, pool.prediction_interval
                records.append({
                    "scenario": scenario.name, "profile": profile, "prevalence": scenario.target_prevalence,
                    "audit_ratio": ratio, "repetition": repetition, "method": method,
                    "point": population.point, "point_rule": POINT_RULES[method],
                    "primary_target": "realized_pool_mean" if method == "pool_tuned" else "population_mean",
                    "selected_power": native.selected_power, "power_method": native.power_method,
                    "power_lower": -1., "power_upper": 1.,
                    "interval_coefficient_handling": ("same-audit selected coefficient plugged into both intervals"
                                                       if method.endswith("tuned") else "fixed prespecified coefficient"),
                    "population_interval_scope": POPULATION_SCOPE, "pool_interval_scope": POOL_SCOPE,
                    "population_truth": population_truth, "pool_truth": pool_truth,
                    "population_error": population.point - population_truth,
                    "pool_error": population.point - pool_truth,
                    "population_variance": population.standard_error**2,
                    "pool_prediction_variance": pool.prediction_standard_error**2,
                    "population_standard_error": population.standard_error,
                    "pool_prediction_standard_error": pool.prediction_standard_error,
                    "population_ci_low": population_interval.low, "population_ci_high": population_interval.high,
                    "pool_pi_low": pool_interval.low, "pool_pi_high": pool_interval.high,
                    "population_interval_width": population_interval.high - population_interval.low,
                    "pool_interval_width": pool_interval.high - pool_interval.low,
                    "population_covered": _coverage_contains(population_interval.low, population_interval.high, population_truth),
                    "pool_covered": _coverage_contains(pool_interval.low, pool_interval.high, pool_truth),
                    "oracle_population_power": oracle_population, "oracle_pool_power": oracle_pool,
                    "point_outside_unit_interval": population.point < 0 or population.point > 1,
                    "n_labeled": len(y_l), "n_unlabeled": len(f_u), "alpha": alpha,
                    **{f"{target}_squared_error_difference_vs_{baseline}": losses[target][method] - losses[target][baseline]
                       for target in ("population", "pool") for baseline in ("population_tuned", "pool_tuned")},
                })
    trials = pd.DataFrame(records)
    summary = _summarize_estimand_trials(trials)
    trials.attrs["configuration"] = {
        "schema_version": 1, "study": "retrospective-estimand-study-v1",
        "repetitions": repetitions, "seed": seed, "alpha": alpha, "coverage_mc_confidence_level": .95,
        "prevalences": list(PREVALENCES), "audit_sizes": list(AUDIT_SIZES), "audit_ratios": list(RATIOS),
        "audit_ratio_definition": "Prediction observations divided by audit observations (N/n)",
        "profiles": {name: {"sensitivity": se, "specificity": sp} for name, se, sp in PROFILES},
        "methods": list(METHODS), "power_bounds": [-1., 1.], "scenarios": scenarios,
        "scenario_seed_scheme": "numpy.SeedSequence(seed).spawn in prevalence/profile/audit-size/ratio order",
        "population_interval_scope": POPULATION_SCOPE, "pool_interval_scope": POOL_SCOPE,
        "coverage_numerical_convention": {
            "comparison": "low - tolerance <= truth <= high + tolerance",
            "tolerance": "8 * float64_epsilon * max(1, abs(low), abs(high), abs(truth))",
            "float64_epsilon": float(np.finfo(float).eps),
            "scope": "Coverage scoring only, both targets and all methods; inference endpoints, widths, variances, points and losses unchanged",
        },
        "notes": [
            "Retrospective target sensitivity of an established correction; not a repair to the population API or a new theorem.",
            "All methods share each draw; held-out outcomes are unavailable until post-inference scoring.",
            "Oracle population and pool coefficients are diagnostic only and never supplied to fitted methods.",
            "Human-only and fixed PPI retain their conventional population primary target; fixed coefficients can be evaluated for both targets.",
            "Every point has a matched population CI and marginal random-pool prediction interval; no mismatched coverage is reported.",
            "Selected coefficients are plugged into asymptotic intervals; calling a fixed-power API does not make selection fixed in repeated sampling.",
            "Pool coverage is marginal over both iid pools, not conditional on every observed proxy pool or frozen corpus.",
            "All draws, losses, zero widths and out-of-domain points are retained without clipping.",
            "Coverage alone allows the documented machine-rounding margin; strict membership can be reconstructed from the unmodified saved endpoints and truths.",
            "Paired MSE differences compare methods within the same target and draw; never compare unlike-target RMSE as superiority.",
            "Exact coverage Monte Carlo intervals are pointwise 95% intervals, not simultaneous claims across cells.",
        ],
    }
    return trials, summary
