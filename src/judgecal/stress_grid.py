"""Retrospective synthetic sensitivity grid and dependence ablation.

This broadens the original study across prevalence, judge quality, and audit
size. It implements no new estimator. Known-truth oracle powers are diagnostic
metadata only: every tuned estimate selects its power from observed pools.
The optional clustered ablation applies correct and deliberately invalid iid
inference to the exact same repeated-turn draws, retaining all outcomes.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from scipy.special import ndtri

from .ppi import prediction_powered_win_rate
from .simulation import SimulationScenario, _draw_pool, _mean_variance, _summarize_trials


PREVALENCES = (.2, .5, .8)
JUDGE_PROFILES = (
    ("strong", .95, .90),
    ("moderate", .90, .60),
    ("uninformative", .60, .40),
    ("anti", .20, .20),
)
AUDIT_UNITS = (20, 50, 200)
CLUSTER_UNITS = (8, 30, 100)
METHODS = ("raw_judge", "human_only", "ppi", "ppi_tuned")


def _conditions(include_cluster_ablation: bool):
    for prevalence in PREVALENCES:
        for profile, sensitivity, specificity in JUDGE_PROFILES:
            for count in AUDIT_UNITS:
                name = f"iid_p{int(100 * prevalence):03d}_{profile}_n{count:03d}"
                yield SimulationScenario(
                    name, prevalence, prevalence, sensitivity, specificity,
                    sensitivity, specificity, count, 10 * count, 1, False,
                    "same_population_iid",
                ), profile
    if include_cluster_ablation:
        for count in CLUSTER_UNITS:
            yield SimulationScenario(
                f"cluster_g{count:03d}", .6, .6, .95, .90, .95, .90,
                count, 10 * count, 4, True,
                "same_population_independent_clusters_normal_approximation",
            ), "strong"


def _oracle_power(scenario: SimulationScenario) -> float:
    """Constrained iid oracle from population moments; never used to infer."""
    p = scenario.audit_prevalence
    q = scenario.audit_judge_rate
    covariance = p * (1 - p) * (
        scenario.audit_sensitivity + scenario.audit_specificity - 1
    )
    denominator = q * (1 - q) * (1 + scenario.audit_units / scenario.target_units)
    return float(np.clip(covariance / denominator, 0, 1)) if denominator > 0 else 0.0


def _mcse(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _paired_summary(trials: pd.DataFrame) -> pd.DataFrame:
    """Summarize within-draw loss differences, preserving method covariance."""
    summary = _summarize_trials(trials)
    paired = []
    for scenario, scenario_trials in trials.groupby("scenario", sort=False):
        human = scenario_trials.loc[scenario_trials.method == "human_only"].set_index("repetition")
        for method, group in scenario_trials.groupby("method", sort=False):
            ordered = group.set_index("repetition").sort_index()
            reference = human.loc[ordered.index]
            error = ordered.error.to_numpy()
            human_error = reference.error.to_numpy()
            loss = error ** 2
            human_loss = human_error ** 2
            mse_difference = loss - human_loss
            mae_difference = np.abs(error) - np.abs(human_error)
            rmse = float(np.sqrt(loss.mean()))
            human_rmse = float(np.sqrt(human_loss.mean()))
            if np.array_equal(loss, human_loss):
                rmse_difference_mcse = 0.0
            elif rmse > 0 and human_rmse > 0:
                # Joint delta-method influence values retain paired covariance.
                rmse_difference_mcse = _mcse(
                    loss / (2 * rmse) - human_loss / (2 * human_rmse)
                )
            else:
                # The square-root derivative is undefined at zero MSE.
                rmse_difference_mcse = np.nan
            powers = ordered.selected_power.dropna()
            paired.append({
                "scenario": scenario, "method": method,
                "draw_group": ordered.draw_group.iloc[0],
                "inference_mode": ordered.inference_mode.iloc[0],
                "judge_profile": ordered.judge_profile.iloc[0],
                "oracle_power": float(ordered.oracle_power.iloc[0]),
                "mse_difference_vs_human": float(mse_difference.mean()),
                "mse_difference_vs_human_mcse": _mcse(mse_difference),
                "mae_difference_vs_human": float(mae_difference.mean()),
                "mae_difference_vs_human_mcse": _mcse(mae_difference),
                "rmse_difference_vs_human": rmse - human_rmse,
                "rmse_difference_vs_human_mcse": rmse_difference_mcse,
                "power_q10": float(powers.quantile(.1)) if len(powers) else np.nan,
                "power_q50": float(powers.quantile(.5)) if len(powers) else np.nan,
                "power_q90": float(powers.quantile(.9)) if len(powers) else np.nan,
                "power_zero_fraction": float(powers.eq(0).mean()) if len(powers) else np.nan,
                "power_one_fraction": float(powers.eq(1).mean()) if len(powers) else np.nan,
            })
    return summary.merge(pd.DataFrame(paired), on=["scenario", "method"],
                         how="left", sort=False, validate="one_to_one")


def run_stress_grid(
    repetitions: int = 500,
    seed: int = 2027,
    include_cluster_ablation: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return every trial, summary, and full retrospective configuration.

    The iid grid crosses three human prevalences, four judge profiles, and
    three audit sizes (36 conditions); target sample size is ten times the
    audit size. All pools within a condition have the same population law.
    Optional ablations use 8/30/100 independent units, each repeated four
    times, with correct clustered and deliberately invalid row-iid inference
    on identical draws. These two settings have distinct scenario names but
    the same ``draw_group`` and repetition ID.

    Output columns extend ``run_simulation_study`` with ``draw_group``,
    ``inference_mode``, ``judge_profile`` and iid ``oracle_power`` diagnostics.
    Summary loss differences subtract human-only error on the same draws;
    negative differences favor the named method. Their Monte Carlo standard
    errors retain pairing; they are not standard errors across real tasks.
    RMSE-difference MCSE uses the joint delta method and is undefined (NaN)
    at a zero estimated MSE, unless the two loss vectors coincide exactly.

    Coverage always refers to target human prevalence; raw judge intervals
    also retain their native judge-prevalence coverage. Deliberately invalid
    iid intervals in the cluster ablation are labeled explicitly. No draw,
    adverse result, zero-width interval, or selected power is discarded.
    """
    for name, value, minimum in (("repetitions", repetitions, 2), ("seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    if not isinstance(include_cluster_ablation, (bool, np.bool_)):
        raise ValueError("include_cluster_ablation must be a boolean")
    repetitions, seed = int(repetitions), int(seed)
    conditions = list(_conditions(bool(include_cluster_ablation)))
    streams = np.random.SeedSequence(seed).spawn(len(conditions))
    alpha = .05
    quantile = float(ndtri(1 - alpha / 2))
    records, configurations = [], []
    for (scenario, profile), stream in zip(conditions, streams):
        rng = np.random.default_rng(stream)
        modes = ("cluster", "naive_iid") if scenario.grouped else ("iid",)
        oracle = None if scenario.grouped else _oracle_power(scenario)
        settings = []
        for mode in modes:
            name = (f"{scenario.name}_correct" if mode == "cluster" else
                    f"{scenario.name}_naive_iid" if mode == "naive_iid" else scenario.name)
            scope = ("assumption_violation_iid_inference_on_repeated_clusters"
                     if mode == "naive_iid" else scenario.validity_scope)
            setting = {**scenario.configuration(), "name": name,
                       "draw_group": scenario.name, "inference_mode": mode,
                       "judge_profile": profile, "validity_scope": scope,
                       "judge_quality": "anticorrelated" if profile == "anti" else profile,
                       "oracle_power": oracle,
                       "oracle_power_role": "analytic iid diagnostic only; never passed to estimator",
                       "grouped": mode == "cluster",
                       "data_have_repeated_clusters": scenario.grouped,
                       "inference_grouped": mode == "cluster"}
            configurations.append(setting)
            settings.append((name, mode, scope))
        for repetition in range(repetitions):
            y_l, f_l, g_l = _draw_pool(
                rng, scenario.audit_units, scenario.turns_per_unit,
                scenario.audit_prevalence, scenario.audit_sensitivity,
                scenario.audit_specificity, 0,
            )
            _, f_u, g_u = _draw_pool(
                rng, scenario.target_units, scenario.turns_per_unit,
                scenario.target_prevalence, scenario.target_sensitivity,
                scenario.target_specificity, scenario.audit_units,
            )
            labels_l = np.where(f_l == 1, "A", "B")
            labels_h = np.where(y_l == 1, "A", "B")
            labels_u = np.where(f_u == 1, "A", "B")
            for name, mode, scope in settings:
                groups = ({"labeled_groups": g_l, "unlabeled_groups": g_u}
                          if mode == "cluster" else {})
                raw_point = float(f_u.mean())
                raw_se = float(np.sqrt(_mean_variance(f_u, g_u if mode == "cluster" else None)))
                estimates = {"raw_judge": (
                    raw_point, raw_se, raw_point - quantile * raw_se,
                    raw_point + quantile * raw_se, np.nan, np.nan,
                )}
                for method, power in (("human_only", 0.0), ("ppi", 1.0), ("ppi_tuned", "auto")):
                    result = prediction_powered_win_rate(
                        labels_l, labels_h, labels_u, alpha=alpha, power=power, **groups
                    )
                    estimates[method] = (
                        result.point, result.standard_error, result.interval.low,
                        result.interval.high, result.selected_power, result.estimated_variance_ratio,
                    )
                for method, (point, se, low, high, power, ratio) in estimates.items():
                    native_truth = (scenario.target_judge_rate if method == "raw_judge"
                                    else scenario.target_prevalence)
                    records.append({
                        "scenario": name, "draw_group": scenario.name,
                        "inference_mode": mode, "judge_profile": profile,
                        "oracle_power": np.nan if oracle is None else oracle,
                        "repetition": repetition, "method": method,
                        "validity_scope": scope,
                        "truth": scenario.target_prevalence, "native_truth": native_truth,
                        "interval_estimand": ("target_judge_positive_rate" if method == "raw_judge"
                                              else "audit_human_preference_rate" if method == "human_only"
                                              else "target_human_preference_rate_under_assumptions"),
                        "point": float(point), "standard_error": float(se),
                        "ci_low": float(low), "ci_high": float(high),
                        "interval_width": float(high - low),
                        "error": float(point - scenario.target_prevalence),
                        "covered": bool(low <= scenario.target_prevalence <= high),
                        "native_covered": bool(low <= native_truth <= high),
                        "selected_power": float(power),
                        "estimated_variance_ratio": np.nan if ratio is None else float(ratio),
                        "n_labeled": len(y_l), "n_unlabeled": len(f_u),
                        "audit_units": scenario.audit_units, "target_units": scenario.target_units,
                        "audit_labels_used": 0 if method == "raw_judge" else len(y_l),
                        "variance_method": ("one-way-cluster-robust" if mode == "cluster"
                                            else "iid-sample-variance"),
                        "alpha": alpha,
                    })
    trials = pd.DataFrame(records)
    config = {
        "schema_version": 1, "study": "retrospective-stress-grid-v1",
        "repetitions": repetitions, "seed": seed, "alpha": alpha,
        "include_cluster_ablation": bool(include_cluster_ablation),
        "iid_grid": {"prevalences": list(PREVALENCES),
                     "judges": {name: {"sensitivity": se, "specificity": sp}
                                for name, se, sp in JUDGE_PROFILES},
                     "audit_units": list(AUDIT_UNITS), "target_audit_ratio": 10},
        "scenario_seed_scheme": "numpy.SeedSequence(seed).spawn in declared draw-group order",
        "methods": list(METHODS), "scenarios": configurations,
        "notes": [
            "Retrospective follow-up to the initial case study; not preregistered or a new estimator.",
            "All conditions, adverse outcomes, and zero-width intervals are retained.",
            "Analytic oracle power is diagnostic only and is never used by any fitted method.",
            "Raw intervals target judge prevalence; target-human containment is a bias diagnostic.",
            "Cluster ablations repeat each independent binary draw exactly four times.",
            "Correct and naive cluster settings share exact draws; naive row-iid inference deliberately violates independence.",
            "Paired loss differences use matched repetitions; negative values favor the named method over human-only.",
            "RMSE-difference MCSE is a paired delta-method approximation, undefined at zero MSE except identical loss vectors.",
            "Coverage MCSE is plug-in; empirical zero or one coverage does not imply certainty.",
            "Oracle metadata is null outside iid settings; NaN table fields mark inapplicable or undefined diagnostics.",
            "Repeated synthetic labels are not independently purchased annotations; audit_labels_used counts observed rows.",
        ],
    }
    return trials, _paired_summary(trials), config
