"""Frozen, known-truth stress tests for human-audit win-rate inference.

These synthetic experiments assess population targets, unlike the MT-Bench
held-out-label error experiment. They are not evidence about real annotation
savings. The two shift scenarios intentionally violate transfer assumptions;
their coverage measures failure against the target population, not a guarantee.

Repeated turns in clustered scenarios are exact copies of one cluster draw.
Cluster sizes are fixed and equal, so the observation-weighted human preference
target equals the declared Bernoulli prevalence. Pools have independent draws
and globally disjoint cluster IDs. All methods see the same generated pools.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.special import ndtri

from .ppi import prediction_powered_win_rate


@dataclass(frozen=True)
class SimulationScenario:
    """A completely specified binary human/judge data-generating process."""

    name: str
    audit_prevalence: float
    target_prevalence: float
    audit_sensitivity: float
    audit_specificity: float
    target_sensitivity: float
    target_specificity: float
    audit_units: int
    target_units: int
    turns_per_unit: int
    grouped: bool
    validity_scope: str

    @staticmethod
    def judge_rate(prevalence: float, sensitivity: float, specificity: float) -> float:
        return prevalence * sensitivity + (1.0 - prevalence) * (1.0 - specificity)

    @property
    def audit_judge_rate(self) -> float:
        return self.judge_rate(
            self.audit_prevalence, self.audit_sensitivity, self.audit_specificity
        )

    @property
    def target_judge_rate(self) -> float:
        return self.judge_rate(
            self.target_prevalence, self.target_sensitivity, self.target_specificity
        )

    def configuration(self) -> dict:
        result = asdict(self)
        result.update(
            truth=self.target_prevalence,
            expected_raw_judge=self.target_judge_rate,
            expected_human_only=self.audit_prevalence,
            expected_fixed_ppi=(
                self.target_judge_rate + self.audit_prevalence - self.audit_judge_rate
            ),
            n_labeled=self.audit_units * self.turns_per_unit,
            n_unlabeled=self.target_units * self.turns_per_unit,
        )
        return result


# Versioned retrospective configurations, informed by development and
# exploratory checks. This is not a preregistered study.
SIMULATION_SCENARIOS = (
    SimulationScenario(
        "iid_strong", .6, .6, .95, .9, .95, .9, 200, 2000, 1, False,
        "same_population_iid",
    ),
    SimulationScenario(
        "iid_weak", .6, .6, .6, .5, .6, .5, 200, 2000, 1, False,
        "same_population_iid",
    ),
    SimulationScenario(
        "iid_anticorrelated", .6, .6, .1, .15, .1, .15, 200, 2000, 1, False,
        "same_population_iid",
    ),
    SimulationScenario(
        "clustered_repeated_turns", .6, .6, .95, .9, .95, .9, 50, 500, 4, True,
        "same_population_independent_clusters",
    ),
    SimulationScenario(
        "few_clusters", .6, .6, .95, .9, .95, .9, 8, 80, 4, True,
        "same_population_small_cluster_normal_stress",
    ),
    SimulationScenario(
        "prevalence_shift", .4, .75, .9, .8, .9, .8, 200, 2000, 1, False,
        "assumption_violation_prevalence_shift_stable_confusion",
    ),
    SimulationScenario(
        "conditional_error_shift", .6, .6, .95, .9, .6, .55, 200, 2000, 1, False,
        "assumption_violation_conditional_judge_error_shift",
    ),
)


def _draw_pool(
    rng: np.random.Generator,
    units: int,
    turns: int,
    prevalence: float,
    sensitivity: float,
    specificity: float,
    group_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample independent units, then repeat their labels within each unit."""
    human = rng.binomial(1, prevalence, size=units)
    judge_probability = np.where(human == 1, sensitivity, 1.0 - specificity)
    judge = rng.binomial(1, judge_probability)
    groups = np.repeat(np.arange(group_offset, group_offset + units), turns)
    return np.repeat(human, turns), np.repeat(judge, turns), groups


def _mean_variance(values: np.ndarray, groups: np.ndarray | None) -> float:
    if groups is None:
        return float(np.var(values, ddof=1) / len(values))
    _, codes = np.unique(groups, return_inverse=True)
    sums = np.bincount(codes, weights=values - values.mean())
    count = len(sums)
    return float(count / (count - 1) * np.dot(sums, sums) / len(values) ** 2)


def _mcse(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, method), group in trials.groupby(["scenario", "method"], sort=False):
        errors = group["error"].to_numpy()
        squared_errors = errors ** 2
        rmse = float(np.sqrt(squared_errors.mean()))
        coverage = float(group["covered"].mean())
        native_coverage = float(group["native_covered"].mean())
        width = group["interval_width"].to_numpy()
        repetitions = len(group)
        rows.append({
            "scenario": scenario,
            "method": method,
            "validity_scope": group["validity_scope"].iloc[0],
            "interval_estimand": group["interval_estimand"].iloc[0],
            "truth": float(group["truth"].iloc[0]),
            "native_truth": float(group["native_truth"].iloc[0]),
            "repetitions": repetitions,
            "n_labeled": int(group["n_labeled"].iloc[0]),
            "n_unlabeled": int(group["n_unlabeled"].iloc[0]),
            "audit_units": int(group["audit_units"].iloc[0]),
            "target_units": int(group["target_units"].iloc[0]),
            "audit_labels_used": int(group["audit_labels_used"].iloc[0]),
            "bias": float(errors.mean()),
            "bias_mcse": _mcse(errors),
            "rmse": rmse,
            "rmse_mcse": _mcse(squared_errors) / (2.0 * rmse) if rmse else 0.0,
            "mean_width": float(width.mean()),
            "mean_width_mcse": _mcse(width),
            "coverage": coverage,
            "coverage_mcse": float(np.sqrt(coverage * (1.0 - coverage) / repetitions)),
            "native_coverage": native_coverage,
            "native_coverage_mcse": float(np.sqrt(
                native_coverage * (1.0 - native_coverage) / repetitions
            )),
            "mean_standard_error": float(group["standard_error"].mean()),
            "empirical_standard_deviation": float(group["point"].std(ddof=1)),
            "mean_selected_power": float(group["selected_power"].mean()),
            "mean_estimated_variance_ratio": float(group["estimated_variance_ratio"].mean()),
            "undefined_variance_ratio_draws": int(group["estimated_variance_ratio"].isna().sum()),
            "zero_width_draws": int((group["interval_width"] == 0.0).sum()),
        })
    return pd.DataFrame(rows)


def run_simulation_study(
    repetitions: int = 1000, seed: int = 2026
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return ``(trials, summary, configuration)`` for all frozen scenarios.

    ``covered`` and summary ``coverage`` always concern the exact *target
    human* prevalence. A raw judge interval natively estimates the target
    judge-positive rate, so its native target and coverage are also retained.
    Human-only intervals natively cover the audit population's human rate;
    this differs from the target population in the prevalence-shift scenario.
    Raw judging uses zero human labels; the other three methods receive the
    same complete audit and target-judge pool, including tuning's label cost.

    Human-only is the public PPI API at power=0, original PPI at power=1,
    and tuned PPI at power='auto'. No observations or repetitions are dropped,
    and no estimates/intervals are clipped. Intervals use alpha=.05. At least
    two repetitions are required for empirical Monte Carlo standard errors;
    they describe simulation noise, not real-world generalization uncertainty.
    """
    for name, value, minimum in (("repetitions", repetitions, 2), ("seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    repetitions, seed = int(repetitions), int(seed)
    alpha = .05
    quantile = float(ndtri(1.0 - alpha / 2.0))
    scenario_seeds = np.random.SeedSequence(seed).spawn(len(SIMULATION_SCENARIOS))
    rows = []
    for scenario, scenario_seed in zip(SIMULATION_SCENARIOS, scenario_seeds):
        rng = np.random.default_rng(scenario_seed)
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
            groups = ({"labeled_groups": g_l, "unlabeled_groups": g_u}
                      if scenario.grouped else {})
            labels_l = np.where(f_l == 1, "A", "B")
            labels_h = np.where(y_l == 1, "A", "B")
            labels_u = np.where(f_u == 1, "A", "B")
            estimates = {}
            for method, power in (("human_only", 0.0), ("ppi", 1.0), ("ppi_tuned", "auto")):
                result = prediction_powered_win_rate(
                    labels_l, labels_h, labels_u, alpha=alpha, power=power, **groups
                )
                estimates[method] = (
                    result.point, result.standard_error, result.interval.low,
                    result.interval.high, result.selected_power, result.estimated_variance_ratio,
                )
            raw_point = float(f_u.mean())
            raw_se = float(np.sqrt(_mean_variance(f_u, g_u if scenario.grouped else None)))
            # A raw judge interval does not estimate human preference, even
            # when it coincidentally contains the target human rate.
            estimates = {"raw_judge": (
                raw_point, raw_se, raw_point - quantile * raw_se,
                raw_point + quantile * raw_se, np.nan, np.nan,
            ), **estimates}
            for method, (point, se, low, high, power, variance_ratio) in estimates.items():
                native_truth = (scenario.target_judge_rate if method == "raw_judge"
                                else scenario.audit_prevalence if method == "human_only"
                                else scenario.target_prevalence)
                rows.append({
                    "scenario": scenario.name,
                    "repetition": repetition,
                    "method": method,
                    "validity_scope": scenario.validity_scope,
                    "truth": scenario.target_prevalence,
                    "native_truth": native_truth,
                    "interval_estimand": ("target_judge_positive_rate" if method == "raw_judge"
                                          else "audit_human_preference_rate" if method == "human_only"
                                          else "target_human_preference_rate_under_assumptions"),
                    "point": float(point),
                    "standard_error": float(se),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "interval_width": float(high - low),
                    "error": float(point - scenario.target_prevalence),
                    "covered": bool(low <= scenario.target_prevalence <= high),
                    "native_covered": bool(low <= native_truth <= high),
                    "selected_power": float(power),
                    "estimated_variance_ratio": (float(variance_ratio)
                                                 if variance_ratio is not None else np.nan),
                    "n_labeled": len(y_l),
                    "n_unlabeled": len(f_u),
                    "audit_units": scenario.audit_units,
                    "target_units": scenario.target_units,
                    "audit_labels_used": 0 if method == "raw_judge" else len(y_l),
                    "variance_method": "one-way-cluster-robust" if scenario.grouped else "iid-sample-variance",
                    "alpha": alpha,
                })
    trials = pd.DataFrame(rows)
    configuration = {
        "schema_version": 1,
        "repetitions": repetitions,
        "seed": seed,
        "alpha": alpha,
        "scenario_seed_scheme": "numpy.SeedSequence(seed).spawn in frozen scenario order",
        "methods": ["raw_judge", "human_only", "ppi", "ppi_tuned"],
        "scenarios": [scenario.configuration() for scenario in SIMULATION_SCENARIOS],
        "notes": [
            "All human and judge labels are binary; no ties are generated in this suite.",
            "Versioned retrospective study; configurations reflect development and exploratory checks, not preregistration.",
            "Target truth is analytic population prevalence, not a realized held-out mean.",
            "Clustered turns repeat one draw exactly; fixed equal sizes preserve observation-weighted truth.",
            "All methods share the same available audit budget; raw_judge uses no human labels.",
            "Raw judge intervals natively cover judge prevalence, not human preference.",
            "Human-only intervals natively cover audit prevalence; target coverage can fail under prevalence shift.",
            "Shift scenarios deliberately violate transfer assumptions and are diagnostic failure cases.",
            "Coverage and bias are not guaranteed; asymptotic normal failures and all draws are retained.",
            "MCSEs are plug-in estimates; values of zero at empirical coverage 0 or 1 do not imply certainty.",
            "NaN power/variance ratio denotes inapplicable raw metadata or an undefined zero-denominator ratio.",
        ],
    }
    return trials, _summarize_trials(trials), configuration
