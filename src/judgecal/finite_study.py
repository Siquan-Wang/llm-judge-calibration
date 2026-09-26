"""Known-truth coverage/width study for normal and finite-sample intervals.

This is a retrospective follow-up, not a new interval construction. Bounds
are obtained exclusively from the public inference APIs. The small iid grid
targets low audit counts, imbalanced outcomes, weak predictions and perfect
predictions. Optional dependence and shift cases are deliberately outside
the intended-target finite-sample guarantee and are labeled as such.
"""
from __future__ import annotations

import json
from numbers import Integral

import numpy as np
import pandas as pd

from .intervals import _binomial_exact_interval
from .ppi import prediction_powered_mean
from .simulation import SimulationScenario, _draw_pool, _summarize_trials


PROFILES = (
    ("balanced_strong", .5, .95, .90),
    ("rare_strong", .95, .95, .90),
    ("balanced_uninformative", .5, .60, .40),
    ("balanced_perfect", .5, 1., 1.),
)
AUDIT_SIZES = (20, 200, 1000)
POWER_GRID = (0., .25, .5, .75, 1.)
METHOD_SPECS = (
    ("human_normal", "normal", 0.),
    ("ppi_normal", "normal", 1.),
    ("ppi_tuned_normal", "normal", "auto"),
    ("human_hoeffding", "hoeffding", 0.),
    ("ppi_hoeffding", "hoeffding", 1.),
    ("human_eb", "empirical_bernstein", 0.),
    ("ppi_eb", "empirical_bernstein", 1.),
    ("ppi_eb_grid", "empirical_bernstein", POWER_GRID),
)
HUMAN_BASELINES = {
    "normal": "human_normal", "hoeffding": "human_hoeffding",
    "empirical_bernstein": "human_eb",
}


def _conditions(include_stress_cases: bool):
    for profile, prevalence, sensitivity, specificity in PROFILES:
        for count in AUDIT_SIZES:
            yield SimulationScenario(
                f"iid_{profile}_n{count:04d}", prevalence, prevalence,
                sensitivity, specificity, sensitivity, specificity,
                count, 10 * count, 1, False, "iid_same_population",
            ), profile, "iid"
    if include_stress_cases:
        yield SimulationScenario(
            "violation_repeated_clusters", .5, .5, .95, .90, .95, .90,
            50, 500, 4, False, "assumption_violation_repeated_clusters_analyzed_as_iid",
        ), "balanced_strong", "dependence_violation"
        yield SimulationScenario(
            "violation_audit_target_shift", .4, .75, .90, .80, .90, .80,
            1000, 10000, 1, False, "assumption_violation_audit_target_prevalence_shift",
        ), "shift_stable_conditional_errors", "shift_violation"


def _mcse(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _summarize_finite_trials(trials: pd.DataFrame) -> pd.DataFrame:
    summary = _summarize_trials(trials)
    extras = []
    for (scenario, method), group in trials.groupby(["scenario", "method"], sort=False):
        # Replications are independent even when rows within a draw are not.
        coverage_interval = _binomial_exact_interval(int(group.covered.sum()), len(group), .05)
        native = group.native_covered.dropna()
        native_interval = (_binomial_exact_interval(int(native.sum()), len(native), .05)
                           if len(native) else None)
        powers = group.selected_power
        extras.append({
            "scenario": scenario, "method": method,
            "profile": group.profile.iloc[0],
            "scenario_family": group.scenario_family.iloc[0],
            "interval_family": group.interval_family.iloc[0],
            "power_rule": group.power_rule.iloc[0],
            "theorem_applicable": bool(group.theorem_applicable.iloc[0]),
            "native_theorem_applicable": group.native_theorem_applicable.iloc[0],
            "native_estimand": group.native_estimand.iloc[0],
            "paired_width_reference": group.paired_width_reference.iloc[0],
            "width_difference_vs_human_family": float(group.width_difference_vs_human_family.mean()),
            "width_difference_vs_human_family_mcse": _mcse(group.width_difference_vs_human_family.to_numpy()),
            "fraction_width_above_one": float(group.interval_width.gt(1).mean()),
            "fraction_full_domain_covered": float(group.full_domain_covered.mean()),
            "coverage_mc_low": coverage_interval.low,
            "coverage_mc_high": coverage_interval.high,
            "native_coverage_mc_low": np.nan if native_interval is None else native_interval.low,
            "native_coverage_mc_high": np.nan if native_interval is None else native_interval.high,
            "native_coverage_repetitions": len(native),
            "candidate_count": int(group.candidate_count.iloc[0]),
            "candidate_powers_json": group.candidate_powers_json.iloc[0],
            "power_q10": float(powers.quantile(.1)),
            "power_q50": float(powers.quantile(.5)),
            "power_q90": float(powers.quantile(.9)),
            "power_zero_fraction": float(powers.eq(0).mean()),
            "power_one_fraction": float(powers.eq(1).mean()),
        })
    result = summary.merge(pd.DataFrame(extras), on=["scenario", "method"],
                           how="left", sort=False, validate="one_to_one")
    result["native_theorem_applicable"] = result["native_theorem_applicable"].astype(object)
    return result


def run_finite_sample_study(
    repetitions: int = 1000,
    seed: int = 2028,
    include_stress_cases: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compare eight interval methods on shared known-truth binary draws.

    Returns every trial, per-scenario/method summaries, and complete generator
    configuration. The 12 iid cells cross four profiles with audit sizes
    20/200/1000; prediction-only pools contain ten times as many observations.
    Two optional stress cells reuse the same APIs on dependent repeated rows
    and on differing audit/target prevalences, with explicit scope flags.

    ``covered`` always tests the analytic target human prevalence, never a
    realized test-set mean. ``theorem_applicable`` refers specifically to a
    finite-sample theorem for that target; it is false for normal intervals
    and both violation cells. Human intervals additionally record coverage
    of audit prevalence. For nonhuman methods under shift, native rectified
    functional coverage is deliberately omitted (NaN), since it is a
    different target and can depend on a selected coefficient. Its omission
    does not assert invalidity of component bounds for that functional.

    All intervals remain untruncated. Paired width differences subtract the
    same-draw human baseline in the corresponding interval family. Exact
    binomial 95% Monte Carlo intervals describe uncertainty about simulated
    coverage; they are pointwise across cells, not simultaneous guarantees.
    Candidate powers/radii are JSON-array strings in trial CSV-ready tables.
    Normal intervals have no finite candidate family and therefore use empty
    arrays. Finite intervals have no reported asymptotic standard error.
    """
    # Keep configuration importable while inference remains owned by its API.
    from .finite_sample import finite_sample_mean

    for name, value, minimum in (("repetitions", repetitions, 2), ("seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    if not isinstance(include_stress_cases, (bool, np.bool_)):
        raise ValueError("include_stress_cases must be a boolean")
    repetitions, seed = int(repetitions), int(seed)
    conditions = list(_conditions(bool(include_stress_cases)))
    streams = np.random.SeedSequence(seed).spawn(len(conditions))
    alpha = .05
    rows, scenarios = [], []
    for (scenario, profile, scenario_family), stream in zip(conditions, streams):
        rng = np.random.default_rng(stream)
        scenarios.append({
            **scenario.configuration(), "profile": profile,
            "scenario_family": scenario_family,
            "data_have_repeated_clusters": scenario.turns_per_unit > 1,
            "inference_mode": "iid", "intervals_clipped": False,
        })
        for repetition in range(repetitions):
            y_l, f_l, _ = _draw_pool(
                rng, scenario.audit_units, scenario.turns_per_unit,
                scenario.audit_prevalence, scenario.audit_sensitivity,
                scenario.audit_specificity, 0,
            )
            _, f_u, _ = _draw_pool(
                rng, scenario.target_units, scenario.turns_per_unit,
                scenario.target_prevalence, scenario.target_sensitivity,
                scenario.target_specificity, scenario.audit_units,
            )
            estimates = {}
            for method, family, power in METHOD_SPECS:
                if family == "normal":
                    result = prediction_powered_mean(f_l, y_l, f_u, alpha=alpha, power=power)
                    estimates[method] = {
                        "point": result.point, "standard_error": result.standard_error,
                        "ci_low": result.interval.low, "ci_high": result.interval.high,
                        "selected_power": result.selected_power,
                        "estimated_variance_ratio": (np.nan if result.estimated_variance_ratio is None
                                                     else result.estimated_variance_ratio),
                        "power_rule": "audit_tuned" if power == "auto" else "fixed",
                        "candidate_powers_json": "[]", "candidate_radii_json": "[]",
                        "candidate_points_json": "[]",
                        "candidate_count": 0,
                    }
                else:
                    result = finite_sample_mean(f_l, y_l, f_u, alpha=alpha, method=family, power=power)
                    estimates[method] = {
                        "point": result.point, "standard_error": np.nan,
                        "ci_low": result.interval.low, "ci_high": result.interval.high,
                        "selected_power": result.selected_power, "estimated_variance_ratio": np.nan,
                        "power_rule": "simultaneous_grid" if isinstance(power, tuple) else "fixed",
                        "candidate_powers_json": json.dumps(list(result.candidate_powers), separators=(",", ":")),
                        "candidate_radii_json": json.dumps(list(result.candidate_radii), separators=(",", ":")),
                        "candidate_points_json": json.dumps(list(result.candidate_points), separators=(",", ":")),
                        "candidate_count": len(result.candidate_powers),
                    }
            for method, family, _ in METHOD_SPECS:
                result = estimates[method]
                human = method.startswith("human_")
                native_reported = scenario_family != "shift_violation" or human
                native_truth = (scenario.audit_prevalence if human else scenario.target_prevalence)
                native_estimand = ("audit_human_preference_rate" if human else
                                   "target_human_preference_rate" if native_reported else
                                   "not_reported_for_shifted_rectified_functional")
                is_finite = family != "normal"
                native_theorem = (
                    bool(is_finite and scenario_family != "dependence_violation") if native_reported else None
                )
                low, high = result["ci_low"], result["ci_high"]
                width = high - low
                baseline = HUMAN_BASELINES[family]
                baseline_width = estimates[baseline]["ci_high"] - estimates[baseline]["ci_low"]
                rows.append({
                    **result,
                    "scenario": scenario.name, "profile": profile,
                    "scenario_family": scenario_family, "repetition": repetition,
                    "method": method, "interval_family": family,
                    "validity_scope": scenario.validity_scope,
                    "theorem_applicable": bool(is_finite and scenario_family == "iid"),
                    "native_theorem_applicable": native_theorem,
                    "truth": scenario.target_prevalence,
                    "native_truth": native_truth if native_reported else np.nan,
                    "native_estimand": native_estimand,
                    "interval_estimand": ("audit_human_preference_rate" if human else
                                          "target_human_preference_rate_under_assumptions"),
                    "interval_width": width,
                    "full_domain_covered": bool(low <= 0 and high >= 1),
                    "error": result["point"] - scenario.target_prevalence,
                    "covered": bool(low <= scenario.target_prevalence <= high),
                    "native_covered": bool(low <= native_truth <= high) if native_reported else np.nan,
                    "paired_width_reference": baseline,
                    "width_difference_vs_human_family": width - baseline_width,
                    "n_labeled": len(y_l), "n_unlabeled": len(f_u),
                    "audit_units": scenario.audit_units, "target_units": scenario.target_units,
                    "audit_labels_used": len(y_l),
                    "variance_method": ("iid-sample-variance" if family == "normal" else
                                        "bounded-range-concentration" if family == "hoeffding" else
                                        "empirical-variance-concentration"),
                    "alpha": alpha,
                })
    trials = pd.DataFrame(rows)
    # Keep nullable native diagnostics schema-stable when stress cases are
    # omitted; no inference or random draw depends on these object columns.
    for column in ("native_theorem_applicable", "native_covered"):
        trials[column] = trials[column].astype(object)
    configuration = {
        "schema_version": 1, "study": "retrospective-finite-sample-interval-study-v1",
        "repetitions": repetitions, "seed": seed, "alpha": alpha,
        "include_stress_cases": bool(include_stress_cases),
        "methods": [name for name, _, _ in METHOD_SPECS],
        "power_grid": list(POWER_GRID), "scenarios": scenarios,
        "finite_interval_api": "judgecal.finite_sample_mean",
        "normal_interval_api": "judgecal.prediction_powered_mean",
        "scenario_seed_scheme": "numpy.SeedSequence(seed).spawn in declared scenario order",
        "monte_carlo_coverage_interval": {"method": "clopper-pearson", "alpha": .05,
                                          "scope": "pointwise per scenario/method, not simultaneous"},
        "notes": [
            "Retrospective follow-up; no new estimator, theorem or preregistration claim.",
            "All methods share the exact same generated audit and prediction-only samples.",
            "The core public APIs compute all intervals; this study does not duplicate bound formulas.",
            "All trials and untruncated intervals are retained, including zero width or width above one.",
            "Width above one differs from containing the full [0,1] domain; both diagnostics are reported.",
            "Theorem flags refer specifically to finite-sample intended-target coverage, not asymptotic validity.",
            "Dependence and shift cases are explicitly outside the intended-human-target guarantee.",
            "Human native intervals under iid shift concern audit prevalence, not target prevalence.",
            "Nonhuman native coverage under shift is omitted; component bounds can still cover a different rectified functional.",
            "Finite-grid powers are specified before each draw and selected by the public simultaneous-bound API.",
            "Normal methods have no finite-grid candidates; finite methods have no asymptotic standard-error diagnostic.",
            "Paired width differences subtract the same-draw human interval in the same interval family.",
            "Coverage MCSE and exact-binomial intervals measure simulation noise, not real-data generalization uncertainty.",
            "The rare_strong profile has positive prevalence .95; negative outcomes are rare.",
        ],
    }
    return trials, _summarize_finite_trials(trials), configuration
