"""Known-truth sensitivity study of signed versus nonnegative PPI power.

This retrospective extension evaluates an established mean-correction
principle under declared coefficient constraints; it is not a new estimator.
All intervals are asymptotic normal intervals, including small-audit cases.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd

from .intervals import _binomial_exact_interval
from .ppi import prediction_powered_mean
from .simulation import SimulationScenario, _draw_pool, _summarize_trials


PREVALENCES = (.5, .95)
PROFILES = (("positive", .95, .90), ("inverse", .05, .10), ("uninformative", .60, .40))
AUDIT_SIZES = (20, 200, 1000)
METHODS = ("human_only", "ppi", "ppi_tuned", "ppi_signed")


def _conditions():
    for prevalence in PREVALENCES:
        for profile, sensitivity, specificity in PROFILES:
            for count in AUDIT_SIZES:
                yield SimulationScenario(
                    f"iid_p{round(100 * prevalence):03d}_{profile}_n{count:04d}",
                    prevalence, prevalence, sensitivity, specificity,
                    sensitivity, specificity, count, 10 * count, 1, False,
                    "iid_same_population",
                ), profile


def _oracle(scenario: SimulationScenario) -> float:
    """Analytic iid coefficient; a diagnostic only, never fitted or selected."""
    p, q = scenario.target_prevalence, scenario.target_judge_rate
    covariance = p * (1 - p) * (scenario.target_sensitivity + scenario.target_specificity - 1)
    denominator = (1 + scenario.audit_units / scenario.target_units) * q * (1 - q)
    return float(covariance / denominator) if denominator > 0 else 0.


def _mcse(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _summarize_signed_trials(trials: pd.DataFrame) -> pd.DataFrame:
    summary = _summarize_trials(trials)
    extras = []
    for (scenario, method), group in trials.groupby(["scenario", "method"], sort=False):
        powers = group.selected_power
        coverage = _binomial_exact_interval(int(group.covered.sum()), len(group), .05)
        row = {
            "scenario": scenario, "method": method,
            "profile": group.profile.iloc[0], "prevalence": float(group.prevalence.iloc[0]),
            "oracle_power": float(group.oracle_power.iloc[0]),
            "oracle_positive_power": float(group.oracle_positive_power.iloc[0]),
            "power_lower": float(group.power_lower.iloc[0]),
            "power_upper": float(group.power_upper.iloc[0]),
            "power_method": group.power_method.iloc[0],
            "coverage_mc_low": coverage.low, "coverage_mc_high": coverage.high,
            "power_negative_fraction": float(powers.lt(0).mean()),
            "power_positive_fraction": float(powers.gt(0).mean()),
            "power_zero_fraction": float(powers.eq(0).mean()),
            "power_lower_boundary_fraction": float(powers.eq(group.power_lower).mean()),
            "power_upper_boundary_fraction": float(powers.eq(group.power_upper).mean()),
            "power_q10": float(powers.quantile(.1)),
            "power_q50": float(powers.quantile(.5)),
            "power_q90": float(powers.quantile(.9)),
            "point_outside_unit_interval_fraction": float(group.point_outside_unit_interval.mean()),
        }
        for baseline in ("human", "positive"):
            differences = group[f"squared_error_difference_vs_{baseline}"].to_numpy()
            row[f"mse_difference_vs_{baseline}"] = float(differences.mean())
            row[f"mse_difference_vs_{baseline}_mcse"] = _mcse(differences)
        extras.append(row)
    return summary.merge(pd.DataFrame(extras), on=["scenario", "method"],
                         sort=False, validate="one_to_one")


def signed_power_study(
    repetitions: int = 1000,
    seed: int = 2029,
    alpha: float = .05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare four normal estimators on all 18 declared iid binary laws.

    The grid crosses outcome prevalence .5/.95, positive/inverse/uninformative
    proxy laws, and audit sizes 20/200/1000, with independent prediction pools
    of size 10*n. Every method uses the same draw: audit-only, fixed power +1,
    automatic power constrained to [0,1], and automatic power in [-1,1].
    Only the audit outcomes enter fitting. The true target is the declared
    prevalence, never the realized prediction-pool outcome mean. Analytic
    oracle coefficients are metadata and never passed to fitted estimators.

    All estimates, untruncated intervals and replications remain in the
    output. Coverage assesses population prevalence in this simulation, not
    a real-corpus coverage claim. Normal intervals remain asymptotic even
    when the study exposes small-sample undercoverage or zero width.

    Per-trial squared-loss contrasts retain shared-draw pairing. Summary
    ``mse_difference_vs_positive`` subtracts the positive-range ``ppi_tuned``
    baseline, and ``mse_difference_vs_human`` subtracts ``human_only``.
    Negative means favor the named method. Their MCSE is the standard error
    of paired differences, not a difference of independent error estimates.
    Exact pointwise 95% binomial intervals additionally quantify uncertainty
    about coverage; their confidence level is independent of inference alpha.

    Returns trials and summary. Complete configuration is attached as
    ``trials.attrs['configuration']`` after aggregation. Monte Carlo errors
    concern independent synthetic replications, not deployment tasks.
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
    rows, configurations = [], []
    for (scenario, profile), stream in zip(conditions, streams):
        rng = np.random.default_rng(stream)
        oracle = _oracle(scenario)
        configurations.append({
            **scenario.configuration(), "profile": profile,
            "prevalence": scenario.target_prevalence,
            "oracle_power": oracle, "oracle_positive_power": float(np.clip(oracle, 0., 1.)),
            "oracle_power_role": "Analytic iid variance-minimizing diagnostic; never passed to estimator",
            "inference_mode": "iid", "intervals_clipped": False,
        })
        for repetition in range(repetitions):
            y_l, f_l, _ = _draw_pool(
                rng, scenario.audit_units, 1, scenario.audit_prevalence,
                scenario.audit_sensitivity, scenario.audit_specificity, 0,
            )
            _, f_u, _ = _draw_pool(
                rng, scenario.target_units, 1, scenario.target_prevalence,
                scenario.target_sensitivity, scenario.target_specificity, scenario.audit_units,
            )
            results = {
                method: prediction_powered_mean(f_l, y_l, f_u, alpha=alpha, power=power)
                for method, power in (("human_only", 0.), ("ppi", 1.), ("ppi_tuned", "auto"))
            }
            results["ppi_signed"] = prediction_powered_mean(
                f_l, y_l, f_u, alpha=alpha, power="auto", power_bounds=(-1., 1.),
            )
            losses = {method: (result.point - scenario.target_prevalence)**2
                      for method, result in results.items()}
            for method in METHODS:
                result = results[method]
                point, low, high = result.point, result.interval.low, result.interval.high
                error = point - scenario.target_prevalence
                rows.append({
                    "scenario": scenario.name, "profile": profile,
                    "prevalence": scenario.target_prevalence,
                    "repetition": repetition, "method": method,
                    "validity_scope": scenario.validity_scope,
                    "truth": scenario.target_prevalence, "native_truth": scenario.target_prevalence,
                    "interval_estimand": "population_binary_outcome_mean_under_iid_assumptions",
                    "point": point, "standard_error": result.standard_error,
                    "ci_low": low, "ci_high": high, "interval_width": high - low,
                    "error": error, "covered": low <= scenario.target_prevalence <= high,
                    "native_covered": low <= scenario.target_prevalence <= high,
                    "selected_power": result.selected_power, "power_method": result.power_method,
                    "power_lower": -1. if method == "ppi_signed" else 0., "power_upper": 1.,
                    "estimated_variance_ratio": result.estimated_variance_ratio,
                    "oracle_power": oracle, "oracle_positive_power": float(np.clip(oracle, 0., 1.)),
                    "point_outside_unit_interval": point < 0. or point > 1.,
                    "n_labeled": len(y_l), "n_unlabeled": len(f_u),
                    "audit_units": scenario.audit_units, "target_units": scenario.target_units,
                    "audit_labels_used": len(y_l), "variance_method": result.variance_method,
                    "alpha": alpha,
                    "squared_error_difference_vs_human": losses[method] - losses["human_only"],
                    "squared_error_difference_vs_positive": losses[method] - losses["ppi_tuned"],
                })
    trials = pd.DataFrame(rows)
    summary = _summarize_signed_trials(trials)
    trials.attrs["configuration"] = {
        "schema_version": 1, "study": "retrospective-signed-power-study-v1",
        "repetitions": repetitions, "seed": seed, "alpha": alpha,
        "coverage_mc_confidence_level": .95,
        "scenario_seed_scheme": "numpy.SeedSequence(seed).spawn in prevalence/profile/audit-size order",
        "prevalences": list(PREVALENCES), "audit_sizes": list(AUDIT_SIZES),
        "target_audit_ratio": 10,
        "profiles": {name: {"sensitivity": se, "specificity": sp} for name, se, sp in PROFILES},
        "methods": list(METHODS),
        "power_bounds": {method: [-1., 1.] if method == "ppi_signed" else [0., 1.] for method in METHODS},
        "paired_references": {"human": "human_only", "positive": "ppi_tuned"},
        "scenarios": configurations,
        "notes": [
            "Retrospective constraint sensitivity of established mean PPI; not a new estimator or preregistered validation.",
            "All methods share the same independent audit and prediction-only pools in each replication.",
            "Only audit outcomes fit power; prediction-pool outcomes are discarded and analytic prevalence scores coverage.",
            "Oracle coefficients describe the data-generating law and never enter fitted estimators.",
            "All intervals and points are untruncated; adverse results, zero widths and out-of-domain points are retained.",
            "Signed power complements the scalar proxy algebraically; it does not change outcomes or canonical label IDs.",
            "Sign rates use strict power<0, ==0, >0; boundary rates use equality to declared constraints.",
            "Normal intervals are asymptotic; empirical variance minimization does not guarantee realized MSE improvement.",
            "Paired MSE MCSE uses the standard deviation of within-draw squared-loss differences divided by sqrt(R).",
            "Exact binomial coverage intervals are pointwise 95% Monte Carlo intervals, not simultaneous across cells.",
        ],
    }
    return trials, summary
