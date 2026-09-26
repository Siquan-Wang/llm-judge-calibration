"""Independent-pilot tuning with its labels charged to a fixed total budget.

This retrospective IID simulation composes existing mean-inference APIs.
Sample splitting, scalar power tuning and bounded-mean inequalities are
established methods; no new estimator or coverage theorem is proposed.
"""
from __future__ import annotations

import json
from numbers import Integral

import numpy as np
import pandas as pd

from .estimand_study import _coverage_contains
from .finite_sample import finite_sample_mean
from .intervals import _binomial_exact_interval
from .ppi import _bounded_scores, _inference_options, prediction_powered_mean


# Codes, unlike tuple positions or Python hashes, are stable RNG identities.
PROFILES = (
    ("balanced_strong", 1, .5, .95, .90),
    ("boundary_strong", 2, .95, .95, .90),
    ("uninformative", 3, .5, .60, .40),
    ("perfect", 4, .5, 1., 1.),
)
TOTAL_BUDGETS = (20, 200, 1000)
POWER_GRID = (0., .25, .5, .75, 1.)
METHODS = (
    "human_normal", "human_eb", "same_audit_normal", "same_audit_grid_eb",
    "pilot20_normal", "pilot20_eb", "pilot50_normal", "pilot50_eb",
)
POINT_METHODS = (
    "human_normal", "same_audit_normal", "same_audit_grid_eb",
    "pilot20_normal", "pilot50_normal",
)


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def pilot_power(f_p, y_p, n_inference: int, n_unlabeled: int) -> dict:
    """Choose [0,1] power using an independent pilot and design counts only.

    The common-marginal plug-in is Cov_P(Y,F)/((1+m/N)*Var_P(F)).
    Pilot labels are not part of the subsequent m-row residual mean. Exact
    constant proxies select zero. Binary inputs use exact integer sufficient
    statistics, preserving zero covariance and the scalar EB zero-power
    allocation. Fractional inputs use a scaled proxy without an arbitrary
    small-variance cutoff; diagnostics remain in original score units
    (extreme squares may underflow as floats).
    This helper does not access inference observations or population truth.
    """
    f_p = _bounded_scores(f_p, "f_p")
    y_p = _bounded_scores(y_p, "y_p")
    if f_p.size != y_p.size:
        raise ValueError("f_p and y_p must have equal lengths")
    m = _integer(n_inference, "n_inference", 2)
    count_u = _integer(n_unlabeled, "n_unlabeled", 2)
    constant = bool(np.all(f_p == f_p[0]))
    covariance = variance = selected = 0.
    binary = bool(np.all((f_p == 0) | (f_p == 1)) and np.all((y_p == 0) | (y_p == 1)))
    if not constant and binary:
        count = len(f_p)
        ones_f, ones_y = int(np.count_nonzero(f_p)), int(np.count_nonzero(y_p))
        joint_ones = int(np.count_nonzero((f_p == 1) & (y_p == 1)))
        covariance_numerator = count*joint_ones - ones_y*ones_f
        variance_numerator = ones_f*(count-ones_f)
        covariance = covariance_numerator / (count*(count-1))
        variance = variance_numerator / (count*(count-1))
        numerator = covariance_numerator*count_u
        denominator = variance_numerator*(m+count_u)
        selected = 0. if numerator <= 0 else 1. if numerator >= denominator else numerator/denominator
    elif not constant:
        delta = f_p - f_p[0]
        scale = float(np.max(np.abs(delta)))
        scaled = delta / scale
        centered = scaled - scaled.mean()
        y_centered = (np.zeros_like(y_p) if np.all(y_p == y_p[0]) else y_p-y_p.mean())
        variance_scaled = float(np.dot(centered, centered) / (len(f_p)-1))
        covariance_scaled = float(np.dot(centered, y_centered) / (len(f_p)-1))
        variance = variance_scaled * scale * scale
        covariance = covariance_scaled * scale
        # Apply the m/N factor before projection. Comparing the scaled slope
        # with the score scale also avoids overflowing a huge raw slope.
        scaled_power = (covariance_scaled / variance_scaled) * (count_u / (m+count_u))
        selected = 0. if scaled_power <= 0 else 1. if scaled_power >= scale else scaled_power/scale
    return {
        "selected_power": float(selected), "covariance": float(covariance),
        "variance": float(variance), "constant_proxy": constant,
        "n_pilot": int(len(f_p)), "n_inference": m, "n_unlabeled": count_u,
        "power_method": "independent-pilot-common-marginal-variance",
    }


def evaluate_draw(y_l, f_l, f_u, alpha: float = .05) -> list[dict]:
    """Apply eight procedures to one shared draw, without population truth.

    The B labeled pairs are IID; the independent proxy-only pool has the
    same proxy marginal. Each pilot is a fixed prefix, its inference audit
    the disjoint suffix. All B reference labels count toward every method's
    budget. Arbitrary bounded fractional scores are supported; no grouping
    or weights are accepted. At least ten audit rows give both pilots and
    both inference audits at least two observations.

    Normal/EB pilot variants use exactly the same fitted power and point.
    Scalar EB is valid conditional on the independent pilot; normal coverage
    remains asymptotic. No hidden outcomes or final-pool moments enter power
    fitting. Errors and coverage are added only by the study's scoring step.
    """
    y_l = _bounded_scores(y_l, "y_l")
    f_l = _bounded_scores(f_l, "f_l")
    f_u = _bounded_scores(f_u, "f_u")
    alpha, _ = _inference_options(alpha, 0.)
    if len(y_l) != len(f_l):
        raise ValueError("y_l and f_l must have equal lengths")
    total = len(y_l)
    if total < 10:
        raise ValueError("the total audit must contain at least ten observations")
    pilots = {
        fraction: pilot_power(f_l[:k], y_l[:k], total-k, len(f_u))
        for fraction, k in ((20, total//5), (50, total//2))
    }
    specs = (
        ("human_normal", "normal", 0., None),
        ("human_eb", "empirical_bernstein", 0., None),
        ("same_audit_normal", "normal", "auto", None),
        ("same_audit_grid_eb", "empirical_bernstein", POWER_GRID, None),
        ("pilot20_normal", "normal", pilots[20]["selected_power"], pilots[20]),
        ("pilot20_eb", "empirical_bernstein", pilots[20]["selected_power"], pilots[20]),
        ("pilot50_normal", "normal", pilots[50]["selected_power"], pilots[50]),
        ("pilot50_eb", "empirical_bernstein", pilots[50]["selected_power"], pilots[50]),
    )
    records = []
    for method, family, power, pilot in specs:
        k = 0 if pilot is None else pilot["n_pilot"]
        y_i, f_i = y_l[k:], f_l[k:]
        if family == "normal":
            result = prediction_powered_mean(f_i, y_i, f_u, alpha=alpha, power=power)
            se = result.standard_error
            candidates = {"candidate_count": 0, "candidate_powers_json": "[]",
                          "candidate_radii_json": "[]", "candidate_points_json": "[]"}
        else:
            result = finite_sample_mean(f_i, y_i, f_u, alpha=alpha, method=family, power=power)
            se = np.nan
            candidates = {
                "candidate_count": len(result.candidate_powers),
                **{f"candidate_{key}_json": json.dumps(list(getattr(result, f"candidate_{key}")),
                                                       separators=(",", ":"))
                   for key in ("powers", "radii", "points")},
            }
        low, high = result.interval.low, result.interval.high
        tuning = ("independent_pilot" if pilot is not None else
                  "same_audit_auto" if method == "same_audit_normal" else
                  "same_audit_simultaneous_grid" if method == "same_audit_grid_eb" else "fixed_zero")
        records.append({
            "method": method, "interval_family": family, "tuning_source": tuning,
            "total_budget": total, "n_pilot": k, "n_labeled": total-k,
            "n_unlabeled": len(f_u), "total_labels_used": total,
            "unique_prediction_scores_used": 0 if method.startswith("human_") else total+len(f_u),
            "proxy_scores_supplied": total+len(f_u),
            "selected_power": result.selected_power,
            "power_method": result.power_method if pilot is None else pilot["power_method"],
            "pilot_covariance": np.nan if pilot is None else pilot["covariance"],
            "pilot_variance": np.nan if pilot is None else pilot["variance"],
            "pilot_constant": False if pilot is None else pilot["constant_proxy"],
            "point": result.point, "standard_error": se, "estimated_variance": se*se,
            "interval_low": low, "interval_high": high, "width": high-low,
            "zero_width": bool(high == low), "full_domain": bool(low <= 0 and high >= 1),
            "point_outside_unit_interval": bool(result.point < 0 or result.point > 1),
            "theorem_applicable": family == "empirical_bernstein", "alpha": alpha,
            **candidates,
        })
    return records


def _draw(seed: int, profile_code: int, total: int, repetition: int,
          prevalence: float, sensitivity: float, specificity: float):
    """Stable streams: adding/reordering cells or extending repetitions is safe."""
    audit_rng, proxy_rng = (
        np.random.Generator(np.random.Philox(np.random.SeedSequence(
            [seed, profile_code, total, repetition, stream]))) for stream in (0, 1)
    )
    y_l = audit_rng.binomial(1, prevalence, size=total)
    f_l = audit_rng.binomial(1, np.where(y_l == 1, sensitivity, 1-specificity))
    proxy_mean = prevalence*sensitivity + (1-prevalence)*(1-specificity)
    f_u = proxy_rng.binomial(1, proxy_mean, size=10*total)
    return y_l, f_l, f_u


def _conditional_variance(power: float, m: int, count_u: int,
                          prevalence: float, sensitivity: float, specificity: float) -> float:
    """Known-law diagnostic, valid for a fixed/independently selected power only."""
    probabilities = np.array([(1-prevalence)*specificity, (1-prevalence)*(1-specificity),
                              prevalence*(1-sensitivity), prevalence*sensitivity])
    residual = np.array([0., -power, 1., 1-power])
    residual_mean = float(probabilities @ residual)
    residual_variance = float(probabilities @ ((residual-residual_mean)**2))
    proxy_mean = prevalence*sensitivity + (1-prevalence)*(1-specificity)
    return residual_variance/m + power**2*proxy_mean*(1-proxy_mean)/count_u


def _mcse(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)/np.sqrt(len(values)))


def _identity(group: pd.DataFrame) -> dict:
    names = ("scenario", "profile", "total_budget", "truth", "method", "interval_family",
             "n_pilot", "n_labeled", "n_unlabeled", "total_labels_used",
             "unique_prediction_scores_used", "proxy_scores_supplied", "theorem_applicable", "alpha")
    return {name: group[name].iloc[0] for name in names}


def _summarize(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in trials.groupby(["scenario", "method"], sort=False):
        error = group.error.to_numpy()
        loss = group.squared_error.to_numpy()
        count = len(group)
        coverage = float(group.covered.mean())
        interval = _binomial_exact_interval(int(group.covered.sum()), count, .05)
        rmse = float(np.sqrt(loss.mean()))
        variance = group.estimated_variance.to_numpy()
        conditional = group.conditional_variance.to_numpy()
        rows.append({
            **_identity(group), "repetitions": count,
            "bias": float(error.mean()), "bias_mcse": _mcse(error),
            "mse": float(loss.mean()), "mse_mcse": _mcse(loss),
            "rmse": rmse, "rmse_mcse": _mcse(loss)/(2*rmse) if rmse > 0 else np.nan,
            "coverage": coverage, "coverage_mcse": float(np.sqrt(coverage*(1-coverage)/count)),
            "coverage_mc_low": interval.low, "coverage_mc_high": interval.high,
            "mean_width": float(group.width.mean()), "width_mcse": _mcse(group.width),
            "mean_selected_power": float(group.selected_power.mean()),
            **{f"power_q{int(q*100):02d}": float(group.selected_power.quantile(q)) for q in (.1, .5, .9)},
            "zero_power_fraction": float(group.selected_power.eq(0).mean()),
            "constant_pilot_fraction": float(group.pilot_constant.mean()),
            "zero_width_fraction": float(group.zero_width.mean()),
            "full_domain_fraction": float(group.full_domain.mean()),
            "outside_unit_interval_fraction": float(group.point_outside_unit_interval.mean()),
            "empirical_variance": float(np.var(error, ddof=1)),
            "mean_estimated_variance": float(variance.mean()),
            "estimated_variance_mcse": _mcse(variance),
            "mean_conditional_variance": float(conditional.mean()),
            "conditional_variance_mcse": _mcse(conditional),
        })
    return pd.DataFrame(rows)


def _paired(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, cell in trials.groupby("scenario", sort=False):
        groups = {method: group.sort_values("repetition").reset_index(drop=True)
                  for method, group in cell.groupby("method", sort=False)}
        specs = [(method, baseline, metric) for method in POINT_METHODS
                 for baseline in ("human_normal", "same_audit_normal") if method != baseline
                 for metric in ("bias", "mse", "rmse")]
        for family, baselines in (("normal", ("human_normal", "same_audit_normal")),
                                  ("empirical_bernstein", ("human_eb", "same_audit_grid_eb"))):
            specs.extend((method, baseline, metric) for method, group in groups.items()
                         if group.interval_family.iloc[0] == family for baseline in baselines
                         if method != baseline for metric in ("width", "coverage"))
        for method, baseline, metric in specs:
            group, reference = groups[method], groups[baseline]
            if not np.array_equal(group.repetition, reference.repetition):
                raise ValueError("paired methods must have identical repetition IDs")
            if metric == "rmse":
                left, right = group.squared_error.to_numpy(), reference.squared_error.to_numpy()
                a, b = float(np.sqrt(left.mean())), float(np.sqrt(right.mean()))
                difference = a-b
                uncertainty = (_mcse(left/(2*a)-right/(2*b)) if a > 0 and b > 0 else
                               0. if np.array_equal(left, right) else np.nan)
            else:
                column = {"bias": "error", "mse": "squared_error", "width": "width", "coverage": "covered"}[metric]
                values = group[column].to_numpy(dtype=float)-reference[column].to_numpy(dtype=float)
                difference, uncertainty = float(values.mean()), _mcse(values)
            rows.append({**_identity(group), "baseline": baseline, "metric": metric,
                         "mean_difference": difference, "mcse": uncertainty, "repetitions": len(group)})
    return pd.DataFrame(rows)


def independent_pilot_study(repetitions: int = 2000, seed: int = 2031,
                            alpha: float = .05) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run 12 known-truth IID cells, returning trials, summaries and contrasts.

    The target is E[Y], not the realized proxy-pool mean. All draws are kept.
    Conditional true variances are diagnostics only for the fixed human and
    independent-pilot rules, never for the same-data auto/grid selection.
    Configuration is attached to trials only after aggregation to avoid
    copying it into pandas grouping operations. Monte Carlo uncertainty is
    pointwise across independent replications, not a guarantee across cells.
    """
    repetitions = _integer(repetitions, "repetitions", 2)
    seed = _integer(seed, "seed", 0)
    alpha, _ = _inference_options(alpha, 0.)
    rows, scenarios = [], []
    for profile, code, prevalence, sensitivity, specificity in PROFILES:
        for total in TOTAL_BUDGETS:
            scenario = f"iid_{profile}_b{total:04d}"
            q = prevalence*sensitivity + (1-prevalence)*(1-specificity)
            covariance = prevalence*(1-prevalence)*(sensitivity+specificity-1)
            scenarios.append({"scenario": scenario, "profile": profile, "profile_code": code,
                              "total_budget": total, "n_unlabeled": 10*total,
                              "truth": prevalence, "sensitivity": sensitivity, "specificity": specificity,
                              "proxy_mean": q, "outcome_proxy_covariance": covariance,
                              "iid_same_population": True})
            for repetition in range(repetitions):
                y_l, f_l, f_u = _draw(seed, code, total, repetition, prevalence, sensitivity, specificity)
                estimates = evaluate_draw(y_l, f_l, f_u, alpha)
                # Known parameters enter scoring only after every fitted rule.
                for estimate in estimates:
                    valid_diagnostic = estimate["tuning_source"] in ("fixed_zero", "independent_pilot")
                    conditional = (_conditional_variance(estimate["selected_power"], estimate["n_labeled"],
                                                         len(f_u), prevalence, sensitivity, specificity)
                                   if valid_diagnostic else np.nan)
                    error = estimate["point"]-prevalence
                    rows.append({"scenario": scenario, "profile": profile, "truth": prevalence,
                                 "repetition": repetition, **estimate,
                                 "error": error, "squared_error": error**2,
                                 "covered": _coverage_contains(estimate["interval_low"], estimate["interval_high"], prevalence),
                                 "conditional_variance": conditional,
                                 "conditional_variance_applicable": valid_diagnostic})
    trials = pd.DataFrame(rows)
    summary, paired = _summarize(trials), _paired(trials)
    trials.attrs["configuration"] = {
        "schema_version": 1, "study": "retrospective-independent-pilot-study-v1",
        "repetitions": repetitions, "seed": seed, "alpha": alpha,
        "numpy_version": np.__version__,
        "methods": list(METHODS), "distinct_point_methods": list(POINT_METHODS),
        "total_budgets": list(TOTAL_BUDGETS), "prediction_pool_ratio_to_total_budget": 10,
        "pilot_fractions": [.2, .5], "power_bounds": [0., 1.], "power_grid": list(POWER_GRID),
        "profile_codes": {name: code for name, code, *_ in PROFILES}, "scenarios": scenarios,
        "random_stream_scheme": "numpy.Philox(SeedSequence([seed,profile_code,total_budget,repetition,stream]))",
        "random_stream_codes": {"audit": 0, "prediction_only": 1},
        "pilot_split": "Fixed prefixes B//5 or B//2; disjoint suffixes for inference; all B labels charged",
        "pilot_power_formula": "clip(Cov_P(Y,F)/((1+m/N)*Var_P(F)),0,1); exact constant proxy selects zero",
        "pilot_power_numerics": "Exact integer covariance/variance numerators for binary pairs; scaled fractional moments without tolerance threshold",
        "same_audit_power_formula": "clip(Cov_L(Y,F)/(Var_L(F)+(B/N)*Var_U(F)),0,1)",
        "finite_interval_api": "judgecal.finite_sample_mean",
        "normal_interval_api": "judgecal.prediction_powered_mean",
        "interval_estimand": "population bounded-outcome mean",
        "monte_carlo_coverage_interval": {"method": "clopper-pearson", "alpha": .05, "scope": "pointwise"},
        "coverage_numerical_convention": "8*float64_epsilon*max(1,abs(low),abs(high),abs(truth)); scoring only",
        "notes": [
            "Established sample splitting and scalar PPI; no new estimator, theorem or preregistration claim.",
            "All methods share one labeled draw and independent prediction pool; no Y_U is generated or used.",
            "Pilot fitting sees only pilot pairs and design counts; final F_U and audit suffix cannot tune its power.",
            "Pilot labels count against B and are not reused in the final residual mean.",
            "All B+N proxies are materialized/supplied; human methods require zero proxies, other procedures B+N regardless of realized power.",
            "Independent pilot fitting is conditionally unbiased, but normal intervals remain asymptotic.",
            "Scalar EB is conditional on the independent pilot; grid EB uses its simultaneous finite family.",
            "Known conditional variance is reported only for fixed human and independent-pilot rules.",
            "Same-audit and pilot coefficient finite-sample variance functionals differ; gaps cannot be attributed solely to independence.",
            "Perfect proxies are a structural control, not information given to interval construction.",
            "All trials, zero widths, constant pilots and out-of-domain points remain; intervals are untruncated.",
            "Normal and EB variants of a pilot share a point; risk contrasts count distinct point rules only.",
            "Paired differences are method minus baseline and retain shared-draw covariance.",
            "Coverage uncertainty concerns independent simulation replications, not deployment uncertainty.",
        ],
    }
    return trials, summary, paired
