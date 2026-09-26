"""Design-based audits of a fixed RewardBench frame, with no new model calls.

Full reference outcomes belong to the scorer. The fitted procedure receives
only sampled outcome totals, known group sizes and fixed gold-free proxies.
Repeated draws measure sampling-design error conditional on this one corpus.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
import json
from numbers import Integral

import numpy as np
import pandas as pd

from .cross_judge_audit import _validate_labels
from .estimand_study import _coverage_contains
from .finite_corpus import finite_corpus_mean
from .intervals import _binomial_exact_interval, _validate_alpha
from .rewardbench import REWARDBENCH_JUDGES


GROUP_FRACTIONS = (.20, .60, .90)
POWER_GRID = (0., .25, .5, .75, 1.)
METHODS = ("ht_hs", "ht_ebs", "agreement_grid_ebs", "size_grid_ebs")


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def catalog_sha256(catalog):
    return hashlib.sha256(_compact(catalog).encode("utf-8")).hexdigest()


def _ids_sha256(ids):
    return hashlib.sha256(("\n".join(sorted(ids))+"\n").encode("utf-8")).hexdigest()


def prepare_frame(labels):
    """Validate source redundancy, then build a stable NonLLMBar-only frame.

    Source validation may inspect gold consistency. Frame membership and proxy
    aggregation do not depend on gold. The returned outcome vectors must stay
    with the scorer, never be passed whole to evaluate_audit.
    """
    data = _validate_labels(labels)
    data = data.loc[~data.subset.str.startswith("llmbar-")]
    panel = data.loc[data.judge == REWARDBENCH_JUDGES[0]].sort_values("sample_id")
    if panel.instruction_id.nunique() < 5:
        raise ValueError("the study needs at least five complete prompt groups")
    row_ids = panel.sample_id.tolist()
    row_index = {value: i for i, value in enumerate(row_ids)}
    groups = [{"instruction_id": key,
               "row_indices": sorted(row_index[value] for value in rows.sample_id)}
              for key, rows in panel.groupby("instruction_id", sort=True)]
    catalog = {"schema_version": 1, "row_ids": row_ids, "groups": groups}
    sizes = np.array([len(group["row_indices"]) for group in groups], dtype=int)
    proxy_rows = panel.agreement_proxy.to_numpy(dtype=float)
    proxy = np.array([proxy_rows[group["row_indices"]].sum() for group in groups])
    outcomes = {}
    for judge in REWARDBENCH_JUDGES:
        y = data.loc[data.judge == judge].set_index("sample_id").loc[row_ids].reference_correct.to_numpy(dtype=float)
        outcomes[judge] = np.array([y[group["row_indices"]].sum() for group in groups])
    return catalog, sizes, proxy, outcomes


def draw_permutation(n_groups, seed, repetition):
    """Addressable uniform permutation; methods/judges/budgets share its prefixes."""
    n_groups = _integer(n_groups, "n_groups", 2)
    seed = _integer(seed, "seed", 0)
    repetition = _integer(repetition, "repetition", 0)
    rng = np.random.Generator(np.random.Philox(np.random.SeedSequence([seed, repetition])))
    return rng.permutation(n_groups)


def decode_sample(catalog, permutation, k):
    """Losslessly recover sorted prompt/row IDs from one indexed permutation."""
    count = len(catalog["groups"])
    values = list(permutation)
    if (len(values) != count or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in values)
            or sorted(values) != list(range(count))):
        raise ValueError("permutation must contain every group index exactly once")
    k = _integer(k, "k", 1)
    if k > count:
        raise ValueError("k exceeds frame group count")
    groups = [catalog["groups"][int(index)] for index in values[:k]]
    group_ids = sorted(group["instruction_id"] for group in groups)
    row_ids = sorted(catalog["row_ids"][index] for group in groups for index in group["row_indices"])
    return group_ids, row_ids


def evaluate_audit(group_sizes, agreement_totals, audited_indices, audited_outcome_totals, alpha=.05):
    """Four predeclared families; this interface has no unsampled outcomes.

    The size-only control uses F_g=n_g and exactly the agreement rule's grid
    and error allocation. We do not select across the four reported families.
    """
    sizes = np.asarray(group_sizes)
    specs = (("ht_hs", np.zeros(len(sizes)), 0., "hoeffding_serfling"),
             ("ht_ebs", np.zeros(len(sizes)), 0., "empirical_bernstein_serfling"),
             ("agreement_grid_ebs", agreement_totals, POWER_GRID, "empirical_bernstein_serfling"),
             ("size_grid_ebs", sizes, POWER_GRID, "empirical_bernstein_serfling"))
    rows = []
    for name, proxy, power, family in specs:
        result = finite_corpus_mean(sizes, proxy, audited_indices, audited_outcome_totals,
                                    alpha=alpha, power=power, method=family)
        point, low, high = result.point, result.interval.low, result.interval.high
        rows.append({
            "method": name, "interval_family": family,
            "point": point, "interval_low": low, "interval_high": high, "width": high-low,
            "selected_power": result.selected_power,
            "candidate_count": len(result.candidate_powers),
            **{f"candidate_{key}_json": _compact(list(getattr(result, f"candidate_{key}")))
               for key in ("powers", "points", "radii", "ranges", "sample_variances")},
            "n_groups": result.n_groups, "n_rows": result.n_rows,
            "n_audited_groups": result.n_audited_groups, "audited_rows": result.audited_rows,
            "expected_audited_rows": result.expected_audited_rows,
            "rho": result.rho, "finite_population_correction": result.finite_population_correction,
            "fixed_coefficient": name.startswith("ht_"),
            "required_agreement_rows": result.n_rows if name == "agreement_grid_ebs" else 0,
            "materialized_cached_choices": 2*result.n_rows,
            "full_domain": bool(low <= 0 and high >= 1), "zero_width": bool(high == low),
            "point_outside_unit_interval": bool(point < 0 or point > 1),
            "finite_design_guarantee": True, "alpha": alpha,
        })
    return rows


def _mcse(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1)/np.sqrt(len(values)))


def summarize_trials(trials):
    rows = []
    for keys, group in trials.groupby(["target_judge", "group_fraction", "method"], sort=False):
        judge, fraction, method = keys
        n = len(group)
        p = float(group.covered.mean())
        ci = _binomial_exact_interval(int(group.covered.sum()), n, .05)
        rmse = float(np.sqrt(group.squared_error.mean()))
        rows.append({
            "target_judge": judge, "group_fraction": fraction, "method": method,
            "truth": float(group.truth.iloc[0]), "repetitions": n,
            "n_audited_groups": int(group.n_audited_groups.iloc[0]),
            "bias": float(group.error.mean()), "bias_mcse": _mcse(group.error),
            "mse": float(group.squared_error.mean()), "mse_mcse": _mcse(group.squared_error),
            "rmse": rmse, "rmse_mcse": _mcse(group.squared_error)/(2*rmse) if rmse else np.nan,
            "coverage": p, "coverage_mcse": float(np.sqrt(p*(1-p)/n)),
            "coverage_mc_low": ci.low, "coverage_mc_high": ci.high,
            "mean_width": float(group.width.mean()), "width_mcse": _mcse(group.width),
            "full_domain_fraction": float(group.full_domain.mean()),
            "zero_width_fraction": float(group.zero_width.mean()),
            "outside_unit_interval_fraction": float(group.point_outside_unit_interval.mean()),
            "mean_audited_rows": float(group.audited_rows.mean()),
            "audit_cost_mcse": _mcse(group.audited_rows),
            "expected_audited_rows": float(group.expected_audited_rows.iloc[0]),
            "minimum_audited_rows": int(group.audited_rows.min()),
            "maximum_audited_rows": int(group.audited_rows.max()),
            **{f"audited_rows_q{int(q*100):02d}": float(group.audited_rows.quantile(q)) for q in (.1, .5, .9)},
            "mean_selected_power": float(group.selected_power.mean()),
            "power_frequencies_json": _compact({str(power): int(group.selected_power.eq(power).sum()) for power in POWER_GRID}),
            "alpha": float(group.alpha.iloc[0]),
        })
    return pd.DataFrame(rows)


def paired_differences(trials):
    """All six unordered comparisons, with named directional differences."""
    rows = []
    for (judge, fraction), cell in trials.groupby(["target_judge", "group_fraction"], sort=False):
        by_method = {name: group.sort_values("repetition").reset_index(drop=True)
                     for name, group in cell.groupby("method", sort=False)}
        for left, right in combinations(METHODS, 2):
            # Later method minus earlier: agreement-size is handled explicitly.
            method, baseline = (left, right) if right == "size_grid_ebs" and left == "agreement_grid_ebs" else (right, left)
            a, b = by_method[method], by_method[baseline]
            if (not np.array_equal(a.repetition, b.repetition)
                    or not np.array_equal(a.selected_group_sha256, b.selected_group_sha256)):
                raise ValueError("paired methods must share each exact sample")
            row = {"target_judge": judge, "group_fraction": fraction,
                   "method": method, "baseline": baseline, "repetitions": len(a)}
            for metric, field in (("bias", "error"), ("mse", "squared_error"), ("width", "width"), ("coverage", "covered")):
                differences = a[field].to_numpy(dtype=float)-b[field].to_numpy(dtype=float)
                row[f"{metric}_difference"] = float(differences.mean())
                row[f"{metric}_mcse"] = _mcse(differences)
            rows.append(row)
    return pd.DataFrame(rows)


def finite_corpus_study(labels, repetitions=2000, seed=2032, alpha=.05):
    """Return tables and lossless draw manifests for this fixed corpus.

    Independent objects are permutations, not methods or nested budgets.
    Census is a deterministic check and excluded from replication summaries.
    """
    repetitions = _integer(repetitions, "repetitions", 2)
    seed = _integer(seed, "seed", 0)
    alpha = _validate_alpha(alpha)
    catalog, sizes, proxy, outcomes = prepare_frame(labels)
    count, total = len(sizes), int(sizes.sum())
    budgets = [(fraction, int(np.floor(fraction*count))) for fraction in GROUP_FRACTIONS]
    rows, splits, draws = [], [], []
    for repetition in range(repetitions):
        permutation = draw_permutation(count, seed, repetition)
        draws.append({"repetition": repetition, "permutation": permutation.tolist()})
        for fraction, k in budgets:
            selected = permutation[:k]
            group_ids, row_ids = decode_sample(catalog, permutation, k)
            identity = {"repetition": repetition, "group_fraction": fraction,
                        "n_audited_groups": k, "audited_rows": len(row_ids),
                        "inclusion_probability": k/count,
                        "selected_group_sha256": _ids_sha256(group_ids),
                        "selected_row_sha256": _ids_sha256(row_ids)}
            splits.append(identity)
            for judge, gold in outcomes.items():
                estimates = evaluate_audit(sizes, proxy, selected, gold[selected], alpha)
                truth = float(gold.sum()/total)  # scoring only, after fitting
                for estimate in estimates:
                    error = estimate["point"]-truth
                    rows.append({**identity, "target_judge": judge, "truth": truth, **estimate,
                                 "error": error, "squared_error": error*error,
                                 "covered": _coverage_contains(estimate["interval_low"], estimate["interval_high"], truth)})
    trials = pd.DataFrame(rows)
    census = []
    for judge, gold in outcomes.items():
        for estimate in evaluate_audit(sizes, proxy, np.arange(count), gold, alpha):
            truth = float(gold.sum()/total)
            census.append({"target_judge": judge, "truth": truth, **estimate,
                           "exact": bool(estimate["point"] == truth and estimate["width"] == 0)})
    configuration = {
        "schema_version": 1, "study": "fixed-corpus-without-replacement-v1", "cohort": "NonLLMBar",
        "target": "full fixed corpus row-average reference correctness",
        "repetitions": repetitions, "independent_random_objects": repetitions,
        "seed": seed, "alpha": alpha, "numpy_version": np.__version__,
        "n_groups": count, "n_rows": total, "group_size_counts": {str(n): int((sizes == n).sum()) for n in np.unique(sizes)},
        "group_fractions": list(GROUP_FRACTIONS), "audit_group_counts": [k for _, k in budgets],
        "methods": list(METHODS), "power_grid": list(POWER_GRID), "judges": list(REWARDBENCH_JUDGES),
        "frame_sha256": catalog_sha256(catalog),
        "frame_hash_encoding": "SHA256(compact sorted-key ensure_ascii=True JSON catalog UTF8; no trailing newline)",
        "selected_id_hash_encoding": "SHA256(sorted IDs joined by LF plus terminal LF; UTF8)",
        "random_stream_scheme": "numpy.Generator(Philox(SeedSequence([seed,repetition]))).permutation(G)",
        "sampling": "fixed-k uniform groups without replacement; shared nested permutation prefixes",
        "cost": "random audited row count; expected k*M/G; shared across methods and target directions",
        "range_policy": "full known-frame min(-power*F_g), max(n_g-power*F_g); no outcome-based range",
        "coverage_scope": "each individual method/judge/budget conditional on fixed corpus and declared SRS design",
        "coverage_scoring": "eight-epsilon membership tolerance only; endpoints unchanged",
        "mc_interval": "pointwise 95% exact binomial for coverage; independent design replications only",
        "census": "one exact check per method and judge; excluded from Monte Carlo summaries",
        "new_model_calls": 0, "materialized_cached_choices": 2*total,
        "reference_scope": "frozen benchmark references, not error-free latent truth",
    }
    return {"trials": trials, "summary": summarize_trials(trials), "paired": paired_differences(trials),
            "splits": pd.DataFrame(splits), "census": pd.DataFrame(census), "catalog": catalog,
            "permutations": {"schema_version": 1, "frame_sha256": configuration["frame_sha256"], "draws": draws},
            "configuration": configuration}
