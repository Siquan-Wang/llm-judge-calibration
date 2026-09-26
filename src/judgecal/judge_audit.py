"""Retrospective accuracy audits of complete panels of cached LLM judgments.

Order agreement is a gold-free proxy for correctness, not correctness itself.
All inference uses instruction groups; this corpus benchmark makes no claim
that repeated split errors or asymptotic interval widths establish coverage.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral, Real
from typing import Sequence

import numpy as np
import pandas as pd

from .ppi import prediction_powered_mean


_SUBSETS = {"Natural", "Neighbor", "GPTInst", "GPTOut", "Manual"}
_COHORTS = ("All", "Natural", "Adversarial")
_METHODS = ("raw_proxy", "human_only", "ppi", "ppi_tuned")
_REQUIRED = {
    "sample_id", "instruction_id", "subset", "judge", "reference_label",
    "forward_prediction", "reverse_prediction",
}


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("frame must be a DataFrame with unique column names")
    missing = _REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"missing cached-audit columns: {sorted(missing)}")
    if frame.empty or frame[list(_REQUIRED)].isna().any().any():
        raise ValueError("cached audit requires nonempty, nonmissing data")
    for column in ("sample_id", "instruction_id", "subset", "judge"):
        if not frame[column].map(
            lambda x: isinstance(x, str) and bool(x) and x == x.strip()
        ).all():
            raise ValueError(f"{column} must contain nonempty, unpadded strings")
    if not frame["subset"].isin(_SUBSETS).all():
        raise ValueError(f"unknown subset; expected one of {sorted(_SUBSETS)}")
    for column, allowed in (("reference_label", {1, 2}),
                            ("forward_prediction", {0, 1, 2}),
                            ("reverse_prediction", {0, 1, 2})):
        if not frame[column].map(
            lambda x: isinstance(x, Integral) and not isinstance(x, (bool, np.bool_))
            and x in allowed
        ).all():
            raise ValueError(f"{column} must contain canonical integers {sorted(allowed)}; "
                             "only explicit prediction 0 denotes an invalid parse")
    if frame.duplicated(["judge", "sample_id"]).any():
        raise ValueError("each judge/sample_id must occur exactly once")
    metadata = frame.groupby("sample_id", sort=False)[
        ["instruction_id", "subset", "reference_label"]
    ].nunique()
    if metadata.ne(1).any().any():
        raise ValueError("sample metadata and reference labels must agree across judges")
    n_judges = frame["judge"].nunique()
    if frame.groupby("sample_id", sort=False).size().ne(n_judges).any():
        raise ValueError("all judges must have the same complete sample panel; missing rows cannot be dropped")
    return frame.sort_values(["judge", "instruction_id", "sample_id"]).reset_index(drop=True)


def _fraction(value: object) -> bool:
    return (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and 0 < value < 1 and np.isfinite(float(value)) and 0 < float(value) < 1)


def _proxy(frame: pd.DataFrame) -> np.ndarray:
    """Use only the two already-canonical cached predictions, never gold."""
    forward = frame["forward_prediction"].to_numpy()
    reverse = frame["reverse_prediction"].to_numpy()
    return ((forward != 0) & (reverse != 0) & (forward == reverse)).astype(float)


def _correct(frame: pd.DataFrame) -> np.ndarray:
    # Invalid forward predictions (0) cannot equal the required gold 1 or 2.
    return (frame["forward_prediction"].to_numpy()
            == frame["reference_label"].to_numpy()).astype(float)


def judge_accuracy_audit(
    frame: pd.DataFrame,
    fractions: Sequence[float] = (0.2, 0.4, 0.6),
    seeds: Sequence[int] = tuple(range(30)),
    evaluation_fraction: float = 0.25,
    min_cohort_instructions: int = 10,
    alpha: float = 0.05,
    *,
    cohorts: Sequence[str] = _COHORTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit forward-judge accuracy with gold-free response-order agreement.

    Gold labels are integer 1/2; both predictions are already in that same
    canonical response orientation. Prediction 0 is an explicit invalid
    parse, retained with forward correctness zero and order-agreement zero.
    Unknown strings, nulls, booleans and unrecognized integers are rejected;
    parsing and canonical orientation are the source loader's responsibility.

    The proxy is 1 iff both predictions are valid and equal. Correctness is
    1 iff the forward prediction equals gold. Rows/comparisons are weighted
    equally, including repeated instructions. Every judge must cover the
    same complete panel, with consistent metadata and gold. One comparison's
    gold label supports every judge, so per-judge human-label costs MUST NOT
    be summed across judges. Cached-judgment costs count method inputs, not
    API calls, tokens, money, or newly purchased labels.

    Within each requested cohort/seed, shuffle sorted instruction IDs, fix
    floor(evaluation_fraction * G) evaluation groups, then use nested bank
    prefixes of floor(fraction * G) audit groups. Unused rows never enter
    inference. All judges share those groups. All/Natural/Adversarial are
    overlapping descriptive analyses; no model is trained across cohorts.
    Adversarial aggregates Neighbor/GPTInst/GPTOut/Manual. Missing or too-small
    requested cohorts raise rather than silently changing the analysis.

    The held-out target is the realized forward accuracy on that fixed pool.
    Evaluation gold is used only for integrity checks and post-inference
    scoring. Correction and tuning use audit correctness only. Normal
    intervals use the instruction-cluster sandwich; they concern hypothetical
    population means under sampling assumptions, not confidence intervals
    for this realized held-out target. No coverage statistic is reported.
    Raw proxy intervals are omitted because they would measure uncertainty
    in agreement rather than accuracy. No iid finite-sample bounds are used.

    Returns trials and descriptive mean errors/widths over seeds. Seeds
    overlap in this fixed corpus, so no iid Monte Carlo errors are attached.
    Trial attrs contain gold-free split_manifest and audit_protocol metadata.
    """
    data = _validate_frame(frame)
    fractions, seeds, cohorts = tuple(fractions), tuple(seeds), tuple(cohorts)
    if (not fractions or any(not _fraction(f) for f in fractions)
            or len(set(float(f) for f in fractions)) != len(fractions)):
        raise ValueError("fractions must be unique nonempty finite numbers strictly between zero and one")
    if not seeds or any(isinstance(s, (bool, np.bool_)) or not isinstance(s, Integral)
                        or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique nonempty nonnegative integers")
    if (not cohorts or any(not isinstance(c, str) or c not in _COHORTS for c in cohorts)
            or len(set(cohorts)) != len(cohorts)):
        raise ValueError(f"cohorts must be unique nonempty selections from {_COHORTS}")
    if not _fraction(evaluation_fraction) or not _fraction(alpha):
        raise ValueError("evaluation_fraction and alpha must be strictly between zero and one")
    if any(float(f) + float(evaluation_fraction) > 1 for f in fractions):
        raise ValueError("audit fraction plus evaluation_fraction must not exceed one")
    if (isinstance(min_cohort_instructions, (bool, np.bool_))
            or not isinstance(min_cohort_instructions, Integral) or min_cohort_instructions < 4):
        raise ValueError("min_cohort_instructions must be an integer >= 4")
    judges = sorted(data["judge"].unique())
    records, manifests = [], []
    for cohort in cohorts:
        selected = (data if cohort == "All" else
                    data.loc[data["subset"].eq("Natural") == (cohort == "Natural")])
        panel = selected.loc[selected["judge"] == judges[0]]
        group_ids = sorted(panel["instruction_id"].unique())
        n_groups = len(group_ids)
        if n_groups < min_cohort_instructions:
            raise ValueError(f"cohort {cohort} has fewer than min_cohort_instructions")
        n_evaluation_groups = int(np.floor(n_groups * float(evaluation_fraction)))
        sizes = {float(f): int(np.floor(n_groups * float(f))) for f in fractions}
        if n_evaluation_groups < 2 or min(sizes.values()) < 2:
            raise ValueError("each audit and evaluation pool requires at least two instruction groups")
        if n_evaluation_groups + max(sizes.values()) > n_groups:
            raise ValueError("audit and evaluation groups exceed the cohort")
        judge_frames = {judge: rows for judge, rows in selected.groupby("judge", sort=True)}
        for seed in seeds:
            order = np.random.default_rng(int(seed)).permutation(group_ids).tolist()
            evaluation_ids = sorted(order[:n_evaluation_groups])
            bank = order[n_evaluation_groups:]
            evaluation_samples = sorted(panel.loc[panel.instruction_id.isin(evaluation_ids), "sample_id"])
            for fraction in fractions:
                labeled_ids = sorted(bank[:sizes[float(fraction)]])
                unused_ids = sorted(bank[sizes[float(fraction)]:])
                labeled_samples = sorted(panel.loc[panel.instruction_id.isin(labeled_ids), "sample_id"])
                unused_samples = sorted(panel.loc[panel.instruction_id.isin(unused_ids), "sample_id"])
                pools = {
                    "labeled_instruction_ids": labeled_ids, "evaluation_instruction_ids": evaluation_ids,
                    "unused_instruction_ids": unused_ids, "labeled_sample_ids": labeled_samples,
                    "evaluation_sample_ids": evaluation_samples, "unused_sample_ids": unused_samples,
                }
                digest = hashlib.sha256(json.dumps(pools, sort_keys=True).encode()).hexdigest()
                common = {
                    "cohort": cohort, "labeled_fraction": float(fraction), "seed": int(seed),
                    "evaluation_fraction": float(evaluation_fraction), "alpha": float(alpha),
                    "n_labeled": len(labeled_samples), "n_evaluation": len(evaluation_samples),
                    "n_unused": len(unused_samples), "total_samples": len(panel),
                    "labeled_instructions": len(labeled_ids), "evaluation_instructions": len(evaluation_ids),
                    "unused_instructions": len(unused_ids), "total_instructions": n_groups,
                    "split_sha256": digest,
                }
                manifests.append({**common, **pools, "judges": judges})
                for judge in judges:
                    rows = judge_frames[judge]
                    audit = rows.loc[rows.instruction_id.isin(labeled_ids)]
                    evaluation = rows.loc[rows.instruction_id.isin(evaluation_ids)]
                    audit_proxy, evaluation_proxy = _proxy(audit), _proxy(evaluation)
                    audit_correct = _correct(audit)
                    arguments = {
                        "outcomes_labeled": audit_correct, "alpha": float(alpha),
                        "labeled_groups": audit["instruction_id"],
                        "unlabeled_groups": evaluation["instruction_id"],
                    }
                    results = {
                        "human_only": prediction_powered_mean(
                            predictions_labeled=np.zeros(len(audit)),
                            predictions_unlabeled=np.zeros(len(evaluation)), **arguments, power=0.),
                        **{method: prediction_powered_mean(
                            predictions_labeled=audit_proxy, predictions_unlabeled=evaluation_proxy,
                            **arguments, power=power)
                           for method, power in (("ppi", 1.), ("ppi_tuned", "auto"))},
                    }
                    # Held-out gold first enters numerical scoring after inference.
                    reference = float(_correct(evaluation).mean())
                    for method in _METHODS:
                        result = results.get(method)
                        estimate = float(evaluation_proxy.mean()) if result is None else result.point
                        error = estimate - reference
                        audit_cache = (0 if method == "raw_proxy" else
                                       len(audit) if method == "human_only" else 2 * len(audit))
                        evaluation_cache = 0 if method == "human_only" else 2 * len(evaluation)
                        records.append({
                            **common, "judge": judge, "method": method,
                            "heldout_accuracy_reference": reference, "estimate": estimate,
                            "signed_error": error, "absolute_error": abs(error), "squared_error": error**2,
                            "interval_low": None if result is None else result.interval.low,
                            "interval_high": None if result is None else result.interval.high,
                            "interval_width": None if result is None else result.interval.high - result.interval.low,
                            "standard_error": None if result is None else result.standard_error,
                            "selected_power": None if result is None else result.selected_power,
                            "power_method": None if result is None else result.power_method,
                            "estimated_variance_ratio": None if result is None else result.estimated_variance_ratio,
                            "variance_method": None if result is None else result.variance_method,
                            "human_labels_used": 0 if result is None else len(audit),
                            "audit_cached_judgments_used": audit_cache,
                            "evaluation_cached_judgments_used": evaluation_cache,
                            "cached_judgments_used": audit_cache + evaluation_cache,
                            "heldout_reference_labels_scored": len(evaluation),
                            "heldout_forward_judgments_scored": len(evaluation),
                        })
    trials = pd.DataFrame(records)
    summary = (
        trials.groupby(["cohort", "judge", "labeled_fraction", "method"], sort=True)
        .agg(mean_absolute_error=("absolute_error", "mean"),
             mean_squared_error=("squared_error", "mean"), mean_bias=("signed_error", "mean"),
             mean_interval_width=("interval_width", "mean"),
             mean_selected_power=("selected_power", "mean"),
             mean_human_labels_used=("human_labels_used", "mean"),
             mean_cached_judgments_used=("cached_judgments_used", "mean"),
             trials=("estimate", "size"))
        .reset_index()
    )
    summary["root_mean_squared_error"] = np.sqrt(summary["mean_squared_error"])
    # Attach large metadata only after grouping, avoiding repeated pandas attrs copies.
    trials.attrs["split_manifest"] = manifests
    trials.attrs["audit_protocol"] = {
        "outcome": "1[canonical forward prediction equals reference label]; invalid forward=0",
        "proxy": "1[both canonical predictions valid and equal]; no reference label",
        "invalid_prediction_policy": "Only explicit integer 0 denotes invalid; retained, never dropped",
        "weighting": "Equal comparison weight, with complete instruction groups assigned together",
        "target": "Realized forward-judge accuracy in the fixed held-out pool",
        "interval_scope": "Untruncated asymptotic instruction-cluster intervals; no held-out or population coverage claim",
        "cost_scope": "Logical inference inputs from cached data, not API charges or purchased annotation",
        "human_label_sharing": "One sample gold label audits every judge; do not sum costs across judges",
        "scoring_cost_scope": "Held-out gold/forward scoring counts are separate and may overlap cached inference inputs",
        "aggregation": "Descriptive dependent-split means, separately by overlapping cohort and judge; no iid MCSE",
    }
    return trials, summary
