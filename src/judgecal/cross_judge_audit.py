"""Fixed-pool accuracy audits with a cached cross-judge agreement proxy."""
from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np
import pandas as pd

from .judge_audit import judge_accuracy_audit
from .rewardbench import REWARDBENCH_JUDGES, REWARDBENCH_SUBSETS


_COHORTS = ("NonLLMBar", "All", "LLMBar")
_REQUIRED = {
    "sample_id", "instruction_id", "source_id", "subset", "judge",
    "reference_choice", "canonical_choice", "reference_correct",
    "agreement_proxy", "auxiliary_judge",
}


def _validate_labels(labels: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(labels, pd.DataFrame) or not labels.columns.is_unique:
        raise ValueError("labels must be a DataFrame with unique column names")
    missing = _REQUIRED - set(labels.columns)
    if missing:
        raise ValueError(f"missing cross-judge audit columns: {sorted(missing)}")
    if labels.empty or labels[list(_REQUIRED)].isna().any().any():
        raise ValueError("cross-judge audit requires nonempty, nonmissing data")
    for name in ("sample_id", "instruction_id", "subset", "judge", "auxiliary_judge"):
        if not labels[name].map(lambda value: isinstance(value, str) and bool(value)
                                and value == value.strip()).all():
            raise ValueError(f"{name} must contain nonempty, unpadded strings")
    if not labels.subset.isin(REWARDBENCH_SUBSETS).all():
        raise ValueError("unknown RewardBench subset")
    if set(labels.judge) != set(REWARDBENCH_JUDGES):
        raise ValueError("the complete fixed two-judge panel is required")
    if not labels.source_id.map(lambda value: isinstance(value, Integral)
                                and not isinstance(value, (bool, np.bool_)) and value >= 0).all():
        raise ValueError("source_id must contain nonnegative integers; it is not a unique row key")
    for name, allowed in (("reference_choice", {1, 2}), ("canonical_choice", {1, 2}),
                          ("reference_correct", {0, 1}), ("agreement_proxy", {0, 1})):
        if not labels[name].map(lambda value: isinstance(value, Integral)
                                and not isinstance(value, (bool, np.bool_)) and value in allowed).all():
            raise ValueError(f"{name} must contain strict integers {sorted(allowed)}")
    if labels.duplicated(["sample_id", "judge"]).any():
        raise ValueError("each sample/judge must occur exactly once")
    if labels.groupby("sample_id", sort=False).size().ne(2).any():
        raise ValueError("each sample must have both judges; rows cannot be dropped")
    metadata = labels.groupby("sample_id", sort=False)[
        ["instruction_id", "source_id", "subset", "reference_choice"]].nunique()
    if metadata.ne(1).any().any():
        raise ValueError("sample metadata and reference choices must agree across judges")
    other = dict(zip(REWARDBENCH_JUDGES, reversed(REWARDBENCH_JUDGES)))
    if not labels.auxiliary_judge.eq(labels.judge.map(other)).all():
        raise ValueError("auxiliary_judge must be the other fixed model")
    data = labels.sort_values(["judge", "instruction_id", "sample_id"]).reset_index(drop=True).copy()
    choices = data.pivot(index="sample_id", columns="judge", values="canonical_choice")
    agreement = choices.iloc[:, 0].eq(choices.iloc[:, 1]).astype(int)
    if not data.agreement_proxy.eq(data.sample_id.map(agreement)).all():
        raise ValueError("agreement_proxy disagrees with the two canonical choices")
    if not data.reference_correct.eq(data.canonical_choice.eq(data.reference_choice).astype(int)).all():
        raise ValueError("reference_correct disagrees with the canonical choice and reference")
    return data


def _transport_panel(data: pd.DataFrame) -> pd.DataFrame:
    """Private aliases for the frozen two-prediction engine, not source fields."""
    choices = data.pivot(index="sample_id", columns="judge", values="canonical_choice")
    partner_choices = [int(choices.loc[sample, auxiliary])
                       for sample, auxiliary in zip(data.sample_id, data.auxiliary_judge)]
    return pd.DataFrame({
        "sample_id": data.sample_id, "instruction_id": data.instruction_id,
        "subset": "Natural", "judge": data.judge,
        "reference_label": data.reference_choice,
        "forward_prediction": data.canonical_choice, "reverse_prediction": partner_choices,
    })


def cross_judge_accuracy_audit(
    labels: pd.DataFrame,
    fractions: Sequence[float] = (.2, .4, .6),
    seeds: Sequence[int] = tuple(range(30)),
    evaluation_fraction: float = .25,
    min_cohort_instructions: int = 10,
    alpha: float = .05,
    *,
    cohorts: Sequence[str] = _COHORTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit both fixed model directions using the same gold-free agreement.

    NonLLMBar is the primary cohort; All and LLMBar are overlapping mixture
    and source-overlap sensitivities. Whole exact-prompt groups stay together
    across subset labels. Evaluation uses floor(evaluation_fraction * G)
    groups, and nested audit budgets use floor(fraction * G), where G is the
    TOTAL cohort group count. Actual reference costs count comparison rows.

    The six rules are raw_proxy, human_only (reference-only), ppi, ppi_tuned,
    ppi_signed and point-only audit_residual. Outcomes are strict correctness
    against the benchmark reference. Proxy agreement uses the two canonical
    choices only. Validation checks redundant reference fields, but held-out
    outcomes never enter estimation or tuning. All choices must be binary;
    the selected complete two-model panel has no invalid predictions.

    Population interval widths remain exploratory cluster-normal outputs;
    they do not establish coverage for the realized held-out accuracy target.
    The audit-residual candidate has no interval or standard error. Repeated
    split summaries are descriptive, without independent-replication MCSE.
    Reference and cached-input costs are shared across directions: do not sum
    method, direction or overlapping cohort counts as separate purchases.

    Returns trials and summaries with explicit target/auxiliary model names.
    Trial attrs contain split_manifest, audit_protocol and cohort_metadata.
    The legacy LLMBar engine is unchanged; transport aliases are private.
    """
    data = _validate_labels(labels)
    cohorts, fractions, seeds = tuple(cohorts), tuple(fractions), tuple(seeds)
    if (not cohorts or any(not isinstance(cohort, str) or cohort not in _COHORTS for cohort in cohorts)
            or len(set(cohorts)) != len(cohorts)):
        raise ValueError(f"cohorts must be unique nonempty selections from {_COHORTS}")
    trials_list, summary_list, manifests, cohort_metadata = [], [], [], []
    rename = {"judge": "target_judge", "human_labels_used": "reference_labels_used",
              "mean_human_labels_used": "mean_reference_labels_used",
              "heldout_forward_judgments_scored": "heldout_target_judgments_scored"}
    auxiliary = dict(zip(REWARDBENCH_JUDGES, reversed(REWARDBENCH_JUDGES)))
    for cohort in cohorts:
        is_llmbar = data.subset.str.startswith("llmbar-")
        selected = data if cohort == "All" else data.loc[is_llmbar == (cohort == "LLMBar")]
        panel = selected.loc[selected.judge == REWARDBENCH_JUDGES[0]]
        cohort_metadata.append({
            "cohort": cohort, "role": "primary" if cohort == "NonLLMBar" else "sensitivity",
            "samples": len(panel), "instruction_groups": int(panel.instruction_id.nunique()),
            "subset_counts": {str(key): int(value) for key, value in panel.subset.value_counts().sort_index().items()},
        })
        trials, summary = judge_accuracy_audit(
            _transport_panel(selected), fractions=fractions, seeds=seeds,
            evaluation_fraction=evaluation_fraction, min_cohort_instructions=min_cohort_instructions,
            alpha=alpha, cohorts=("All",), include_signed=True, include_pool_tuned=True,
        )
        for manifest in trials.attrs["split_manifest"]:
            manifest = dict(manifest)
            manifest["cohort"] = cohort
            manifest["target_judges"] = manifest.pop("judges")
            manifest["auxiliary_by_target"] = dict(auxiliary)
            manifests.append(manifest)
        # Avoid copying large inherited attrs while translating/grouping tables.
        trials.attrs = {}
        for table in (trials, summary):
            table.rename(columns=rename, inplace=True)
            table["cohort"] = cohort
            table["auxiliary_judge"] = table.target_judge.map(auxiliary)
        trials_list.append(trials)
        summary_list.append(summary)
    trials = pd.concat(trials_list, ignore_index=True)
    summary = pd.concat(summary_list, ignore_index=True)
    trials.attrs["split_manifest"] = manifests
    trials.attrs["cohort_metadata"] = cohort_metadata
    trials.attrs["audit_protocol"] = {
        "outcome": "1[target canonical choice equals benchmark reference choice]",
        "proxy": "1[target canonical choice equals auxiliary canonical choice]; no reference label",
        "models": list(REWARDBENCH_JUDGES), "auxiliary_by_target": auxiliary,
        "choice_contract": "Complete fixed two-model panel; strict binary canonical choices only; no coercion or row dropping",
        "cohorts": {"NonLLMBar": "Primary: subsets without llmbar- prefix",
                    "All": "Sensitivity: full comparison mixture",
                    "LLMBar": "Sensitivity: llmbar- source overlap, not independent new data"},
        "evaluation_fraction": float(evaluation_fraction), "labeled_fractions": [float(f) for f in fractions],
        "seeds": [int(seed) for seed in seeds], "alpha": float(alpha),
        "fraction_denominator": "Total cohort exact-prompt groups; floor each size; audits nested in remaining bank",
        "weighting": "Equal comparison weight; complete exact-prompt groups assigned together across subsets",
        "target": "Realized target-judge strict reference accuracy in the fixed held-out pool",
        "interval_scope": "Untruncated asymptotic prompt-cluster population intervals; no empirical coverage claim",
        "point_only_method": "audit_residual: no interval, standard error or population variance ratio",
        "human_only_definition": "Established method identifier for the reference-only baseline; references need not be human annotations",
        "power_bounds": {"ppi_tuned": [0., 1.], "ppi_signed": [-1., 1.], "audit_residual": [-1., 1.]},
        "cost_scope": "Logical reference labels and cached judgments read, not new API calls, annotations, tokens or money",
        "cost_sharing": "One reference serves both targets; both directions reuse the same two caches; deduplicate unions across methods, targets and cohorts",
        "scoring_cost_scope": "Held-out reference and target-judgment scoring reads are separate and may overlap inference inputs",
        "aggregation": "Descriptive dependent-split errors; no independent-replication MCSE; not the official weighted leaderboard score",
        "identity": "Gold-independent unordered-response sample IDs and exact-prompt group IDs are certified by the source loader",
        "model_scope": "Historical dated model caches; no assertion about current models or recovered API invocation settings",
    }
    return trials, summary
