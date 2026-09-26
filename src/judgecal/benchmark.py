"""Reproducible, question-disjoint human-label-budget evaluation.

The benchmark compares point estimates to a hidden evaluation-set human
reference. Repeated splits of one dataset are a descriptive stress test,
not independent replications or an empirical population-CI coverage study.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral, Real
from typing import Sequence

import numpy as np
import pandas as pd

from .ppi import prediction_powered_win_rate


def question_disjoint_split(
    frame: pd.DataFrame, labeled_fraction: float, random_state: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete questions; no turn of an audit question enters evaluation."""
    if not np.isfinite(labeled_fraction) or not 0 < labeled_fraction < 1:
        raise ValueError("labeled_fraction must be strictly between zero and one")
    if "question_id" not in frame or frame["question_id"].isna().any():
        raise ValueError("question_id must be present and nonmissing")
    if any(isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) or q < 0
           for q in frame["question_id"]):
        raise ValueError("question_id values must be nonnegative integers")
    groups = sorted(frame["question_id"].unique().tolist())
    n_labeled = int(np.floor(len(groups) * labeled_fraction))
    if n_labeled < 2 or len(groups) - n_labeled < 2:
        raise ValueError("each pool needs at least two distinct questions")
    rng = np.random.default_rng(random_state)
    selected = {groups[i] for i in rng.permutation(len(groups))[:n_labeled]}
    mask = frame["question_id"].isin(selected)
    return frame.loc[mask].copy(), frame.loc[~mask].copy()


def _scores(labels: pd.Series) -> np.ndarray:
    mapped = labels.map({"A": 1.0, "B": 0.0, "tie": 0.5})
    if mapped.isna().any():
        raise ValueError("winner labels must be A, B, or tie")
    return mapped.to_numpy(dtype=float)


def evaluate_label_budget(
    frame: pd.DataFrame,
    fractions: Sequence[float] = (0.2, 0.4, 0.6),
    seeds: Sequence[int] = tuple(range(30)),
    min_pair_questions: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare raw judge, human-only and original PPI across every eligible pair.

    Model A is the alphabetically first model; a tie is half a win. All
    comparisons for each question stay in one pool. Evaluation human labels
    are accessed only after constructing the estimates. No method is tuned
    on them. Reported interval widths are asymptotic PPI widths, not a claim
    that these intervals cover this fixed evaluation subset's realized mean.
    """
    required = {"question_id", "model_a", "model_b", "human_winner", "gpt4_winner"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("benchmark requires nonempty, nonmissing aligned data")
    if (frame["model_a"] >= frame["model_b"]).any():
        raise ValueError("model pairs must be distinct and canonically ordered")
    if isinstance(min_pair_questions, bool) or not isinstance(min_pair_questions, int) or min_pair_questions < 4:
        raise ValueError("min_pair_questions must be an integer >= 4")
    fractions, seeds = tuple(fractions), tuple(seeds)
    if not fractions or not seeds or len(set(fractions)) != len(fractions) or len(set(seeds)) != len(seeds):
        raise ValueError("fractions and seeds must be nonempty and unique")
    for value in fractions:
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError("fractions must be strictly between zero and one")
    if any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0 for s in seeds):
        raise ValueError("seeds must be nonnegative integers")
    _scores(frame["human_winner"])
    _scores(frame["gpt4_winner"])
    keys = ["question_id", "model_a", "model_b"]
    if "turn" in frame:
        keys.append("turn")
    if frame.duplicated(keys).any():
        raise ValueError("aggregate repeated annotator votes before benchmarking")

    records, manifests = [], []
    for (model_a, model_b), pair in frame.groupby(["model_a", "model_b"], sort=True):
        pair = pair.sort_values(keys).reset_index(drop=True)
        if pair["question_id"].nunique() < min_pair_questions:
            continue
        for fraction in fractions:
            for seed in seeds:
                audit, evaluation = question_disjoint_split(pair, fraction, int(seed))
                result = prediction_powered_win_rate(
                    audit["gpt4_winner"], audit["human_winner"], evaluation["gpt4_winner"],
                    labeled_groups=audit["question_id"], unlabeled_groups=evaluation["question_id"],
                )
                # Hidden labels are used only for scoring, after inference.
                reference = float(_scores(evaluation["human_winner"]).mean())
                split = {
                    "labeled_question_ids": sorted(int(x) for x in audit["question_id"].unique()),
                    "evaluation_question_ids": sorted(int(x) for x in evaluation["question_id"].unique()),
                }
                digest = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
                common = {
                    "model_a": model_a, "model_b": model_b, "labeled_fraction": float(fraction),
                    "seed": int(seed), "n_labeled": len(audit), "n_evaluation": len(evaluation),
                    "labeled_questions": audit["question_id"].nunique(),
                    "evaluation_questions": evaluation["question_id"].nunique(),
                    "heldout_human_reference": reference, "split_sha256": digest,
                }
                for method, estimate in [
                    ("raw_judge", result.raw_rate), ("human_only", result.human_only_rate), ("ppi", result.point)
                ]:
                    records.append({
                        **common, "method": method, "estimate": estimate,
                        "absolute_error": abs(estimate - reference),
                        "squared_error": (estimate - reference) ** 2,
                        "ppi_interval_width": result.interval.high - result.interval.low if method == "ppi" else None,
                    })
                manifests.append({**common, **split})
    if not records:
        raise ValueError("no model pair meets min_pair_questions")
    trials = pd.DataFrame(records)
    trials.attrs["split_manifest"] = manifests
    summary = (
        trials.groupby(["labeled_fraction", "method"], sort=True)
        .agg(mean_absolute_error=("absolute_error", "mean"),
             mean_squared_error=("squared_error", "mean"),
             mean_interval_width=("ppi_interval_width", "mean"),
             trials=("estimate", "size"))
        .reset_index()
    )
    return trials, summary


def fixed_evaluation_label_budget(
    frame: pd.DataFrame,
    fractions: Sequence[float] = (0.2, 0.4, 0.6),
    seeds: Sequence[int] = tuple(range(30)),
    evaluation_fraction: float = 0.25,
    min_pair_questions: int = 40,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare audit budgets against the same held-out target within each split.

    For each eligible model pair and seed, a single random permutation of
    questions first reserves ``floor(evaluation_fraction * G)`` evaluation
    questions. Its remaining questions form an ordered audit bank. Budget
    ``f`` selects the bank's first ``floor(f * G)`` questions, where G is the
    total question count for that pair. Audit sets are therefore nested and
    the evaluation pool stays fixed across budgets. Other bank questions are
    unused: their human labels and judge predictions do not enter inference.

    Raw judge, human-only, original PPI and audit-tuned PPI use identical
    partitions. Tuning sees only the audit human labels. All turns remain
    together within each pair; model pairs are analyzed separately
    and may assign the same question to different pools. No cross-pair model
    is trained. Equal-weight pair/split aggregates describe this fixed data;
    they do not establish population generalization or annotation savings.

    Human/PPI intervals are asymptotic population-mean intervals, not
    confidence intervals for the realized held-out human reference. Raw
    judge intervals are omitted because they would quantify uncertainty in
    the judge mean, rather than its error relative to human preference.
    ``human_comparisons_used`` counts aggregated audit comparisons and
    ``human_votes_used`` counts their recorded annotations when available;
    neither represents dollars or a measured annotation-time budget.

    Returns trial and aggregate tables. ``trials.attrs['split_manifest']``
    contains explicit labeled, evaluation, and unused question IDs for every
    pair/budget/seed. The legacy complementary-pool benchmark is unchanged.
    """
    required = {"question_id", "model_a", "model_b", "human_winner", "gpt4_winner"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("benchmark requires nonempty, nonmissing aligned data")
    for column in ("model_a", "model_b"):
        if not frame[column].map(
            lambda x: isinstance(x, str) and bool(x) and x == x.strip()
        ).all():
            raise ValueError("model names must be nonempty, unpadded strings")
    if (frame["model_a"] >= frame["model_b"]).any():
        raise ValueError("model pairs must be distinct and canonically ordered")
    if any(isinstance(q, (bool, np.bool_)) or not isinstance(q, Integral) or q < 0
           for q in frame["question_id"]):
        raise ValueError("question_id values must be nonnegative integers")
    if (isinstance(min_pair_questions, (bool, np.bool_))
            or not isinstance(min_pair_questions, Integral) or min_pair_questions < 4):
        raise ValueError("min_pair_questions must be an integer >= 4")

    def valid_fraction(value: object) -> bool:
        return (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                and np.isfinite(value) and 0 < value < 1)

    fractions, seeds = tuple(fractions), tuple(seeds)
    if not fractions or not seeds:
        raise ValueError("fractions and seeds must be nonempty and unique")
    if any(not valid_fraction(value) for value in fractions):
        raise ValueError("fractions must be finite numbers strictly between zero and one")
    if any(isinstance(s, (bool, np.bool_)) or not isinstance(s, Integral) or s < 0 for s in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(fractions)) != len(fractions) or len(set(seeds)) != len(seeds):
        raise ValueError("fractions and seeds must be nonempty and unique")
    if not valid_fraction(evaluation_fraction):
        raise ValueError("evaluation_fraction must be strictly between zero and one")
    if any(f + evaluation_fraction > 1 + 1e-12 for f in fractions):
        raise ValueError("audit fractions plus evaluation_fraction must not exceed one")
    if not valid_fraction(alpha):
        raise ValueError("alpha must be strictly between zero and one")
    _scores(frame["human_winner"])
    _scores(frame["gpt4_winner"])
    keys = ["question_id", "model_a", "model_b"]
    if "turn" in frame:
        if frame["turn"].isna().any():
            raise ValueError("turn must be nonmissing when provided")
        keys.append("turn")
    if frame.duplicated(keys).any():
        raise ValueError("aggregate repeated annotator votes before benchmarking")
    if "n_human_votes" in frame and not frame["n_human_votes"].map(
        lambda x: isinstance(x, Integral) and not isinstance(x, (bool, np.bool_)) and x > 0
    ).all():
        raise ValueError("n_human_votes must contain positive integer annotation counts")

    records, manifests = [], []
    for (model_a, model_b), pair in frame.groupby(["model_a", "model_b"], sort=True):
        pair = pair.sort_values(keys).reset_index(drop=True)
        groups = sorted(int(q) for q in pair["question_id"].unique())
        n_groups = len(groups)
        if n_groups < min_pair_questions:
            continue
        n_evaluation_groups = int(np.floor(n_groups * evaluation_fraction))
        sizes = {float(f): int(np.floor(n_groups * f)) for f in fractions}
        if n_evaluation_groups < 2 or min(sizes.values()) < 2:
            raise ValueError("each audit and evaluation pool needs at least two distinct questions")
        if max(sizes.values()) + n_evaluation_groups > n_groups:
            raise ValueError("audit and evaluation pools exceed the available questions")
        for seed in seeds:
            order = np.random.default_rng(int(seed)).permutation(groups).tolist()
            evaluation_ids = sorted(order[:n_evaluation_groups])
            bank = order[n_evaluation_groups:]
            evaluation = pair.loc[pair["question_id"].isin(evaluation_ids)]
            for fraction in fractions:
                labeled_ids = sorted(bank[:sizes[float(fraction)]])
                unused_ids = sorted(bank[sizes[float(fraction)]:])
                audit = pair.loc[pair["question_id"].isin(labeled_ids)]
                arguments = {
                    "judge_labeled": audit["gpt4_winner"],
                    "human_labeled": audit["human_winner"],
                    "judge_unlabeled": evaluation["gpt4_winner"],
                    "labeled_groups": audit["question_id"],
                    "unlabeled_groups": evaluation["question_id"],
                    "alpha": float(alpha),
                }
                results = {
                    "human_only": prediction_powered_win_rate(**arguments, power=0.0),
                    "ppi": prediction_powered_win_rate(**arguments, power=1.0),
                    "ppi_tuned": prediction_powered_win_rate(**arguments, power="auto"),
                }
                # Held-out human labels enter only this post-inference scoring step.
                reference = float(_scores(evaluation["human_winner"]).mean())
                split = {
                    "labeled_question_ids": labeled_ids,
                    "evaluation_question_ids": evaluation_ids,
                    "unused_question_ids": unused_ids,
                }
                digest = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
                common = {
                    "model_a": model_a, "model_b": model_b,
                    "labeled_fraction": float(fraction),
                    "evaluation_fraction": float(evaluation_fraction), "seed": int(seed),
                    "n_labeled": len(audit), "n_evaluation": len(evaluation),
                    "n_unused": int(pair["question_id"].isin(unused_ids).sum()),
                    "labeled_questions": len(labeled_ids),
                    "evaluation_questions": len(evaluation_ids),
                    "unused_questions": len(unused_ids), "total_questions": n_groups,
                    "heldout_human_reference": reference, "split_sha256": digest,
                }
                for method in ("raw_judge", "human_only", "ppi", "ppi_tuned"):
                    result = results.get(method)
                    estimate = results["ppi"].raw_rate if result is None else result.point
                    error = estimate - reference
                    records.append({
                        **common, "method": method, "estimate": estimate,
                        "signed_error": error, "absolute_error": abs(error),
                        "squared_error": error ** 2,
                        "interval_low": None if result is None else result.interval.low,
                        "interval_high": None if result is None else result.interval.high,
                        "standard_error": None if result is None else result.standard_error,
                        "interval_width": None if result is None else result.interval.high - result.interval.low,
                        "selected_power": None if result is None else result.selected_power,
                        "power_method": None if result is None else result.power_method,
                        "estimated_variance_ratio": None if result is None else result.estimated_variance_ratio,
                        "human_comparisons_used": 0 if result is None else len(audit),
                        "human_votes_used": (0 if result is None else
                                             int(audit["n_human_votes"].sum())
                                             if "n_human_votes" in audit else None),
                    })
                manifests.append({**common, **split})
    if not records:
        raise ValueError("no model pair meets min_pair_questions")
    trials = pd.DataFrame(records)
    trials.attrs["split_manifest"] = manifests
    summary = (
        trials.groupby(["labeled_fraction", "method"], sort=True)
        .agg(mean_absolute_error=("absolute_error", "mean"),
             mean_squared_error=("squared_error", "mean"),
             mean_interval_width=("interval_width", "mean"),
             mean_bias=("signed_error", "mean"),
             mean_selected_power=("selected_power", "mean"),
             trials=("estimate", "size"))
        .reset_index()
    )
    return trials, summary
