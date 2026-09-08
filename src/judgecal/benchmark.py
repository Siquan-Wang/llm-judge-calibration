"""Reproducible, question-disjoint human-label-budget evaluation.

The benchmark compares point estimates to a hidden evaluation-set human
reference. Repeated splits of one dataset are a descriptive stress test,
not independent replications or an empirical population-CI coverage study.
"""
from __future__ import annotations

import hashlib
import json
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
