"""Dataset loaders.

The demo dataset is `lmsys/mt_bench_human_judgments` (from "Judging
LLM-as-a-Judge with MT-Bench and Chatbot Arena", Zheng et al., NeurIPS 2023):
3.3k expert human pairwise votes and matching GPT-4 pairwise votes over the
same (question, model_a, model_b) tuples. Requires the optional `datasets`
dependency (`pip install judgecal[data]`).

`make_synthetic` generates aligned judge/human labels with a controllable
agreement level so tests and offline demos need no network access.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np
import pandas as pd


MT_BENCH_DATASET = "lmsys/mt_bench_human_judgments"
MT_BENCH_REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"
_KEYS = ["question_id", "turn", "m_first", "m_second"]
_WINNERS = {"model_a", "model_b", "tie", "tie (inconsistent)"}
_DEDUPLICATION = (
    "One vote per (question_id, turn, canonical model pair, judge); "
    "identical repeated labels are deduplicated, retaining the first source row's "
    "diagnostics; conflicting labels raise ValueError."
)
_AGGREGATION = (
    "Unique modal label among A/B/tie (plurality); any tie for the largest "
    "vote count becomes tie. Includes both author_* and expert_* annotators."
)


def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort model identities and flip their winner, rejecting unknown labels.

    ``input_reversed`` describes source ordering relative to alphabetical
    ordering. It is not evidence of position bias. The source's separate
    ``tie (inconsistent)`` flag is retained before normalizing ties.
    """
    missing = {"model_a", "model_b", "winner"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing pairwise columns: {sorted(missing)}")
    for column in ("model_a", "model_b"):
        if not df[column].map(
            lambda x: isinstance(x, str) and bool(x) and x == x.strip()
        ).all():
            raise ValueError(f"{column} must contain nonempty, unpadded model names")
    if (df["model_a"] == df["model_b"]).any():
        raise ValueError("Pairwise comparisons require two distinct model names")
    if not df["winner"].map(lambda x: isinstance(x, str) and x in _WINNERS).all():
        raise ValueError(f"winner must be one of {sorted(_WINNERS)}")
    out = df.copy()
    reversed_order = df["model_a"] > df["model_b"]
    out["m_first"] = df["model_a"].where(~reversed_order, df["model_b"])
    out["m_second"] = df["model_b"].where(~reversed_order, df["model_a"])
    out["label"] = np.where(
        df["winner"].isin(["tie", "tie (inconsistent)"]),
        "tie",
        np.where((df["winner"] == "model_a") != reversed_order, "A", "B"),
    )
    out["input_reversed"] = reversed_order
    out["inconsistent_order"] = df["winner"] == "tie (inconsistent)"
    return out


def _response_digest(value: object) -> str:
    # Hugging Face's pandas conversion represents conversations as ndarrays.
    text = json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prepare_split(df: pd.DataFrame) -> pd.DataFrame:
    out = _canonicalize(df)
    for column in ("question_id", "turn", "judge"):
        if column not in out or out[column].isna().any():
            raise ValueError(f"{column} is required and cannot contain missing values")
    for column in ("question_id", "turn"):
        if not out[column].map(
            lambda x: isinstance(x, Integral) and not isinstance(x, bool) and x > 0
        ).all():
            raise ValueError(f"{column} must contain positive integer identifiers")
    if not out["judge"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("judge must contain nonempty annotator identifiers")
    conversation_columns = {"conversation_a", "conversation_b"} & set(out.columns)
    if conversation_columns and len(conversation_columns) != 2:
        raise ValueError("Both conversation_a and conversation_b are required together")
    if conversation_columns:
        a = out["conversation_a"].map(_response_digest)
        b = out["conversation_b"].map(_response_digest)
        out["_response_first"] = a.where(~out["input_reversed"], b)
        out["_response_second"] = b.where(~out["input_reversed"], a)
        signature_counts = out.groupby(_KEYS)[
            ["_response_first", "_response_second"]
        ].nunique()
        if (signature_counts > 1).any().any():
            raise ValueError("A canonical comparison key identifies different responses")
    return out


def _deduplicate_human(human: pd.DataFrame) -> pd.DataFrame:
    keys = _KEYS + ["judge"]
    if (human.groupby(keys)["label"].nunique() > 1).any():
        raise ValueError("Conflicting human votes from the same annotator and comparison")
    return human.drop_duplicates(keys).copy()


def _majority(labels: pd.Series) -> str:
    counts = labels.value_counts()
    top = counts[counts == counts.max()].index
    return str(top[0]) if len(top) == 1 else "tie"


def _load_source(revision: str):
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a nonempty Hugging Face revision")
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The MT-Bench loader needs the `datasets` package: "
            "pip install judgecal[data]"
        ) from e
    return load_dataset(MT_BENCH_DATASET, revision=revision, token=False)


def _provenance(revision: str) -> dict:
    return {
        "dataset_id": MT_BENCH_DATASET,
        "dataset_revision": revision,
        "dataset_url": f"https://huggingface.co/datasets/{MT_BENCH_DATASET}/tree/{revision}",
        "license": "CC-BY-4.0",
        "citation": "Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, NeurIPS 2023",
        "deduplication_policy": _DEDUPLICATION,
        "human_vote_aggregation": _AGGREGATION,
        "tie_policy": "Both tie and tie (inconsistent) map to tie; the latter flag is retained.",
        "order_diagnostics": (
            "The source contains a combined GPT-4 judgment, not its separate order-specific votes. "
            "input_reversed is source ordering versus canonical ordering, not a position-bias estimate."
        ),
    }


def load_mt_bench(
    min_human_votes: int = 1, *, revision: str = MT_BENCH_REVISION,
) -> pd.DataFrame:
    """Load pinned public MT-Bench judgments, aligned by question/turn/model pair.

    Core columns remain ``question_id, turn, model_a, model_b, human_winner,
    n_human_votes, gpt4_winner``. A/B always refer to the alphabetically sorted
    model identities. Human labels use a unique plurality, with vote-count
    ties resolved to ``tie``; ``n_human_votes`` counts distinct annotators.
    Additional columns retain human vote counts and GPT-4 source diagnostics.
    ``DataFrame.attrs`` records revision, attribution, filtering and join counts.

    The source is CC-BY-4.0 (separate from this library's license). No model
    API is called. Split evaluation by question_id: turns, model pairs and
    repeated annotations from one prompt must stay in the same partition.
    """
    if isinstance(min_human_votes, bool) or not isinstance(min_human_votes, Integral) or min_human_votes < 1:
        raise ValueError("min_human_votes must be a positive integer")
    ds = _load_source(revision)
    human_raw = _prepare_split(ds["human"].to_pandas())
    human = _deduplicate_human(human_raw)
    gpt4 = _prepare_split(ds["gpt4_pair"].to_pandas())
    if gpt4.duplicated(_KEYS).any():
        raise ValueError("GPT-4 split must contain one combined judgment per canonical comparison")
    response_checked = "_response_first" in human and "_response_first" in gpt4
    if response_checked:
        responses = human.drop_duplicates(_KEYS).merge(gpt4, on=_KEYS, suffixes=("_h", "_g"))
        if any((responses[f"{col}_h"] != responses[f"{col}_g"]).any()
               for col in ("_response_first", "_response_second")):
            raise ValueError("Human and GPT-4 comparison keys refer to different responses")
    human_agg = human.groupby(_KEYS)["label"].agg(
        human_winner=_majority, n_human_votes="size",
        n_human_a=lambda x: int((x == "A").sum()),
        n_human_b=lambda x: int((x == "B").sum()),
        n_human_ties=lambda x: int((x == "tie").sum()),
    ).reset_index()
    gpt4_agg = gpt4[_KEYS + ["label", "winner", "inconsistent_order", "input_reversed"]].rename(
        columns={"label": "gpt4_winner", "winner": "gpt4_raw_winner",
                 "inconsistent_order": "gpt4_inconsistent_order", "input_reversed": "gpt4_input_reversed"}
    )
    merged = human_agg.merge(gpt4_agg, on=_KEYS, how="inner", validate="one_to_one")
    counts = {
        "human_raw_rows": len(human_raw), "human_unique_votes": len(human),
        "human_duplicates_removed": len(human_raw) - len(human),
        "human_comparisons": len(human_agg), "gpt4_rows": len(gpt4),
        "aligned_comparisons_before_filter": len(merged),
        "human_only_comparisons": len(human_agg) - len(merged),
        "gpt4_only_comparisons": len(gpt4) - len(merged),
        "gpt4_inconsistent_order_rows": int(gpt4["inconsistent_order"].sum()),
        "aligned_inconsistent_order_before_filter": int(merged["gpt4_inconsistent_order"].sum()),
    }
    merged = merged.loc[merged["n_human_votes"] >= min_human_votes].copy()
    counts["aligned_comparisons_after_filter"] = len(merged)
    counts["aligned_questions_after_filter"] = int(merged["question_id"].nunique())
    counts["aligned_inconsistent_order_after_filter"] = int(merged["gpt4_inconsistent_order"].sum())
    merged = merged.rename(columns={"m_first": "model_a", "m_second": "model_b"}).reset_index(drop=True)
    core = ["question_id", "turn", "model_a", "model_b", "human_winner", "n_human_votes", "gpt4_winner"]
    merged = merged[core + [c for c in merged if c not in core]]
    merged.attrs = {**_provenance(revision), "source_counts": counts,
                    "min_human_votes": int(min_human_votes), "response_alignment_checked": response_checked}
    return merged


def load_mt_bench_annotator_votes(*, revision: str = MT_BENCH_REVISION) -> pd.DataFrame:
    """Canonical human votes, with identical repeated annotator votes removed.

    The first six columns preserve the original API: question_id, turn,
    model_a, model_b, judge, label. Raw winner/order diagnostics follow.
    Includes author_* and expert_* annotators; no inter-rater noise-ceiling
    claim follows from this dataset or overlapping agreement intervals.
    """
    ds = _load_source(revision)
    raw = _prepare_split(ds["human"].to_pandas())
    human = _deduplicate_human(raw)
    out = human[_KEYS + ["judge", "label", "winner", "input_reversed", "inconsistent_order"]].rename(
        columns={"m_first": "model_a", "m_second": "model_b", "winner": "raw_winner"}
    ).reset_index(drop=True)
    out.attrs = {**_provenance(revision), "source_counts": {
        "human_raw_rows": len(raw), "human_unique_votes": len(out),
        "human_duplicates_removed": len(raw) - len(out),
    }}
    return out


def make_synthetic(
    n: int = 500,
    true_win_rate: float = 0.6,
    judge_sensitivity: float = 0.85,
    judge_specificity: float = 0.80,
    tie_rate: float = 0.1,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Synthetic aligned human/judge pairwise labels with known ground truth.

    Human labels are drawn with P(A) = true_win_rate (before ties); the judge
    reproduces the human label with the given sensitivity (on A) and
    specificity (on B). Useful for tests and for validating the
    measurement-error correction recovers `true_win_rate`.
    """
    rng = np.random.default_rng(random_state)
    human = np.where(rng.random(n) < true_win_rate, "A", "B")
    judge = np.empty(n, dtype=object)
    is_a = human == "A"
    judge[is_a] = np.where(rng.random(is_a.sum()) < judge_sensitivity, "A", "B")
    judge[~is_a] = np.where(rng.random((~is_a).sum()) < judge_specificity, "B", "A")
    human = human.astype(object)
    tie_mask = rng.random(n) < tie_rate
    human[tie_mask] = "tie"
    return pd.DataFrame({"human_winner": human, "judge_winner": judge})
