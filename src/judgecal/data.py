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

import numpy as np
import pandas as pd


def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the model pair alphabetically and re-express winner as A/B/tie."""
    first = df[["model_a", "model_b"]].min(axis=1)
    second = df[["model_a", "model_b"]].max(axis=1)
    winner = df["winner"].astype(str)
    label = np.where(
        winner.str.startswith("tie"),
        "tie",
        np.where(
            (winner == "model_a") == (df["model_a"] == first),
            "A",
            "B",
        ),
    )
    out = df.copy()
    out["m_first"] = first
    out["m_second"] = second
    out["label"] = label
    return out


def load_mt_bench(min_human_votes: int = 1) -> pd.DataFrame:
    """Aligned GPT-4 vs. human majority votes on MT-Bench pairwise comparisons.

    Returns a DataFrame with one row per (question_id, turn, model pair):
      question_id, turn, model_a, model_b,
      human_winner  -- majority vote over human annotators ('A'/'B'/'tie'),
      n_human_votes -- number of human votes behind the majority label,
      gpt4_winner   -- GPT-4 pairwise vote ('A'/'B'/'tie').
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The MT-Bench loader needs the `datasets` package: "
            "pip install judgecal[data]"
        ) from e

    ds = load_dataset("lmsys/mt_bench_human_judgments")
    human = _canonicalize(ds["human"].to_pandas())
    gpt4 = _canonicalize(ds["gpt4_pair"].to_pandas())

    keys = ["question_id", "turn", "m_first", "m_second"]

    def majority(labels: pd.Series) -> str:
        counts = labels.value_counts()
        top = counts[counts == counts.max()].index.tolist()
        return top[0] if len(top) == 1 else "tie"

    human_agg = (
        human.groupby(keys)["label"]
        .agg([("human_winner", majority), ("n_human_votes", "size")])
        .reset_index()
    )
    gpt4_agg = (
        gpt4.groupby(keys)["label"]
        .agg([("gpt4_winner", majority)])
        .reset_index()
    )
    merged = human_agg.merge(gpt4_agg, on=keys, how="inner")
    merged = merged[merged["n_human_votes"] >= min_human_votes]
    merged = merged.rename(columns={"m_first": "model_a", "m_second": "model_b"})
    return merged.reset_index(drop=True)


def load_mt_bench_annotator_votes() -> pd.DataFrame:
    """Per-annotator human votes, for inter-annotator agreement analysis.

    Returns columns: question_id, turn, model_a, model_b, judge, label.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The MT-Bench loader needs the `datasets` package: "
            "pip install judgecal[data]"
        ) from e

    ds = load_dataset("lmsys/mt_bench_human_judgments")
    human = _canonicalize(ds["human"].to_pandas())
    human = human.rename(columns={"m_first": "pair_first", "m_second": "pair_second"})
    return human[
        ["question_id", "turn", "pair_first", "pair_second", "judge", "label"]
    ].rename(columns={"pair_first": "model_a", "pair_second": "model_b"})


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
