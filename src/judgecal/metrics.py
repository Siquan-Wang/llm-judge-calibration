"""Agreement and correlation metrics between an LLM judge and human labels."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

from .intervals import Interval, beta_binomial_interval, bootstrap_ci

_PAIRWISE = ("A", "B", "tie")


def _as_array(labels: Sequence) -> np.ndarray:
    return np.asarray(list(labels), dtype=object)


def agreement_rate(
    judge: Sequence,
    human: Sequence,
    exclude_ties: bool = True,
) -> float:
    """Fraction of items where judge and human labels match.

    When `exclude_ties` is True, items where either side is "tie" are dropped
    before computing agreement (the common protocol for pairwise judging).
    """
    j = _as_array(judge)
    h = _as_array(human)
    if j.shape != h.shape:
        raise ValueError("judge and human must have the same length")
    mask = np.ones(j.shape, dtype=bool)
    if exclude_ties:
        mask = (j != "tie") & (h != "tie")
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(j[mask] == h[mask]))


def agreement_with_ci(
    judge: Sequence,
    human: Sequence,
    exclude_ties: bool = True,
    alpha: float = 0.05,
) -> Interval:
    """Judge/human agreement rate with a Beta-Binomial credible interval."""
    j = _as_array(judge)
    h = _as_array(human)
    mask = np.ones(j.shape, dtype=bool)
    if exclude_ties:
        mask = (j != "tie") & (h != "tie")
    n = int(mask.sum())
    k = int(np.sum(j[mask] == h[mask]))
    return beta_binomial_interval(k, n, alpha=alpha)


def cohens_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa between two label sequences (chance-corrected agreement)."""
    a = _as_array(a)
    b = _as_array(b)
    if a.shape != b.shape:
        raise ValueError("inputs must have the same length")
    labels = sorted(set(a.tolist()) | set(b.tolist()))
    idx = {lab: i for i, lab in enumerate(labels)}
    n = a.shape[0]
    conf = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        conf[idx[x], idx[y]] += 1
    po = np.trace(conf) / n
    row = conf.sum(axis=1) / n
    col = conf.sum(axis=0) / n
    pe = float(np.sum(row * col))
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def score_correlation(
    judge_scores: Sequence[float],
    human_scores: Sequence[float],
    method: str = "spearman",
) -> float:
    """Correlation between scalar judge scores and human scores."""
    js = np.asarray(judge_scores, dtype=float)
    hs = np.asarray(human_scores, dtype=float)
    if method == "spearman":
        return float(stats.spearmanr(js, hs).statistic)
    if method == "pearson":
        return float(stats.pearsonr(js, hs).statistic)
    raise ValueError("method must be 'spearman' or 'pearson'")


def win_rate(labels: Sequence, target: str = "A", tie_value: float = 0.5) -> float:
    """Win-rate of `target` from pairwise labels; ties count as `tie_value`."""
    arr = _as_array(labels)
    n = arr.shape[0]
    if n == 0:
        return float("nan")
    wins = np.sum(arr == target)
    ties = np.sum(arr == "tie")
    return float((wins + tie_value * ties) / n)


def win_rate_with_ci(
    labels: Sequence,
    target: str = "A",
    alpha: float = 0.05,
    n_boot: int = 2000,
    random_state: int | None = None,
) -> Interval:
    """Win-rate with a bootstrap CI (ties counted as 0.5)."""
    arr = _as_array(labels)
    numeric = np.where(arr == target, 1.0, np.where(arr == "tie", 0.5, 0.0))
    return bootstrap_ci(numeric, np.mean, n_boot=n_boot, alpha=alpha, random_state=random_state)
