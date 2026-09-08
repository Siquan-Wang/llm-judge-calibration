"""Agreement and correlation metrics between an LLM judge and human labels."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

from .intervals import (
    Interval, beta_binomial_interval, bootstrap_ci, _as_1d,
    _binomial_exact_interval, _bounded_mean_interval, _validate_alpha,
    _validate_n_boot, _validate_probability,
)

_PAIRWISE = ("A", "B", "tie")


def _as_array(labels: Sequence) -> np.ndarray:
    return _as_1d(labels, "labels", dtype=object)


def _pairwise_array(labels: Sequence, name: str = "labels") -> np.ndarray:
    arr = _as_1d(labels, name, dtype=object)
    if not np.all(np.isin(arr, _PAIRWISE)):
        raise ValueError(f"{name} must contain only A, B, or tie")
    return arr


def _validate_target(target: str) -> None:
    if target not in ("A", "B"):
        raise ValueError("target must be A or B")


def agreement_rate(
    judge: Sequence,
    human: Sequence,
    exclude_ties: bool = True,
) -> float:
    """Fraction of items where judge and human labels match.

    When `exclude_ties` is True, items where either side is "tie" are dropped
    before computing agreement (the common protocol for pairwise judging).
    """
    j = _pairwise_array(judge, "judge")
    h = _pairwise_array(human, "human")
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
    j = _pairwise_array(judge, "judge")
    h = _pairwise_array(human, "human")
    if j.shape != h.shape:
        raise ValueError("judge and human must have the same length")
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
    if a.size == 0:
        return float("nan")
    if any(v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)) for v in [*a, *b]):
        raise ValueError("labels must not contain missing or non-finite values")
    try:
        labels = list(dict.fromkeys([*a, *b]))
    except TypeError as exc:
        raise ValueError("labels must be hashable scalar values") from exc
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
        # Agreement is certain under the marginals; kappa is 0/0.
        return float("nan")
    return (po - pe) / (1 - pe)


def score_correlation(
    judge_scores: Sequence[float],
    human_scores: Sequence[float],
    method: str = "spearman",
) -> float:
    """Correlation between scalar judge scores and human scores."""
    js = _as_1d(judge_scores, "judge_scores", dtype=float)
    hs = _as_1d(human_scores, "human_scores", dtype=float)
    if js.shape != hs.shape or len(js) < 2:
        raise ValueError("scores must have the same length and at least two observations")
    if not np.all(np.isfinite(js)) or not np.all(np.isfinite(hs)):
        raise ValueError("scores must be finite")
    if method not in ("spearman", "pearson"):
        raise ValueError("method must be 'spearman' or 'pearson'")
    if np.ptp(js) == 0 or np.ptp(hs) == 0:
        return float("nan")
    if method == "spearman":
        return float(stats.spearmanr(js, hs)[0])
    if method == "pearson":
        return float(stats.pearsonr(js, hs)[0])
    raise ValueError("method must be 'spearman' or 'pearson'")


def win_rate(labels: Sequence, target: str = "A", tie_value: float = 0.5) -> float:
    """Win-rate of `target` from pairwise labels; ties count as `tie_value`."""
    _validate_target(target)
    tie_value = _validate_probability(tie_value, "tie_value")
    arr = _pairwise_array(labels)
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
    """Win-rate interval for iid comparisons (ties counted as 0.5).

    Binary samples use an exact binomial interval. Samples containing ties
    use an approximate percentile bootstrap, with a conservative bounded
    interval for constant samples instead of a zero-width confidence claim.
    Repeated questions require cluster-aware inference outside this API.
    """
    _validate_target(target)
    alpha = _validate_alpha(alpha)
    n_boot = _validate_n_boot(n_boot)
    arr = _pairwise_array(labels)
    numeric = np.where(arr == target, 1.0, np.where(arr == "tie", 0.5, 0.0))
    if len(arr) and not np.any(arr == "tie"):
        return _binomial_exact_interval(int(np.sum(arr == target)), len(arr), alpha)
    if len(arr) and np.ptp(numeric) == 0:
        return _bounded_mean_interval(numeric, 0.0, 1.0, alpha)
    return bootstrap_ci(numeric, np.mean, n_boot=n_boot, alpha=alpha, random_state=random_state)
