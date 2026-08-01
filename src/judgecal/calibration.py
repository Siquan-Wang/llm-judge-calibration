"""Calibrating an LLM judge to human labels and correcting for judge error.

Two complementary tools:
  * `judge_confusion` / `rogan_gladen_correction` treat the judge as a noisy
    binary classifier of the human label and correct an observed win-rate for
    the judge's measurement error (sensitivity/specificity).
  * `platt_scaling` / `isotonic_calibration` map continuous judge scores to
    calibrated probabilities that a human prefers the item.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import optimize


@dataclass
class JudgeConfusion:
    sensitivity: float  # P(judge = A | human = A)
    specificity: float  # P(judge = B | human = B)
    accuracy: float
    n: int

    @property
    def youden_j(self) -> float:
        """Youden's J = se + sp - 1; the denominator of the Rogan-Gladen fix."""
        return self.sensitivity + self.specificity - 1.0


def judge_confusion(
    judge: Sequence,
    human: Sequence,
    positive: str = "A",
    negative: str = "B",
) -> JudgeConfusion:
    """Estimate judge sensitivity/specificity treating human labels as truth.

    Ties on either side are dropped; only clean `positive`/`negative` pairs
    are used.
    """
    j = np.asarray(list(judge), dtype=object)
    h = np.asarray(list(human), dtype=object)
    mask = np.isin(h, [positive, negative]) & np.isin(j, [positive, negative])
    j, h = j[mask], h[mask]
    pos = h == positive
    neg = h == negative
    se = float(np.mean(j[pos] == positive)) if pos.sum() else float("nan")
    sp = float(np.mean(j[neg] == negative)) if neg.sum() else float("nan")
    acc = float(np.mean(j == h)) if h.size else float("nan")
    return JudgeConfusion(se, sp, acc, int(h.size))


def rogan_gladen_correction(
    observed_rate: float,
    sensitivity: float,
    specificity: float,
) -> float:
    """Correct an observed positive rate for a noisy classifier's error.

    The Rogan-Gladen estimator inverts the measurement model
    p_obs = se * p_true + (1 - sp) * (1 - p_true):
        p_true = (p_obs + sp - 1) / (se + sp - 1)
    Result is clipped to [0, 1]. Returns NaN when the judge is uninformative
    (se + sp == 1).
    """
    denom = sensitivity + specificity - 1.0
    if abs(denom) < 1e-12:
        return float("nan")
    corrected = (observed_rate + specificity - 1.0) / denom
    return float(min(1.0, max(0.0, corrected)))


def platt_scaling(
    scores: Sequence[float],
    labels: Sequence[int],
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a 1-D logistic map sigmoid(a*score + b) by maximum likelihood.

    `labels` are binary (1 = human prefers the item). Returns a function that
    maps raw scores to calibrated probabilities.
    """
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)

    def nll(params):
        a, b = params
        z = a * x + b
        # numerically stable log-loss
        log_p = -np.logaddexp(0.0, -z)
        log_1mp = -np.logaddexp(0.0, z)
        return -np.sum(y * log_p + (1 - y) * log_1mp)

    res = optimize.minimize(nll, x0=np.array([1.0, 0.0]), method="BFGS")
    a, b = res.x

    def calibrate(new_scores):
        z = a * np.asarray(new_scores, dtype=float) + b
        return 1.0 / (1.0 + np.exp(-z))

    calibrate.params_ = (float(a), float(b))  # type: ignore[attr-defined]
    return calibrate


def _pava_expand(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators isotonic regression, one fitted value per point."""
    n = len(y)
    # blocks: list of [sum_wy, sum_w, start, end]
    blocks = [[y[i] * w[i], w[i], i, i] for i in range(n)]
    i = 0
    while i < len(blocks) - 1:
        mean_i = blocks[i][0] / blocks[i][1]
        mean_next = blocks[i + 1][0] / blocks[i + 1][1]
        if mean_i > mean_next:
            blocks[i][0] += blocks[i + 1][0]
            blocks[i][1] += blocks[i + 1][1]
            blocks[i][3] = blocks[i + 1][3]
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    fitted = np.empty(n)
    for swy, sw, start, end in blocks:
        fitted[start : end + 1] = swy / sw
    return fitted


def isotonic_calibration(
    scores: Sequence[float],
    labels: Sequence[int],
) -> Callable[[np.ndarray], np.ndarray]:
    """Monotone (isotonic) calibration of judge scores to human-preference rate."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    fitted = _pava_expand(ys, np.ones_like(ys))

    def calibrate(new_scores):
        ns = np.asarray(new_scores, dtype=float)
        return np.interp(ns, xs, fitted, left=fitted[0], right=fitted[-1])

    return calibrate
