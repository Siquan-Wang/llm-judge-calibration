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
from numbers import Real
from typing import Callable, Sequence

import numpy as np
from scipy import optimize
from scipy.special import expit

from .intervals import _as_1d, _validate_probability


@dataclass
class JudgeConfusion:
    sensitivity: float  # P(judge = A | human = A)
    specificity: float  # P(judge = B | human = B)
    accuracy: float
    n: int
    positive: str = "A"
    negative: str = "B"
    n_positive: int = 0
    n_negative: int = 0
    n_dropped: int = 0

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

    Ties on either side are dropped for this descriptive statistic, and the
    dropped count is reported. It therefore describes the retained subset,
    not general human preferences with ties. Missing human classes yield
    NaN rates; they cannot be used for binary measurement-error correction.
    """
    if not isinstance(positive, str) or not isinstance(negative, str) or not positive or not negative or positive == negative or positive == "tie" or negative == "tie":
        raise ValueError("positive and negative must be distinct non-tie labels")
    j = _as_1d(judge, "judge", dtype=object)
    h = _as_1d(human, "human", dtype=object)
    if j.shape != h.shape:
        raise ValueError("judge and human must have the same length")
    for arr in (j, h):
        if not np.all(np.isin(arr, [positive, negative, "tie"])):
            raise ValueError("judge and human contain an unknown label")
    mask = np.isin(h, [positive, negative]) & np.isin(j, [positive, negative])
    n_dropped = int(np.sum(~mask))
    j, h = j[mask], h[mask]
    pos = h == positive
    neg = h == negative
    se = float(np.mean(j[pos] == positive)) if pos.sum() else float("nan")
    sp = float(np.mean(j[neg] == negative)) if neg.sum() else float("nan")
    acc = float(np.mean(j == h)) if h.size else float("nan")
    return JudgeConfusion(se, sp, acc, int(h.size), positive, negative,
                          int(pos.sum()), int(neg.sum()), n_dropped)


def rogan_gladen_correction(
    observed_rate: float,
    sensitivity: float,
    specificity: float,
    *,
    clip: bool = True,
) -> float:
    """Correct an observed positive rate for a noisy classifier's error.

    The Rogan-Gladen estimator inverts the measurement model
    p_obs = se * p_true + (1 - sp) * (1 - p_true):
        p_true = (p_obs + sp - 1) / (se + sp - 1)
    Result is clipped to [0, 1] by default; ``clip=False`` exposes an
    out-of-range estimate as a diagnostic. Returns NaN when se + sp == 1.
    This plug-in ratio is not generally unbiased with estimated error
    rates, and clipping adds bias. Error rates must transfer to the target
    population. This binary model does not handle ties or abstentions.
    """
    observed_rate = _validate_probability(observed_rate, "observed_rate")
    sensitivity = _validate_probability(sensitivity, "sensitivity")
    specificity = _validate_probability(specificity, "specificity")
    if not isinstance(clip, (bool, np.bool_)):
        raise ValueError("clip must be a boolean")
    denom = sensitivity + specificity - 1.0
    if abs(denom) < 1e-12:
        return float("nan")
    corrected = (observed_rate + specificity - 1.0) / denom
    return float(np.clip(corrected, 0.0, 1.0) if clip else corrected)


def _calibration_arrays(scores, labels):
    x = _as_1d(scores, "scores", dtype=float)
    y = _as_1d(labels, "labels", dtype=float)
    if len(x) == 0 or x.shape != y.shape:
        raise ValueError("scores and labels must be nonempty and have the same length")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("scores and labels must be finite")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("labels must be binary 0 or 1")
    return x, y


def _prediction_scores(scores):
    try:
        arr = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("prediction scores must be finite scalars or a 1-D array") from exc
    if arr.ndim > 1 or not np.all(np.isfinite(arr)):
        raise ValueError("prediction scores must be finite scalars or a 1-D array")
    return arr


def platt_scaling(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    regularization: float = 1e-6,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a 1-D logistic map with a small L2 slope penalty.

    `labels` are binary (1 = human prefers the item). Returns a function that
    maps raw scores to calibrated probabilities. The strictly positive
    penalty keeps the slope finite under separation; both classes are
    required. Fit only on calibration data and evaluate on separate data.
    """
    x, y = _calibration_arrays(scores, labels)
    if len(np.unique(y)) != 2:
        raise ValueError("Platt scaling requires both label classes")
    if isinstance(regularization, (bool, np.bool_)) or not isinstance(regularization, Real) or not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and strictly positive")

    def nll(params):
        a, b = params
        z = a * x + b
        # numerically stable log-loss
        log_p = -np.logaddexp(0.0, -z)
        log_1mp = -np.logaddexp(0.0, z)
        return -np.sum(y * log_p + (1 - y) * log_1mp) + 0.5 * regularization * a**2

    def gradient(params):
        a, b = params
        residual = expit(a * x + b) - y
        return np.array([np.dot(residual, x) + regularization * a, np.sum(residual)])

    res = optimize.minimize(nll, x0=np.array([1.0, 0.0]), jac=gradient, method="BFGS")
    if not res.success or not np.all(np.isfinite(res.x)):
        raise RuntimeError(f"Platt scaling optimization failed: {res.message}")
    a, b = res.x

    def calibrate(new_scores):
        z = a * _prediction_scores(new_scores) + b
        return expit(z)

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
    x, y = _calibration_arrays(scores, labels)
    # Equal scores must share a fitted probability. Pool before PAVA so
    # permuting labels within a tied score cannot change predictions.
    xs, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ys = np.bincount(inverse, weights=y) / counts
    fitted = _pava_expand(ys, counts.astype(float))

    def calibrate(new_scores):
        ns = _prediction_scores(new_scores)
        return np.interp(ns, xs, fitted, left=fitted[0], right=fitted[-1])

    return calibrate
