"""Model-vs-model comparison from pairwise judge labels, with uncertainty."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .calibration import JudgeConfusion, rogan_gladen_correction
from .intervals import Interval, beta_binomial_interval
from .metrics import win_rate, win_rate_with_ci


def compare_models(
    judge_labels: Sequence,
    target: str = "A",
    alpha: float = 0.05,
    confusion: JudgeConfusion | None = None,
    n_boot: int = 2000,
    random_state: int | None = None,
) -> dict:
    """Win-rate of `target` with bootstrap CI and an optional error-corrected estimate.

    If `confusion` (judge sensitivity/specificity vs. humans, estimated on a
    held-out human-labeled set) is provided, the observed judge win-rate is
    additionally corrected with the Rogan-Gladen estimator, so the reported
    number estimates the *human-judged* win-rate rather than the raw judge one.

    Returns a dict with keys: win_rate, ci, n, significant (CI excludes 0.5),
    and optionally corrected_win_rate.
    """
    labels = list(judge_labels)
    ci: Interval = win_rate_with_ci(
        labels, target=target, alpha=alpha, n_boot=n_boot, random_state=random_state
    )
    result = {
        "win_rate": win_rate(labels, target=target),
        "ci": ci.as_dict(),
        "n": len(labels),
        "significant": ci.excludes(0.5),
    }
    if confusion is not None:
        arr = np.asarray(labels, dtype=object)
        non_tie = arr[arr != "tie"]
        observed = float(np.mean(non_tie == target)) if non_tie.size else float("nan")
        result["corrected_win_rate"] = rogan_gladen_correction(
            observed, confusion.sensitivity, confusion.specificity
        )
    return result


def paired_agreement_gap(
    judge_a: Sequence,
    judge_b: Sequence,
    human: Sequence,
    alpha: float = 0.05,
) -> dict:
    """Which of two judges agrees with humans more, with a CI on the gap.

    Uses the per-item paired difference in agreement indicators and a
    Beta-Binomial interval on each judge's agreement.
    """
    ja = np.asarray(list(judge_a), dtype=object)
    jb = np.asarray(list(judge_b), dtype=object)
    h = np.asarray(list(human), dtype=object)
    mask = (h != "tie") & (ja != "tie") & (jb != "tie")
    ja, jb, h = ja[mask], jb[mask], h[mask]
    n = int(h.size)
    ka = int(np.sum(ja == h))
    kb = int(np.sum(jb == h))
    return {
        "judge_a_agreement": beta_binomial_interval(ka, n, alpha=alpha).as_dict(),
        "judge_b_agreement": beta_binomial_interval(kb, n, alpha=alpha).as_dict(),
        "gap": (ka - kb) / n if n else float("nan"),
        "n": n,
    }
