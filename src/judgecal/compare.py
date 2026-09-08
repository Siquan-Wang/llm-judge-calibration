"""Model-vs-model comparison from pairwise judge labels, with uncertainty."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .calibration import JudgeConfusion, judge_confusion, rogan_gladen_correction
from .intervals import (Interval, beta_binomial_interval, bootstrap_ci,
                        _bounded_mean_interval, _validate_alpha, _validate_n_boot)
from .metrics import win_rate, win_rate_with_ci, _pairwise_array, _validate_target


def _target_rates(confusion: JudgeConfusion, target: str) -> tuple[float, float]:
    if not isinstance(confusion.positive, str) or not isinstance(confusion.negative, str) or {confusion.positive, confusion.negative} != {"A", "B"}:
        raise ValueError("confusion must describe the A/B labels")
    if confusion.n_dropped:
        raise ValueError("Rogan-Gladen correction requires binary calibration data without dropped ties")
    if target == confusion.positive:
        return confusion.sensitivity, confusion.specificity
    return confusion.specificity, confusion.sensitivity


def _joint_correction_interval(judge, human, evaluation, target, point, alpha,
                               n_boot, random_state):
    """Independent calibration/evaluation bootstrap, retaining failure diagnostics."""
    conf = judge_confusion(judge, human)
    se, sp = _target_rates(conf, target)
    observed = float(np.mean(evaluation == target))
    diagnostics = {
        "method": "joint-calibration-bootstrap", "status": "ok",
        "calibration_n": len(judge), "evaluation_n": len(evaluation),
        "requested_draws": n_boot, "attempted_draws": 0,
        "valid_draws": 0, "invalid_draws": 0, "out_of_range_draws": 0,
        "invalid_reasons": {},
        "assumptions": "Independent iid calibration and evaluation samples; transferable binary error rates",
    }
    # Ordinary resampling cannot estimate rate uncertainty at empirical
    # boundaries. Do not manufacture a zero-width calibrated interval.
    if not np.isfinite(point):
        diagnostics["status"] = "unidentifiable"
        return None, diagnostics
    if not 0 <= point <= 1:
        diagnostics["status"] = "out-of-range-estimate"
        return None, diagnostics
    if any(rate in (0.0, 1.0) for rate in (observed, se, sp)):
        diagnostics["status"] = "boundary-rate-bootstrap-unavailable"
        return None, diagnostics
    rng = np.random.default_rng(random_state)
    samples = []
    sign = np.sign(se + sp - 1)
    for _ in range(n_boot):
        diagnostics["attempted_draws"] += 1
        indices = rng.integers(0, len(judge), size=len(judge))
        draw = judge_confusion(judge[indices], human[indices])
        reason = None
        if not np.isfinite(draw.sensitivity) or not np.isfinite(draw.specificity):
            reason = "missing-human-class"
        elif abs(draw.youden_j) < 1e-12:
            reason = "uninformative-confusion"
        elif np.sign(draw.youden_j) != sign:
            reason = "reversed-youden-sign"
        if reason:
            diagnostics["invalid_draws"] += 1
            reasons = diagnostics["invalid_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        eval_draw = evaluation[rng.integers(0, len(evaluation), size=len(evaluation))]
        draw_se, draw_sp = _target_rates(draw, target)
        value = rogan_gladen_correction(float(np.mean(eval_draw == target)), draw_se, draw_sp, clip=False)
        samples.append(value)
        diagnostics["out_of_range_draws"] += int(not 0 <= value <= 1)
    diagnostics["valid_draws"] = len(samples)
    # Dropping failed draws would condition the interval on identifiable
    # replicates. Report the failure instead of silently doing so.
    if diagnostics["invalid_draws"]:
        diagnostics["status"] = "unstable-bootstrap"
        return None, diagnostics
    if np.ptp(samples) == 0:
        diagnostics["status"] = "degenerate-bootstrap"
        return None, diagnostics
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    ci = Interval(float(point), float(np.clip(low, 0, 1)), float(np.clip(high, 0, 1)),
                  alpha, "joint-calibration-bootstrap")
    return ci, diagnostics


def compare_models(
    judge_labels: Sequence,
    target: str = "A",
    alpha: float = 0.05,
    confusion: JudgeConfusion | None = None,
    n_boot: int = 2000,
    random_state: int | None = None,
    *,
    calibration_judge: Sequence | None = None,
    calibration_human: Sequence | None = None,
) -> dict:
    """Raw judge win-rate, plus optional *binary* human-rate correction.

    Legacy ``win_rate``, ``ci`` and ``significant`` ALWAYS describe the raw
    judge (ties score 0.5); explicit ``raw_*`` aliases make this distinction
    visible. ``significant`` is two-sided CI exclusion of 0.5, not a claim
    that the target wins and not a corrected significance statement.

    Supply either a summary ``confusion`` (point correction only), or both
    ``calibration_judge`` and ``calibration_human`` for an approximate joint
    bootstrap. Calibration must be independent of evaluation; both samples
    must be iid binary A/B comparisons with transferable error rates. Ties
    in either sample are rejected rather than silently changing estimands.
    Overlapping samples or repeated prompts need other inference methods.

    Corrected uncertainty is separately returned in ``corrected_ci`` and
    ``corrected_significant``. It is unavailable (None) for summary-only,
    empirical-boundary, unidentified, or unstable bootstrap cases. Consult
    ``correction_diagnostics``; failed resamples are never silently dropped.
    """
    _validate_target(target)
    alpha = _validate_alpha(alpha)
    n_boot = _validate_n_boot(n_boot)
    labels = _pairwise_array(judge_labels, "judge_labels")
    ci: Interval = win_rate_with_ci(
        labels, target=target, alpha=alpha, n_boot=n_boot, random_state=random_state
    )
    result = {
        "win_rate": win_rate(labels, target=target),
        "ci": ci.as_dict(),
        "n": len(labels),
        "significant": ci.excludes(0.5),
    }
    result.update(raw_win_rate=result["win_rate"], raw_ci=result["ci"],
                  raw_significant=result["significant"],
                  raw_estimand="judge win-rate; ties score 0.5")
    supplied_pairs = calibration_judge is not None or calibration_human is not None
    if supplied_pairs and (calibration_judge is None or calibration_human is None):
        raise ValueError("supply both calibration_judge and calibration_human")
    if supplied_pairs and confusion is not None:
        raise ValueError("supply calibration pairs or confusion, not both")
    if supplied_pairs:
        cj = _pairwise_array(calibration_judge, "calibration_judge")
        ch = _pairwise_array(calibration_human, "calibration_human")
        if cj.shape != ch.shape or len(cj) == 0:
            raise ValueError("calibration arrays must be nonempty and have the same length")
        if np.any(cj == "tie") or np.any(ch == "tie"):
            raise ValueError("Rogan-Gladen correction requires binary calibration labels; ties are not supported")
        confusion = judge_confusion(cj, ch)
    if confusion is not None:
        if len(labels) == 0 or np.any(labels == "tie"):
            raise ValueError("Rogan-Gladen correction requires nonempty binary evaluation labels; ties are not supported")
        se, sp = _target_rates(confusion, target)
        observed = float(np.mean(labels == target))
        unbounded = rogan_gladen_correction(observed, se, sp, clip=False)
        result.update(corrected_win_rate=float(np.clip(unbounded, 0, 1)),
                      corrected_win_rate_unclipped=unbounded,
                      corrected_ci=None, corrected_significant=None,
                      corrected_estimand="binary human preference rate under transferable judge error rates")
        diagnostics = {"status": "point-only", "method": "summary-confusion",
                       "youden_j": confusion.youden_j,
                       "estimate_clipped": bool(np.isfinite(unbounded) and not 0 <= unbounded <= 1),
                       "calibration_n": confusion.n}
        if not np.isfinite(unbounded):
            diagnostics["status"] = "unidentifiable"
        if supplied_pairs:
            corrected_ci, detail = _joint_correction_interval(
                cj, ch, labels, target, unbounded, alpha, n_boot, random_state)
            diagnostics.update(detail)
            if corrected_ci is not None:
                result["corrected_ci"] = corrected_ci.as_dict()
                result["corrected_significant"] = corrected_ci.excludes(0.5)
        result["correction_diagnostics"] = diagnostics
    return result


def paired_agreement_gap(
    judge_a: Sequence,
    judge_b: Sequence,
    human: Sequence,
    alpha: float = 0.05,
    n_boot: int = 2000,
    random_state: int | None = None,
) -> dict:
    """Which of two judges agrees with humans more, with a CI on the gap.

    The gap CI resamples paired per-item differences; the two marginal
    intervals are separately labeled Beta posterior credible intervals.
    Ties on any side are excluded, so the estimand is conditional on all
    three labels being non-ties. This assumes independent items. Constant
    differences use a conservative bounded-mean fallback.
    """
    alpha = _validate_alpha(alpha)
    n_boot = _validate_n_boot(n_boot)
    ja = _pairwise_array(judge_a, "judge_a")
    jb = _pairwise_array(judge_b, "judge_b")
    h = _pairwise_array(human, "human")
    if ja.shape != jb.shape or ja.shape != h.shape:
        raise ValueError("all three label arrays must have the same length")
    original_n = len(h)
    mask = (h != "tie") & (ja != "tie") & (jb != "tie")
    ja, jb, h = ja[mask], jb[mask], h[mask]
    n = int(h.size)
    ka = int(np.sum(ja == h))
    kb = int(np.sum(jb == h))
    differences = (ja == h).astype(float) - (jb == h).astype(float)
    if n == 0 or np.ptp(differences) == 0:
        gap_ci = _bounded_mean_interval(differences, -1.0, 1.0, alpha)
    else:
        gap_ci = bootstrap_ci(differences, n_boot=n_boot, alpha=alpha, random_state=random_state)
        gap_ci.method = "paired-bootstrap"
    return {
        "judge_a_agreement": beta_binomial_interval(ka, n, alpha=alpha).as_dict(),
        "judge_b_agreement": beta_binomial_interval(kb, n, alpha=alpha).as_dict(),
        "gap": (ka - kb) / n if n else float("nan"),
        "gap_ci": gap_ci.as_dict(),
        "gap_significant": gap_ci.excludes(0.0) if n else None,
        "n": n,
        "n_dropped": original_n - n,
        "estimand": "agreement gap conditional on all three labels being non-ties",
    }
