"""judgecal: statistically rigorous evaluation of LLM-as-a-judge pipelines."""

from .calibration import (
    JudgeConfusion,
    isotonic_calibration,
    judge_confusion,
    platt_scaling,
    rogan_gladen_correction,
)
from .compare import compare_models, paired_agreement_gap
from .intervals import Interval, beta_binomial_interval, bootstrap_ci, wilson_interval
from .metrics import (
    agreement_rate,
    agreement_with_ci,
    cohens_kappa,
    score_correlation,
    win_rate,
    win_rate_with_ci,
)
from .ppi import PPIWinRateResult, prediction_powered_win_rate

__version__ = "0.4.0"

__all__ = [
    "Interval",
    "JudgeConfusion",
    "PPIWinRateResult",
    "agreement_rate",
    "agreement_with_ci",
    "beta_binomial_interval",
    "bootstrap_ci",
    "cohens_kappa",
    "compare_models",
    "isotonic_calibration",
    "judge_confusion",
    "paired_agreement_gap",
    "platt_scaling",
    "prediction_powered_win_rate",
    "rogan_gladen_correction",
    "score_correlation",
    "win_rate",
    "win_rate_with_ci",
    "wilson_interval",
]
