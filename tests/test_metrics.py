import numpy as np
import pytest

from judgecal import (
    agreement_rate,
    agreement_with_ci,
    cohens_kappa,
    score_correlation,
    win_rate,
    win_rate_with_ci,
)


def test_agreement_exact():
    judge = ["A", "B", "A", "B"]
    human = ["A", "B", "B", "B"]
    assert agreement_rate(judge, human) == 0.75


def test_agreement_excludes_ties():
    judge = ["A", "tie", "B"]
    human = ["A", "A", "tie"]
    assert agreement_rate(judge, human, exclude_ties=True) == 1.0
    assert agreement_rate(judge, human, exclude_ties=False) == pytest.approx(1 / 3)


def test_agreement_ci_brackets_point():
    judge = ["A"] * 80 + ["B"] * 20
    human = ["A"] * 100
    ci = agreement_with_ci(judge, human)
    assert ci.low < 0.8 < ci.high


def test_kappa_perfect_and_chance():
    labels = ["A", "B", "A", "B", "tie", "A"]
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)
    # independent labels -> kappa near 0
    rng = np.random.default_rng(0)
    a = rng.choice(["A", "B"], size=5000)
    b = rng.choice(["A", "B"], size=5000)
    assert abs(cohens_kappa(a, b)) < 0.05


def test_win_rate_with_ties():
    labels = ["A", "A", "B", "tie"]
    assert win_rate(labels, target="A") == pytest.approx((2 + 0.5) / 4)


def test_win_rate_ci_significance():
    labels = ["A"] * 90 + ["B"] * 10
    ci = win_rate_with_ci(labels, random_state=0)
    assert ci.excludes(0.5)


def test_score_correlation():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0]
    assert score_correlation(x, y, "spearman") == pytest.approx(1.0)
    assert score_correlation(x, y, "pearson") == pytest.approx(1.0)
