import numpy as np
import pytest

from judgecal import beta_binomial_interval, bootstrap_ci, wilson_interval


def test_wilson_basic():
    ci = wilson_interval(50, 100)
    assert 0.0 <= ci.low < 0.5 < ci.high <= 1.0
    assert ci.point == 0.5


@pytest.mark.parametrize("alpha", [1e-20, 1e-100, np.nextafter(0.0, 1.0)])
def test_wilson_extreme_confidence_has_finite_score_bounds(alpha):
    with np.errstate(all="raise"):
        ci = wilson_interval(5000, 10000, alpha=alpha)
    assert 0 < ci.low < .5 < ci.high < 1
    assert ci.low + ci.high == pytest.approx(1)


def test_beta_upper_tail_does_not_round_to_one():
    ci = beta_binomial_interval(5000, 10000, alpha=1e-20)
    assert .4 < ci.low < .5 < ci.high < .6
    assert ci.low + ci.high == pytest.approx(1)


def test_wilson_extreme_counts_stay_in_unit_interval():
    ci = wilson_interval(0, 20)
    assert ci.low == 0.0
    assert ci.high < 0.25
    ci = wilson_interval(20, 20)
    assert ci.high == 1.0
    assert ci.low > 0.75


def test_beta_binomial_uniform_prior():
    ci = beta_binomial_interval(50, 100)
    assert ci.low < 0.5 < ci.high
    assert abs(ci.point - 51 / 102) < 1e-12
    # more data -> tighter interval
    wide = beta_binomial_interval(5, 10)
    narrow = beta_binomial_interval(500, 1000)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_beta_binomial_informative_prior_shrinks_point():
    flat = beta_binomial_interval(8, 10)
    skeptical = beta_binomial_interval(8, 10, prior_a=1, prior_b=9)
    assert skeptical.point < flat.point


def test_bootstrap_recovers_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=2.0, scale=1.0, size=500)
    ci = bootstrap_ci(data, np.mean, n_boot=1000, random_state=1)
    assert ci.low < 2.0 < ci.high
    assert abs(ci.point - data.mean()) < 1e-12


def test_interval_excludes():
    ci = beta_binomial_interval(90, 100)
    assert ci.excludes(0.5)
    assert not ci.excludes(0.9)


@pytest.mark.parametrize("alpha", [0, 1, -.1, 1.1, np.nan, np.inf, True])
def test_interval_alpha_validation(alpha):
    for function, args in [(wilson_interval, (1, 2)), (beta_binomial_interval, (1, 2)), (bootstrap_ci, ([0, 1],))]:
        with pytest.raises(ValueError):
            function(*args, alpha=alpha)


@pytest.mark.parametrize("k,n", [(-1, 2), (3, 2), (0, -1), (1.0, 2), (True, 2)])
def test_invalid_binomial_counts(k, n):
    for function in (wilson_interval, beta_binomial_interval):
        with pytest.raises(ValueError):
            function(k, n)


@pytest.mark.parametrize("n_boot", [0, 1, -1, 2.5, True])
def test_bootstrap_count_validation(n_boot):
    with pytest.raises(ValueError):
        bootstrap_ci([0, 1], n_boot=n_boot)


def test_bootstrap_shape_and_nonfinite_statistics():
    with pytest.raises(ValueError):
        bootstrap_ci([[1], [2]])
    with pytest.raises(ValueError):
        bootstrap_ci([0, np.nan])
    with pytest.raises(ValueError):
        beta_binomial_interval(1, 2, prior_a=0)
