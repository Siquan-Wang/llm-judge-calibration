# Finite-sample intervals: retrospective study protocol

This study asks how much interval width is paid for an explicit finite-sample
coverage guarantee when an imperfect frozen judge supplements a human audit.
It compares normal intervals with bounded-data concentration intervals on
identical synthetic observations. The guarantees concern the population mean
of the defined observed outcome, not latent human consensus or judge accuracy.

This is a versioned retrospective extension of the earlier simulations. Prior
results and development checks informed the design; it is not preregistered.
All configured cells, methods and repetitions are retained. The mathematical
guarantees follow established concentration results and PPI reasoning; neither
this composition nor the simulation is claimed as a new estimator or theorem.

The [finite-sample report and artifacts](../reports/finite-sample/REPORT.md)
record the resulting run. This protocol specifies the analysis and scope;
the report supplies observed results.

## Target, sampling assumptions and notation

Let the human outcome `Y` and frozen prediction `F` take values in `[0,1]`.
The target is `mu = E[Y]`. The labeled pool contains `n` iid pairs `(Y_i,F_i)`;
the separate prediction pool contains `N` iid predictions with the same
prediction marginal. The pools are independent. Dependence between `Y_i` and
`F_i` within a pair is allowed and useful. Sample sizes are fixed in advance.
The judge is fixed independently of both pools, or inference is conditional
on an independently trained frozen judge. Fitting the judge to these audit
labels is outside this protocol.

For a prespecified power `lambda` in `[0,1]`, define

```text
R_lambda = Y - lambda * F                    in [-lambda, 1]
estimate_lambda = mean(R_lambda,L) + lambda * mean(F_U)
E[estimate_lambda] = mu.
```

The unbiasedness statement is for fixed power. A power selected using the
same observations need not produce an unbiased point estimate. Its coverage
can nevertheless be protected by the simultaneous interval construction
below. Each score is an equally weighted observation; no clustering,
inverse-probability weighting or annotator model is included here.

The PPI paper's common-population and independent-predictor setup is in
Section 1.2. Its Theorem 1 composes confidence sets for a correction and a
prediction term; Algorithm 10 and Corollary C.1 already give nonasymptotic
mean inference. The particular concentration bounds used here are specified
separately; they are not a reproduction of its betting-based `MeanCI`
Algorithm 13. [Angelopoulos, Bates, Fannjiang, Jordan and Zrnic,
Prediction-Powered Inference](https://arxiv.org/pdf/2301.09633).

## Fixed-power weighted Hoeffding interval

Apply Hoeffding's inequality directly to the independent summands
`R_lambda,i / n` and `lambda * F_U,j / N`. Their squared range lengths sum to

```text
W_lambda = (1 + lambda)^2 / n + lambda^2 / N.
radius_H = sqrt(log(2 / alpha) * W_lambda / 2).
interval_H = [estimate_lambda - radius_H, estimate_lambda + radius_H].
```

The two-sided failure probability is at most `alpha`. Pairing is preserved:
`Y_i` and `F_i` are not incorrectly treated as independent summands.
At `lambda=0`, the formula reduces to the ordinary human-mean Hoeffding
radius. The underlying unequal-range inequality is Theorem 2 of
[Hoeffding (1963), Probability Inequalities for Sums of Bounded Random
Variables](https://doi.org/10.1080/01621459.1963.10500830).

This bound uses the known range, not the observed residual range. A perfect
judge in one sample does not justify declaring a smaller population range.
The radius increases with nonnegative power under these worst-case ranges;
it does not automatically reward predictive accuracy. The study reports
fixed powers zero and one. The API also supports a prespecified tuple of
`K` powers: replace `log(2/alpha)` by `log(2*K/alpha)` and take a union bound
over candidates. Its minimum-radius selection always chooses the smallest
power. Choosing among separately calibrated scalar intervals after seeing
data is not protected by that tuple-specific adjustment.

## Empirical Bernstein interval and finite-grid selection

For iid observations `Z_1,...,Z_m` in a known interval of length `L`, let
`s_Z^2 = sum((Z_i - mean(Z))^2)/(m-1)`, with `m >= 2`. The two-sided radius is

```text
B(Z, delta, L) = sqrt(2 * s_Z^2 * log(4 / delta) / m)
                + 7 * L * log(4 / delta) / (3 * (m - 1)).
```

This follows from Maurer and Pontil's Theorem 4: normalize to `[0,1]`, apply
the one-sided theorem to that variable and its complement at error
`delta/2` each, then rescale. Their pairwise sample-variance definition equals
the usual `ddof=1` variance. This derivation uses Theorem 4 directly and an
explicit union bound. [Maurer and Pontil (2009), Empirical Bernstein Bounds
and Sample Variance Penalization, p. 2](https://www.learningtheory.org/colt2009/papers/012.pdf#page=2).

For a prespecified finite grid `Lambda` with `K` distinct powers, allocate
failure probability as follows:

```text
each residual event:   delta_R = alpha / (2*K)
one prediction event:  delta_F = alpha / 2

radius_EB(lambda) = B(residuals_lambda, delta_R, 1+lambda)
                   + lambda * B(F_U, delta_F, 1)
```

Here `residuals_lambda` is the vector of labeled residuals; the last argument
of `B` is its known range length. The residual logarithm is `log(8*K/alpha)`
and the prediction logarithm is `log(8/alpha)`. The prediction event is
shared across all powers and is counted only once. No independence among
the events for different powers is assumed or needed.

With probability at least `1-alpha`, every residual mean deviation and the
shared prediction mean deviation satisfy their bounds. The triangle
inequality then places `mu` inside every grid interval simultaneously.
Therefore selecting any one of these grid intervals from the same data
preserves coverage. The implemented rule selects the smallest untruncated
radius, breaking exact ties toward the smaller power after sorting the grid.

Fixed positive-power Bernstein inference is the `K=1` case. A standalone
zero-power call uses the sharper human-only `B(Y_L, alpha, 1)` without an
unused prediction allocation. Zero inside a nontrivial grid retains the
grid's residual allocation. Thus the selected grid interval is no wider
than the candidates under that common allocation; this is not a guarantee
of beating the standalone human-only interval. Small samples can give very
wide intervals even with excellent predictions.

The grid is fixed before the observations used in an interval. This result
does not validate continuous `power="auto"`, a grid constructed from those
observations, fitting the judge to audit labels, optional stopping, or
post-hoc selection between separately calibrated Hoeffding and Bernstein
intervals. The implementation keeps intervals untruncated so that the width
cost remains visible. Intersecting an individual valid interval with the
known parameter space `[0,1]` would preserve its coverage, but those clipped
widths are not the reported metric.

## Experiment matrix

The generator draws `Y ~ Bernoulli(p)` and then draws a binary judge with
`P(F=1 | Y=1)=sensitivity` and `P(F=0 | Y=0)=specificity`. Its judge-positive
rate is `p*sensitivity + (1-p)*(1-specificity)`. The human population truth
is exactly `p`; a realized sample average is not substituted for it.

Twelve iid cells cross these four profiles with audit sizes
`n in {20,200,1000}` and prediction size `N=10*n`:

| Profile | Human prevalence | Sensitivity | Specificity |
|---|---:|---:|---:|
| Balanced, strong judge | 0.50 | 0.95 | 0.90 |
| Imbalanced, strong judge (`rare_strong`) | 0.95 | 0.95 | 0.90 |
| Balanced, uninformative judge | 0.50 | 0.60 | 0.40 |
| Balanced, perfect judge | 0.50 | 1.00 | 1.00 |

The imbalanced profile has rare negative outcomes. In the uninformative
profile, the probability of a positive prediction is 0.60 for either human
outcome, so the judge is independent of the outcome. All observations in
the two pools are generated independently in these twelve cells.

Two additional cells deliberately leave the target guarantee's scope:

| Cell | Construction | Violated assumption |
|---|---|---|
| Repeated observations treated as iid | Balanced strong judge; 50 audit and 500 target independent units, each repeated four times, yielding 200 and 2,000 rows | Rows within a repeated unit are dependent; all methods intentionally receive an iid analysis |
| Prevalence shift | Audit `p=0.40`, target `p=0.75`, fixed sensitivity 0.90 and specificity 0.80; `n=1000`, `N=10000` | Audit and target do not share the human/judge distribution |

These are tests of misuse, not evidence of cluster or shift validity.
Conservative intervals can still cover in a violating cell by chance or
because they are wide. The shift cell has judge-positive rates 0.48 and
0.725. For fixed power, the estimator's expectation is
`0.40 + 0.245*lambda`, rather than target human prevalence 0.75.
The human-only interval still has its native audit target 0.40; its native
finite-sample guarantee must not be mistaken for target-population coverage.
For a fixed positive power, the concentration bounds can still cover the
shifted rectified functional `E_L[Y] - lambda*E_L[F] + lambda*E_U[F]`.
That functional is not the target human mean. To avoid comparing different
native functionals as if they were the same target, this study leaves
nonhuman methods' shifted native truth, coverage and native-theorem fields
unset, with reason `not_reported_for_shifted_rectified_functional`.

The default run uses 1,000 independent repetitions per cell, seed 2028,
`alpha=0.05`, and the grid `(0,0.25,0.5,0.75,1)`. These choices, including
smaller smoke-run repetition counts, are exported in the run configuration.
Every method in a repetition sees the same audit and prediction data:

| Method ID | Point / interval rule |
|---|---|
| `human_normal` | Human audit mean with normal interval |
| `ppi_normal` | Fixed power one with normal interval |
| `ppi_tuned_normal` | Continuous variance-tuned power with normal interval |
| `human_hoeffding` | Human audit mean with Hoeffding interval |
| `ppi_hoeffding` | Fixed power one with weighted Hoeffding interval |
| `human_eb` | Standalone human mean with empirical Bernstein interval |
| `ppi_eb` | Fixed power one with empirical Bernstein interval |
| `ppi_eb_grid` | Prespecified finite-grid empirical Bernstein selection |

All eight methods consume the same number of human labels. The normal
methods are existing asymptotic comparators and do not acquire finite-sample
validity through this comparison. A `theorem_applicable` field refers to
the finite-sample construction for the intended target, not a blanket claim
that a normal method has no asymptotic justification. Native-target scope
is recorded separately for the human-only shift case.

## Reporting and interpretation

Retain every repetition's point estimate, interval endpoints, width, selected
power, human-label budget, exact target, coverage and assumption-scope flags.
Report coverage with its binomial Monte Carlo standard error, mean width,
point-estimation bias and RMSE, and power selection. Compare widths using
paired differences on the same generated data and the appropriate human-only
family baseline. Monte Carlo intervals quantify simulation precision, not
uncertainty about the mathematical theorem or robustness outside assumptions.
Plug-in coverage MCSE is zero when all draws cover or all miss; that does not
establish certainty. Also report pointwise 95% exact binomial intervals
(`coverage_mc_low`, `coverage_mc_high`), including nonzero uncertainty at
1,000/1,000 covered draws. These are not simultaneous intervals across all
112 cell/method combinations. Do not drop wide intervals or degenerate
normal results.

Show the twelve valid iid cells separately from the two assumption violations.
Distinguish theoretical coverage from observed simulation coverage. A numerical
study cannot prove a distribution-wide guarantee, and an observed value near
95% does not confer validity on a procedure outside its assumptions. Conversely,
the finite-sample guarantee is a lower bound; it does not promise coverage
exactly equal to 95%, useful width, unbiased selected points, or smaller RMSE.

This study uses synthetic data only. It makes no finite-sample coverage claim
for the dependent MT-Bench comparison rows, no claim about latent correctness,
no annotation-saving guarantee and no empirical conclusion about newer judge
families. The evidence is the stated derivation plus reproducible simulations
with all configured cases retained.

## Reproduce and audit

```bash
pip install -e ".[dev,report]"
python examples/finite_sample_study.py --output reports/finite-sample --plot
```

The runner exports deterministic compressed trial CSV, all 112 default
cell/method summaries, exact scenario configurations, package versions,
normalized source hashes and artifact hashes. The report table includes all
eight methods; the main iid plot may show six explicitly named methods for
readability. Assumption-violation rows must not enter those iid curves, even
when they share the same generator profile. The width axis is logarithmic;
width above one and containment of the entire `[0,1]` domain are different
diagnostics and are retained separately.
