# Signed coefficient sensitivity: retrospective protocol

This study asks whether allowing a negative calibration coefficient helps
when a fixed proxy is inversely associated with the outcome. It extends the
numeric mean API with an explicit opt-in coefficient range and evaluates
that range on the same cached LLMBar splits and on IID simulations with known
population means. The proxy, target and label budgets remain explicit.

The study is retrospective and exploratory. The earlier LLMBar analysis and
its weak or inverse proxy associations motivated this follow-up; the choice
of a signed range is not an independently validated hypothesis or a
preregistered analysis. The method is an application of established PPI++
mean estimation with a declared coefficient constraint, not a new estimator.
Versioned configurations record the analysis that was run; they do not
establish that no exploratory results informed its design.

## Estimator and coefficient range

For a fixed numeric proxy `F` and outcome `Y`, both in `[0,1]`, let `L` be
the audit pool and `U` the prediction-only pool. Retain

```text
theta(lambda) = mean(Y_L) + lambda * [mean(F_U) - mean(F_L)]
Q(lambda) = V_L(Y) - 2*lambda*C_L(Y,F)
            + lambda^2 * [V_L(F) + V_U(F)]
lambda_hat = project(C_L(Y,F) / [V_L(F)+V_U(F)], [lower, upper]).
```

Here `V` and `C` estimate the variance or covariance of the corresponding
sample means using the existing per-pool convention. For IID observations,
these are sample variances/covariances divided by their pool sizes. Grouped
data retain the existing centered cluster-sum sandwich calculation and
observation weighting; unequal cluster sizes do not imply equal cluster
weights. A zero denominator selects zero deterministically.

The numeric `prediction_powered_mean` API keeps `power_bounds=(0,1)` as its
default. The research option is `power_bounds=(-1,1)`. Allowed bounds must
be finite, ordered, contained in `[-1,1]` and contain zero. Fixed coefficients
must lie within the requested bounds; automatic tuning minimizes `Q`
continuously over that interval. Results disclose requested bounds, selected
coefficient and selection method. The categorical and finite-sample APIs
keep their existing behavior.

Both ranges contain zero, so their fitted quadratic variance cannot exceed
the audit-only value, apart from numerical tolerance. The signed range also
contains the positive range, so its fitted minimum cannot exceed the
positive-range minimum. These are algebraic statements about an estimated
variance criterion, not finite-sample guarantees about error, bias, interval
coverage or annotation savings. The point and normal interval are retained
without clipping to the outcome domain.

## Primary-source relationship and inversion

[PPI++ v2](https://arxiv.org/pdf/2311.01453v2), Section 2.2, discusses real
coefficients. Example 6.1 in Section 6.2 explicitly permits real coefficients
for mean estimation and gives the IID optimum
`Cov(Y,F) / ((1+n/N)*Var(F))`. Proposition 2 gives the variance-minimization
principle and Corollary 4 an asymptotic normality result under its assumptions.
This repository uses separate audit and prediction-pool variance estimates,
and a cluster-sandwich adaptation where requested. It does not claim that the
paper's IID theorem directly establishes grouped-data validity.

The authors' [ppi_py v0.2.3 mean implementation](https://raw.githubusercontent.com/aangelopoulos/ppi_py/v0.2.3/ppi_py/ppi.py)
uses clipped automatic tuning. The signed option follows the mean-estimation
principle in the paper, rather than claiming numerical identity with that
package or its pooled-variance convention. The existing pinned-author
[reference comparison](REFERENCE_BASELINE.md) remains a separate check of
the methods it actually compares.

For every fixed nonnegative `a`,

```text
theta(-a; F) = theta(a; 1-F).
```

Their residual-plus-prediction variances agree as well, including centered
cluster sums. Searching both proxy orientations with magnitudes in `[0,1]`
therefore duplicates the signed `[-1,1]` search, with a deterministic zero
tie. The study does not count inversion as an additional independent method.

Complementing the LLMBar proxy includes every invalid-judgment zero: those
scores become one under `1-F`. That complement is not literally a valid
disagreement indicator. It does not change canonical response IDs, reference
labels, parsing outcomes or the designated correctness target.

For a nonconstant binary proxy and bounded outcome, the IID population
regression slope is `E[Y|F=1]-E[Y|F=0]` and lies in `[-1,1]`. Thus the
population oracle coefficient has magnitude at most `N/(N+n)` under the
matched IID model. This observation motivates the range for this binary
proxy; it is not a universal bound for continuous rescaled proxies or
dependent clusters. In those settings the chosen range is a constraint.

## Selection, leakage and interval limits

All outcome-dependent sign and magnitude selection uses audit outcomes only.
Prediction-only proxy values can enter their own pool variance. Neither
prediction-only outcomes nor full-corpus outcome associations choose the
sign, coefficient, proxy orientation, best seed or reported method. The
historical full-corpus diagnostics motivated the study but are not tuning
inputs. Existing disjoint split manifests are reused exactly.

A fixed coefficient gives an unbiased correction under matched independent
IID pools. Estimating the coefficient on the same audit can produce finite
sample bias. Under suitable moments, a nondegenerate population denominator
and consistent coefficient estimation, the additional tuning error is
asymptotically negligible. Reported normal intervals use the plug-in variance;
they do not receive a new finite-sample coverage guarantee by admitting
negative coefficients. Small samples, constant observed outcomes and nearly
degenerate proxy variance remain relevant stress cases.

Choosing a fixed magnitude of one solely from the sign of noisy audit
correlation is not the proposed continuous quadratic minimization. Nor can
finite-sample concentration bounds for a fixed coefficient be reused after
unallocated adaptive orientation selection. This iteration leaves the
finite-sample API unchanged and makes no such finite-sample claim.

## Known-truth IID study

The simulation has 18 population/sample-size cells:

| Component | Values |
|---|---|
| Outcome prevalence `p` | `0.50`, `0.95` |
| Proxy sensitivity/specificity | positive `(.95,.90)`; inverse `(.05,.10)`; uninformative `(.60,.40)` |
| Audit observations `n` | `20`, `200`, `1000` |
| Prediction observations `N` | `10*n` |
| Replications per cell | `1000` |
| Seed | `2029` |

Independent audit and prediction pools are drawn from the same binary joint
law. All four methods share each replication's draws: `human_only`, fixed
coefficient `+1` (`ppi`), tuned `[0,1]` (`ppi_tuned`), and tuned `[-1,1]`
(`ppi_signed`). The target is the exact
population prevalence `p`, not the realized prediction-pool outcome mean.
No replication is discarded because of constant scores, unfavorable errors,
unusual coefficients or intervals extending outside `[0,1]`.

For `q = p*sensitivity + (1-p)*(1-specificity)`, the analytic covariance is
`p*(1-p)*(sensitivity+specificity-1)`. The IID oracle coefficient divides
that covariance by `(1+n/N)*q*(1-q)` and is retained only as a diagnostic;
it is not supplied to the fitted procedures. The inverse law complements the
positive law at each prevalence; the uninformative law has covariance zero.

Retain every point estimate, error, selected coefficient, estimated standard
error, interval and population-target coverage event. Summaries include bias,
RMSE, mean interval width, coverage and pointwise 95% exact binomial Monte Carlo
uncertainty, negative/zero/boundary coefficient frequencies and the frequency
of estimates outside `[0,1]`. These are repeated independent synthetic draws,
unlike the dependent repeated splits below. Pointwise intervals do not
establish a simultaneous statement across all methods and cells.

Method comparisons use shared-draw paired differences. For MSE, report
`mean(error_signed^2 - error_baseline^2)` and Monte Carlo standard error
`sd(error_signed^2 - error_baseline^2)/sqrt(R)`. Negative differences favor
signed tuning. The prespecified baselines are `human_only` and `ppi_tuned`.
RMSE remains descriptive unless an explicitly documented
paired calculation is provided, including its degenerate cases. Retain
small-sample undercoverage and no-signal costs rather than interpreting a
smaller fitted variance as proof of better inference.

## LLMBar constraint sensitivity

Reuse the exact certified source, canonicalization, invalid-row convention,
comparison weighting, cohorts, historical judge caches and source limitations
in the [LLMBar protocol](LLMBAR_PROTOCOL.md): 419 pairs, 418 exact instruction
groups, three judges, All/Natural/Adversarial, seeds 0 through 29, evaluation
fraction `.25`, and nested audit fractions `.20`, `.40`, `.60`. The source
revision is `900616bff90b6c6c8e1681f7d079250637c55992`; the certified label
snapshot SHA-256 is
`aafff984b410bd1f60e0a3196a10957c5a1f241caf8c85c4c391d8b1e2ef1f97`.

The outcome remains correctness of the forward canonical judge choice
against the supplied instruction-following reference. The gold-free proxy
remains valid canonical agreement across the two cached orders. Invalid
forward choices count as incorrect, and either invalid order makes the
proxy zero. No judge labels or reference targets are inverted.

Five methods are compared on identical pools and realized held-out targets:
`raw_proxy`, `human_only`, `ppi`, `ppi_tuned`, and `ppi_signed`.
The full study retains 4,050 rows
(`3 judges * 3 cohorts * 3 budgets * 30 seeds * 5 methods`). Existing-method
results and every split manifest must reproduce the prior study, with the
new method appended. Corrected methods use the same audit gold and cached
prediction inputs. Gold is shared across judges; per-judge label counts are
not distinct purchases to be summed. No new model API or annotation call is
made.

The general benchmark requires the explicit `include_signed=True` option;
its default retains the existing four-method output. The signed-enabled
tables disclose `power_lower` and `power_upper`: `[-1,1]` for signed tuning,
`[0,1]` for the original numeric methods and null for the raw proxy.

Report paired absolute- and squared-error differences within each identical
split, coefficient distributions and interval widths, retaining all cells
and losses. Compare each method to the same judge-specific held-out accuracy;
do not compare errors across different accuracy definitions. The overlapping
cohorts and repeated finite-corpus splits are dependent, so their averages
are descriptive. There is no IID Monte Carlo standard error, hypothesis-test
claim or interval coverage statistic for this benchmark.

The cluster-normal intervals retain the earlier hypothetical population-mean
interpretation. They are not confidence intervals for the realized held-out
accuracy used to score point error. Selected historical model versions and
ChatGPT-influenced adversarial construction limit generalization. A negative
coefficient is evidence about an audit's proxy association, not a claim that
disagreement universally identifies correct answers.

## Verification and reporting requirements

Verification includes hand-calculated signed quadratic minima, range and
boundary validation, exact inversion identities, constant proxies, and
positive/signed estimated-variance ordering. Grouped checks preserve unequal
cluster sizes and all repeated instructions. Hidden evaluation-gold changes
must leave fitted coefficients, estimates, intervals, costs and manifests
unchanged; unused rows must not affect fitting. The numerical API default,
categorical API and finite-sample API retain their existing contract.

Exports retain the exact configurations, method/range metadata, all trial
rows, paired comparisons, manifests and source/artifact fingerprints. Claims
must distinguish empirical error comparisons, algebraic variance minimization
and asymptotic inference. None entails guaranteed finite-sample dominance,
universal model ranking or measured annotation savings.

The [signed-power report](../reports/signed-power/REPORT.md) contains both
studies, all trial/summary tables, paired comparisons and the shared split
manifest. Run the frozen full configuration from the repository root:

```bash
python examples/signed_power_study.py --output reports/reproduced/signed-power --plot
```

The default full run uses 1,000 simulation replications and 30 empirical
split seeds. Smaller development runs must record their actual sizes and
must not be presented as the full results. Original certified LLMBar inputs
and prior report artifacts remain separate from the extension outputs.
