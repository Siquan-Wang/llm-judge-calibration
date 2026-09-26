# Population versus held-out mean: retrospective protocol

This study compares correction rules when the target is either a population
mean or the realized mean of a prediction-only pool. The existing population
API retains its target and coefficient criterion. Its documented target is correct; the question is
how a different target changes the preferred coefficient and uncertainty.
This is a retrospective application of regression and difference estimation,
not a new estimator or a repair of a wrongly specified existing API.

Earlier synthetic and LLMBar results motivated the analysis. The design is
not preregistered or independent external validation. Versioned configuration
records identify the experiment without implying that its choices preceded
all exploratory results. The synthetic experiment separates the targets;
the grouped empirical follow-up compares point errors only.

## Two targets and two variance criteria

Let independent IID pools L and U contain `n` and `N` draws from the same
joint law of a frozen proxy `F` and outcome `Y`, both in `[0,1]`. Define
`mu=E[Y]`, `R(lambda)=Y-lambda*F`, and the common point formula

```text
t(lambda) = mean(Y_L) + lambda * [mean(F_U)-mean(F_L)].
```

For a fixed coefficient the population error and held-out error differ:

| Target | Error | Error variance |
|---|---|---|
| Population mean `mu` | `Rbar_L-E[R] + lambda*(Fbar_U-E[F])` | `Var(R)/n + lambda^2*Var(F)/N` |
| Random held-out mean `Ybar_U` | `Rbar_L-Rbar_U` | `(1/n+1/N)*Var(R)` |

The point and held-out target share the prediction-pool proxy. Their
covariance matters: one cannot obtain the second variance merely by adding
the target variance to the first. For nonzero proxy variance, let
`b=Cov(Y,F)/Var(F)`. The fixed-coefficient oracles are

```text
lambda_population = N/(n+N) * b
lambda_heldout = b.
```

The study projects both onto the same prespecified `[-1,1]` bounds. With a
binary proxy and bounded outcome, `b` is the difference between the two
conditional outcome means and lies in that range. This bound is not universal
for continuous rescaled proxies or grouped designs. A perfect proxy is an
exact illustration: coefficient one recovers `Ybar_U`; the population-optimal
coefficient instead produces the combined-pool mean. There is no dominance
across these two different targets.

The population rule follows the mean-estimation principle in
[PPI++ v2, Example 6.1](https://arxiv.org/html/2311.01453v2), with this
repository's declared per-pool variance convention. The held-out expression
above follows by subtracting `Ybar_U` from the same point formula. This
elementary target-specific derivation is not an attribution of a different
target to the PPI++ population result.

## IID prediction API and its limits

The separate `predict_heldout_mean` API is IID-only. For automatic tuning it
projects the audit sample covariance divided by the audit proxy variance
onto the requested bounds, using a deterministic zero fallback when that
variance is zero. Neither prediction-pool outcomes nor analytic oracle values
enter fitting. Prediction-pool proxies enter the point estimate; their
outcomes are unavailable until the experiment's scoring step.

For a fixed coefficient, the variance estimate is

```text
v_hat_heldout(lambda) = (1/n+1/N) * s^2_L(Y-lambda*F),
```

where sample variance uses `ddof=1`. Under the stated IID sampling model this
is unbiased for the fixed-coefficient prediction-error variance. Same-audit
coefficient tuning is not exactly unbiased in general. A plug-in normal
prediction interval requires consistent tuning and residual variance
estimation, nondegenerate limiting error variance, and an appropriate joint
large-sample regime for both pools. Constant observed outcomes or residuals
can produce zero-width intervals without demonstrating certainty.

The result is explicitly an **asymptotic marginal prediction interval for
the random held-out mean**. Its probability statement averages over draws
of both L and U. It is not a confidence interval for `mu`, a finite-sample
guarantee, or a guarantee conditional on every realized U, every set of
observed proxy values, or one frozen audit. Knowing `F_U` does not reveal
the target pool's residual conditional means or variances. Covariate or
residual shift between the pools can invalidate audit-moment transport.

The API does not accept group identifiers or silently use an IID formula for
grouped data. The existing population, categorical and finite-sample APIs
retain their prior contracts. Estimates and intervals are not clipped to
`[0,1]`; all draws and degenerate fits remain in the outputs.

## Fixed finite populations require a separate sampling description

The following fixed-coefficient identities clarify scope; they do not add
a design-based guarantee to the IID API. Fix M outcome/proxy pairs, let
`S_R^2` have denominator `M-1`, and uniformly assign disjoint samples of
sizes n and N, leaving any remaining rows unused. Then

```text
Var_design(Rbar_L) = (1/n-1/M)*S_R^2
Var_design(Rbar_U) = (1/N-1/M)*S_R^2
Cov_design(Rbar_L,Rbar_U) = -S_R^2/M
Var_design(Rbar_L-Rbar_U) = (1/n+1/N)*S_R^2.
```

The finite-population corrections cancel with the negative covariance. This
is joint randomization over both pools and their changing held-out target,
not independence between the samples. Conditional on an already frozen U,
an audit sampled from its complement need not have the same residual mean
as U, so the joint-design statement does not give conditional coverage.

If instead the target is the full fixed-population mean and the audit is a
simple random sample from it, the generalized difference estimator uses all
population proxies:

```text
t_full = lambda*Fbar_M + Rbar_L
Var_design(t_full-Ybar_M) = (1/n-1/M)*S_R^2.
```

This is the setting with an actual finite-population correction. The
[original PPI supplement](https://www.stat.ubc.ca/~john/papers/AngelopoulosScience2023.pdf),
section *Inference on a Finite Population*, Theorem S5, Corollary S14 and
Propositions S3/S4, explicitly treats audits sampled from a full finite
target population. Its fixed-function coverage cannot automatically be
transferred to a residual function selected on the same audit. Finite-grid
simultaneous bounds or a separately accounted pilot would be additional
methods, outside this iteration.

An external audit for a separately frozen target panel requires a transport
model or a probability-sampling relationship. Calling that panel finite
does not supply either assumption. Independently redrawing whole partitions
can support conditional-design Monte Carlo uncertainty, even when realized
rows overlap, but that is a separately declared randomization experiment;
it does not turn reused benchmark rows into independent deployment evidence.

## Known-truth target sensitivity experiment

The full configuration has 36 IID cells and 1,000 independent replications
per cell, using master seed 2030:

| Component | Values |
|---|---|
| Outcome prevalence p | `.50`, `.95` |
| Proxy sensitivity/specificity | positive `(.95,.90)`; inverse `(.05,.10)`; uninformative `(.60,.40)` |
| Audit size n | `20`, `200` |
| Prediction/audit size ratio N/n | `.5`, `1`, `10` |
| Coefficient bounds for both tuned rules | `[-1,1]` |

Four point rules share every draw: `human_only`, fixed coefficient `+1`
(`fixed_ppi`), existing signed population tuning (`population_tuned`), and
signed audit-residual tuning for the held-out target (`pool_tuned`).
Analytic population and held-out oracles are diagnostic
metadata only. The two targets are the exact prevalence p and that draw's
realized `mean(Y_U)`. Prediction-pool outcomes are retained by the scorer
but never supplied to the fitting procedures.

`run_estimand_study(repetitions=1000, seed=2030, alpha=.05)` returns trial and
summary tables, with full configuration in trial attributes. The complete
run retains 144,000 rows: one point per rule, cell and replication, scored
against both targets.

For each rule retain the same point estimate scored against both targets.
Store target-specific interval variances: population variance uses the
existing residual-plus-prediction expression, while held-out prediction
variance uses the audit residual variance multiplied by `1/n+1/N`.
Population confidence coverage and random-pool prediction coverage are
separate metrics with separate labels. A deliberately cross-target interval
diagnostic, if included, must be marked as a mismatch; it must not be called
a failure of an API's correctly stated target.

Coverage scoring uses a floating-point margin of
`8 * float64_epsilon * max(1, abs(lower), abs(upper), abs(truth))`
at either endpoint, for every method and both targets. This is a numerical
evaluation convention, added after independent review found roundoff-only
misses for algebraically exact inverse-proxy predictions. It does not alter
the point estimate, interval endpoints, width, variance or loss. Exact
round-trip float parsing is required when independently recomputing coverage
from trial CSVs. The margin is not an inferential correction or a coverage
guarantee.

Report bias, RMSE, interval width, matched coverage, pointwise Monte Carlo
uncertainty, selected coefficient and out-of-domain or zero-width rates.
Compare rules with paired squared-error differences within the same target
and draw, against the `population_tuned` and `pool_tuned` baselines, preserving
pairing in Monte Carlo standard errors. Exact pointwise 95% binomial intervals
quantify coverage Monte Carlo uncertainty, separately from inference alpha.
Do not compare
the numerical RMSE of unlike targets as a superiority claim. Retain all
cells, adverse outcomes and small-sample undercoverage; no result-dependent
method or cell exclusion is permitted.

The design includes audit-heavy ratios because the coefficient distinction
shrinks when `N` is much larger than `n`. Required mathematical checks cover
perfect and inverse proxies, zero covariance, constant proxies, exact oracle
and excess-risk identities, and no hidden-outcome access. A small exhaustive
fixed-population partition fixture verifies the covariance cancellation for
fixed coefficients, separately from any simulation of adaptive tuning.

## Grouped LLMBar: a point-only comparison

Use the certified source and construction in the
[LLMBar protocol](LLMBAR_PROTOCOL.md): 419 comparisons, 418 exact instruction
groups, three historical judges, retained invalid predictions, forward
correctness as outcome and canonical two-order agreement as the gold-free
proxy. The source revision is
`900616bff90b6c6c8e1681f7d079250637c55992`; label SHA-256 is
`aafff984b410bd1f60e0a3196a10957c5a1f241caf8c85c4c391d8b1e2ef1f97`.

Preserve all 270 previous instruction manifests: All/Natural/Adversarial,
30 seeds, fixed 25% evaluation groups and nested 20/40/60% audits. Append
one `audit_residual` method, using the separate `audit_residual_mean` point-only
procedure, to the preceding five methods,
giving 4,860 trial rows and 162 method/cohort/judge/budget summaries. The
additional rule minimizes audit cluster residual variance with the matching
audit cluster covariance/proxy variance, signed bounds and zero fallback.
It uses the same audit and prediction inputs as the preceding corrected
methods. Label cost is shared across judges, and no new API calls or human
annotations are purchased.

The empirical extension is explicit: `include_pool_tuned=True` requires
`include_signed=True`. Without that extension, prior method rows and table
schemas retain their existing contract. The synthetic table separates
`population_error` from `pool_error`, `population_variance` from
`pool_prediction_variance`, and each matched interval's width and coverage.

The target remains the identical realized forward accuracy of each held-out
pool. Compare paired point errors and coefficient shifts, preserving all
worsened cells. Existing-method results and manifests must remain compatible
with the preceding study. Full-corpus gold associations, evaluation outcomes
and unused rows cannot select the new rule's coefficient.

This grouped procedure is an audit-residual variance criterion, **not a
claim of exact conditional finite-corpus optimality**. Equal row weights
with unequal instruction-group sizes make means ratios of random totals.
The row-SRS formulas do not follow by substituting row counts into a cluster
formula. A cluster superpopulation linearization would involve centered
outcome and proxy totals and the common size distribution; a finite-cluster
design analysis would additionally require its actual inclusion probabilities
and covariance structure. Neither is supplied merely by retaining group IDs.

Accordingly the new empirical procedure has no standard error, confidence
interval or prediction interval. Existing population interval diagnostics
keep their original interpretation. The report makes no empirical pool
coverage claim. Overlapping cohorts and reused finite-corpus splits remain
descriptive benchmark evidence. Historical versions, curated references,
ChatGPT-influenced adversarial selection and the retrospective motivation
limit generalization.

## Reproduction and acceptance

The [estimand report](../reports/estimand/REPORT.md) retains both target
definitions, complete configurations, all trial rows and paired contrasts,
unchanged empirical manifests, and source/artifact fingerprints. The full
runner is:

```bash
python examples/estimand_study.py --output reports/reproduced/estimand --plot
```

Smaller development runs must record their actual replication/seed counts.
Verification includes analytical target-specific variances, invariance and
limiting cases, strict IID-only prediction boundaries, hidden-evaluation-gold
checks, grouped point-only null uncertainty fields, and reproduction of the
prior empirical methods/manifests. Any unresolved conditional or grouped
inference question remains explicit rather than being filled by a normal
interval without a justified probability model.
