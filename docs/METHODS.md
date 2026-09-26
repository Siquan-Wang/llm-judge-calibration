# Methods and interpretation

`judgecal` estimates bounded outcome means using a frozen prediction or
proxy and a smaller labeled audit. Outcomes include human-preference
scores and a judge's correctness against benchmark references. Its mean
inference APIs with two independent pools target population expectations;
the separate held-out-mean prediction API and fixed finite-corpus sampling
design are described below. The package implements established mean
correction and power-tuning ideas, with an explicit cluster-variance
adaptation. It does not introduce a new PPI estimator or establish general
annotation savings.

## Target and sampling units

For the categorical human-preference API and a consistently oriented
comparison, let the human score `Y` and judge
score `F` equal 1 for a target-model win, 0 for a loss, and 0.5 for a tie.
The population target is `theta = E[Y]` under that convention. This is a
human-preference target; it is not automatically factual correctness, an
individual annotator's preference, or latent consensus.

`prediction_powered_mean` extends the same mean-estimation algebra to
finite numeric predictions and outcomes in `[0,1]`, without rounding them
to categories. Its `PPIMeanResult` names the raw prediction-pool mean
`prediction_mean` and the audit outcome mean `outcome_mean`; it does not
assign an A/B target. For example, an outcome can be the mean recorded
vote score within one comparison. Averaging those rows weights comparisons
equally, not individual votes. Changing from a plurality outcome to an
average vote score changes `Y` and therefore the estimand. Fractional
scores are not automatically calibrated probabilities or latent truth.
See the [human-reference sensitivity protocol](REFERENCE_SENSITIVITY.md).

In the [LLMBar](LLMBAR_PROTOCOL.md) and
[RewardBench](REWARDBENCH_PROTOCOL.md) audits, `Y` instead denotes a
designated judge's correctness against the supplied benchmark reference,
and `F` is an observable choice-agreement proxy. These references are not
uniformly human-preference annotations, and agreement is not itself a
calibrated correctness probability. The chosen outcome defines the target.

The generic and categorical APIs share the same inference core. Both
require aligned audit predictions/outcomes, at least two observations per
pool, and (when supplied) at least two disjoint groups per pool. These are
computational minima, not claims of adequate sample size. Estimates and
normal intervals can extend beyond the bounded outcome range.

The labeled audit contains aligned `(Y_L, F_L)` values, and a separate
prediction-only pool contains `F_U`. Outcomes in the prediction-only
pool are unavailable to the estimator. The pools must be independent and
represent the same relevant population, and both use the same frozen
prediction or proxy definition.
For a fixed power, the correction is unbiased when the audit outcome
sample mean is unbiased for the target and the two proxy sample means have
the same expectation, as under iid sampling from the same population.
Random unequal cluster sizes can introduce finite-sample ratio bias in
row-weighted means; their asymptotic population interpretation is described
below. The documented same-population condition is a convenient sampling
assumption. Changing prompts, model versions, or audit selection can
violate it.

Rows are independent in iid mode. In grouped mode, groups are independent
and dependence within a group is allowed. Group IDs must be supplied for
both pools and must not overlap. Turns of one question belong to the same
group. Grouping does not handle arbitrary crossed question/model/annotator
dependence, unequal sampling probabilities, or selection bias.

The point estimate averages observations, including when groups have
unequal sizes. It is not an equally weighted question mean. With iid random
clusters, its population interpretation is the ratio of expected total
score per cluster to expected cluster size; regularity conditions on sizes
and independent cluster sampling still matter.

## Mean correction and power tuning

For a fixed coefficient `lambda`, the estimate is

```text
theta_hat(lambda) = mean(Y_L) + lambda * [mean(F_U) - mean(F_L)]
                  = lambda * mean(F_U) + mean(Y_L - lambda * F_L).
```

The default `power=1` is the original mean correction; `power=0` returns
the labeled audit mean. The PPI framework is due to Angelopoulos,
Bates, Fannjiang, Jordan, and Zrnic; efficient power tuning is developed by
Angelopoulos, Duchi, and Zrnic in PPI++.
[PPI, 2023, v4](https://arxiv.org/abs/2301.09633v4),
[PPI++, 2023/2024, v2](https://arxiv.org/abs/2311.01453v2).

Write `V_L(X)` and `V_U(X)` for estimated variances of sample means, and
`C_L(Y,F)` for the paired covariance of the audit means. With the default
coefficient range `[0,1]`, this implementation minimizes

```text
Q(lambda) = lambda^2 * V_U(F) + V_L(Y - lambda * F)
          = V_L(Y) - 2 * lambda * C_L(Y,F)
            + lambda^2 * [V_L(F) + V_U(F)].

lambda_hat = projection_to_[0,1](C_L(Y,F) / [V_L(F) + V_U(F)]).
```

If the denominator is zero, the implementation selects zero. In the default
range, an empirically nonpositive covariance selects the human-only endpoint;
an unconstrained minimum above one selects one. The categorical win-rate API
retains this range. The point estimate and interval endpoints themselves are
never clipped to `[0,1]`.

Moment calculations recognize exactly constant input arrays before centering,
so summation roundoff cannot create spurious variance or covariance for a
repeated decimal value. This uses exact equality, not a near-zero tolerance:
representable variation, however small, is retained.

The numeric mean API also accepts explicit `power_bounds=(-1,1)`, projecting
the same ratio onto that signed interval. Allowed bounds are contained in
`[-1,1]`, contain zero, and must be chosen before examining evaluation outcomes.
This option can exploit negative audit covariance. It follows the mean case
in PPI++ Example 6.1, while retaining this package's per-pool variance convention.
See the [signed coefficient protocol](SIGNED_POWER_PROTOCOL.md) for the
proxy-complement identity, binary-oracle derivation, empirical study and
small-sample limitations. The finite-sample API is unchanged.

Because zero is feasible, `Q(lambda_hat) <= Q(0)` up to floating-point
roundoff. `estimated_variance_ratio` reports `Q(lambda_hat)/V_L(Y)` when
the denominator is positive, and is undefined otherwise. This algebraic
property concerns a fitted variance estimate, not realized squared error,
sampling variance at a finite sample size, or confidence-interval coverage.

Useful limiting cases are:

- A judge with zero audit covariance receives zero power, even if its
  marginal prediction mean happens to be close to the human rate.
- With a perfect judge and common nonzero variance, the large-sample iid
  optimum is `lambda = N/(n+N)`, combining the audit and prediction-only
  human-equivalent information. It need not be one.
- A large prediction-only pool makes `V_U(F)` small; it does not remove
  uncertainty in the audit correction or protect against shift.
- An audit with no observed human variation can yield a zero-width
  human-only/tuned interval. That sample does not establish certainty.

## Variance and covariance scaling

For `n` iid audit observations, the package uses sample moments with
`ddof=1`:

```text
V_L(X)   = sum_i (X_i - mean(X))^2 / [n * (n - 1)]
C_L(Y,F) = sum_i (Y_i - mean(Y)) * (F_i - mean(F)) / [n * (n - 1)].
```

For `G` groups containing `n` observations, define centered cluster sums
`S_g(X) = sum_{i in g}(X_i - mean(X))`. The one-way sandwich forms are

```text
V_L(X)   = G/(G-1) * sum_g S_g(X)^2 / n^2
C_L(Y,F) = G/(G-1) * sum_g S_g(Y) * S_g(F) / n^2.
```

The prediction-only variance uses its own observation and group counts.
Using the same scaling for covariance and variance makes the quadratic
minimization exact for the chosen sandwich estimate. Equal-sized exact
repeats within clusters produce the same variance as their independent
cluster-level values. Counting those repeated rows as independent would
understate uncertainty.

The cluster adaptation requires a suitable cluster central limit theorem,
consistent cluster moment estimates, sufficiently many independent groups,
and no dominating group. The iid PPI++ theorem alone does not prove these
conditions for every grouped dataset. Two groups pass input validation but
do not make a normal approximation reliable.

## Relationship to the authors' implementation

The iid coefficient can also be written as

```text
lambda_hat = projection_to_[0,1](
    s_YF,L / [s_F,L^2 + (n/N) * s_F,U^2]
).
```

The paper's common-population limit replaces the two judge variances by
the same population variance. The inspected scalar-mean implementation
in `ppi-python==0.2.3` uses pooled judge variance in its automatic
coefficient, with its own finite-sample covariance conventions. Its normal-interval variance
also uses different finite-sample moment divisors. Consequently numerical
equality at finite sample sizes is not expected. This package uses separate
pool variances so its selected coefficient minimizes exactly the variance
formula it reports; the cluster version applies the same principle to
cluster moments. This is an implementation choice, not a claimed statistical
innovation or a claim of superiority to `ppi_py`.
[Authors' versioned code](https://github.com/aangelopoulos/ppi_py/blob/v0.2.3/ppi_py/ppi.py),
[release 0.2.3](https://pypi.org/project/ppi-python/0.2.3/).
The [reference comparison](REFERENCE_BASELINE.md) records the inspected
wheel/source hashes and same-array numerical checks; it is not a coverage
or superiority experiment.

## What same-audit tuning does and does not justify

`power='auto'` uses the audit labels and both proxy pools, without accessing
evaluation outcomes. It estimates one coefficient; it does not fit or
fine-tune the underlying judge or proxy. To see why same-audit scalar tuning
can be first-order valid, compare a consistent coefficient with its deterministic
limit:

```text
theta_hat(lambda_hat) - theta_hat(lambda_star)
    = (lambda_hat - lambda_star) * [mean(F_U) - mean(F_L)].
```

Under the same-population and regular moment/CLT conditions, the first
factor tends to zero and the second has sampling-error scale. Their product
is smaller order than the leading estimation error; independence between
the fitted coefficient and audit is not needed for this argument. Appropriate
rate/cluster regularity and nondegenerate limiting variance remain necessary.

The interval is the normal approximation
`theta_hat +/- z_(1-alpha/2) * sqrt(Q(lambda_hat))`. It is asymptotic, not
finite-sample or distribution-free. The fitted coefficient can create
finite-sample bias and overoptimistic uncertainty, particularly with few
clusters or nearly degenerate data. The package does not add a finite-sample
correction for coefficient selection. Repeatedly selecting judge prompts,
models, methods, or datasets on the same audit is a separate selection
problem, outside this justification.

## Real-label benchmark: a fixed reference across audit budgets

`fixed_evaluation_label_budget` reserves 25% of each eligible model pair's
questions for evaluation within a seed. The remaining questions form an
ordered audit bank. Audit fractions 20%, 40%, and 60% refer to that pair's
total question count, rounded down; audits are nested, while evaluation
questions and their human reference stay fixed across these budgets.
Unused questions supply neither labels nor predictions. All turns remain
together. Different model pairs are analyzed separately and can assign a
question differently; no cross-pair fitting is performed.

Methods are scored against the realized mean of the aggregated human labels
on those held-out questions. This target is a fixed dataset reference for
point-error comparison. It is different from the population mean targeted
by the API's normal intervals, and those intervals are not evaluated as
coverage intervals for that realized held-out mean. In particular, power
tuning minimizes estimated population-mean variance, not error relative to
the observed held-out reference.

Conditionally on the finite corpus, audit and evaluation samples are
partitions drawn without replacement, not fresh independent population
samples. Seeds repeatedly reuse the same questions; model pairs also share
questions. Equal-weight pair/seed averages describe this dataset and cannot
be treated as independent replications for population significance claims.

The prediction-only pool is intentionally fixed and relatively small;
this is not a simulation of arbitrarily abundant unlabeled deployment data.
The benchmark records questions, aggregated comparisons, and available vote
counts. These counts are not annotation time, dollars, or demonstrated
label savings. Raw judging consumes no human audit labels. Raw intervals are
omitted here because uncertainty in the judge mean is not uncertainty about
its discrepancy from the human reference. The earlier complementary-pool
benchmark remains available separately.

## Predicting a random held-out mean

The separate IID-only `predict_heldout_mean` API targets the realized mean
of a random prediction pool. Its result explicitly names a
`prediction_interval` and `prediction_standard_error`, rather than reusing
population-confidence-interval fields. For fixed coefficient `lambda`,
error against this target equals `mean(R_L)-mean(R_U)`, where `R=Y-lambda*F`.
Its marginal variance is `(1/n+1/N)*Var(R)` under independent IID sampling.
The audit-only fitted slope minimizes residual variance; it lacks the
population coefficient's `N/(n+N)` shrinkage. Both constructions remain
asymptotic when a coefficient is estimated from the same audit.

This prediction interval averages over draws of both pools. It is not a
conditional guarantee for every frozen U or observed proxy composition.
The grouped `audit_residual_mean` candidate returns only a point estimate
and tuning diagnostics. It provides no grouped prediction interval. The
[estimand protocol](ESTIMAND_PROTOCOL.md) derives the target distinction,
separates finite-population sampling designs, and defines the paired study.
Finite-population CLTs and random-partition arguments have established
sampling-design assumptions; they do not give the grouped audit-selected
rule a conditional interval for an arbitrary fixed target pool.
[Li and Ding, 2017; inspected preprint v1](https://arxiv.org/abs/1610.04821v1).

## Synthetic study: exact targets and failure cases

The scenario configuration is versioned and retrospective. Choices reflect
development and exploratory checks; this is not a preregistered experiment.
The study generates independent audit and target pools with analytically
specified binary human prevalences and conditional judge sensitivity and
specificity. Its target truth is a population expectation, not a noisy
realized test-set mean.

The suite includes strong, weak, and anticorrelated iid judges, independent
clusters of repeated turns, and a few-cluster stress test. Cluster sizes are
equal and fixed, with exact within-cluster repeated labels. This is a clear
dependence stress case, not an empirical model of all conversational data.
The suite does not currently simulate ties, unequal sizes, multiway
dependence, multiple annotators, or fitted judges. Ties and unequal cluster
algebra have separate unit checks, which do not establish their coverage.

Two additional scenarios deliberately violate transfer assumptions:

- **Prevalence shift:** the audit and target human rates differ while
  conditional judge sensitivity and specificity remain fixed.
- **Conditional judge-error shift:** human prevalence stays fixed while
  judge sensitivity and specificity change between pools.

Both test failure under specified mechanisms; they do not validate a
shift-robust estimator. A fixed-power estimate has expectation
`p_L + lambda * (q_U - q_L)`, where `p_L` is audit human prevalence and
`q_L,q_U` are judge-positive rates. It need not equal target human prevalence
`p_U` under either shift. Variance minimization cannot remove this bias.

Every draw is retained, including poor results and zero-width intervals.
The main error and `covered` fields always use target human prevalence.
Native interval targets are recorded separately:

| Method | Native interval target | Shift interpretation |
|---|---|---|
| Raw judge | Target judge-positive rate | Native coverage says nothing about human bias. |
| Human-only | Audit human-preference rate | Native coverage can be reasonable while target coverage fails under prevalence shift. |
| PPI / tuned PPI | Intended target human-preference rate under assumptions | Shift violates those assumptions; failed coverage is retained. |

All corrected methods receive the same generated audit, including all labels
used to tune the coefficient. The raw method uses zero audit labels. Summary
statistics include bias, RMSE, interval width, target and native coverage,
and Monte Carlo standard errors. These describe simulation error under the
specified generator. Plug-in coverage MCSE is zero at observed coverage zero
or one; it does not imply certainty. RMSE MCSE uses a delta approximation.
Separate method MCSEs are not a paired significance test of method differences.

## Related work and scope

Difference and generalized regression estimation with auxiliary information
are classical survey methods. PPI explicitly identifies that connection;
the scalar residual correction here is not a new estimator principle.
This package does not implement general survey weights or unequal-probability
design inference. [Cassel, Särndal, and Wretman, 1976](https://doi.org/10.1093/biomet/63.3.615).

PPI and PPI++ supply the statistical foundation; wrapping them for judge
labels, adding tests, or using cluster sandwiches is not by itself a novelty
claim. The authors' repository also contains cross-fitting, bootstrap,
nonuniform-sampling, and distribution-shift methods. [PPI author repository](https://github.com/aangelopoulos/ppi_py).

Combining automatic metrics with human evaluation through control variates
also predates PPI. Chaganty, Mussmann, and Liang study that construction
for NLP evaluation and discuss bias from estimating its coefficient on
the same sample. Observable judge-agreement proxies here are applications
of this established idea, not the first correction of automatic evaluation.
[Chaganty, Mussmann, and Liang, ACL 2018, Sections 3.2–3.3](https://nlp.stanford.edu/pubs/chaganty2018price.pdf).

Finite-sample costs of power tuning also have direct prior analysis.
Mani, Xu, Lipton, and Oberst characterize settings where PPI++ increases
error and where same-sample variance estimates are optimistic. The
repository's stress results are setting-specific evidence, not discovery
of these effects. Their exact results depend on their estimator and
sampling regime; Gaussian thresholds or effectively infinite prediction
pools do not automatically apply to the constrained, per-pool or clustered
rules here. [No Free Lunch: Non-Asymptotic Analysis of Prediction-Powered
Inference, 2025/2026, v2](https://arxiv.org/abs/2505.20178v2).

Adaptive evaluation with fallback and reliable model selection already has
direct prior work in R-AutoEval+. That method addresses a sequential
model-selection problem, so comparing it with these fixed-sample mean
intervals requires aligning the decision target and protocol.
[Park, Zecchin, and Simeone, NeurIPS 2025, v2](https://arxiv.org/abs/2505.18659v2).

Misclassification correction, calibration/test uncertainty, and audit
allocation are studied specifically for LLM judges by Lee and colleagues.
Their transfer analysis assumes invariant conditional judge behavior;
it is not a guarantee under arbitrary changes in judge error.
[How to Correctly Report LLM-as-a-Judge Evaluations, ICML 2026, v4](https://arxiv.org/abs/2511.21140v4).

Judge reliability also involves the validity of the reference outcome.
JudgeBench supplies response pairs labeled for objective correctness,
which differs from human stylistic preference; a benchmark adapter must
preserve that distinction. Position bias and balanced-order mitigation also
predate this project. These are relevant external benchmarks and baselines,
not implemented contributions of the current study.
[JudgeBench](https://arxiv.org/abs/2410.12784),
[FairEval](https://arxiv.org/abs/2305.17926).

## Fixed finite-corpus inference by uniform group sampling

The [finite-corpus protocol](FINITE_CORPUS_PROTOCOL.md) defines a separate
design-based target: the recorded row mean of a complete fixed corpus. It
does not reinterpret the earlier empirical held-out means as population
confidence targets or transfer their normal intervals to this setting.
The dedicated `finite_corpus_mean` API receives full-frame `group_sizes`
and `group_prediction_totals`, plus `audited_group_indices` and only those
groups' `audited_outcome_totals`. It returns a `FiniteCorpusMeanResult`
with the point, interval, radius and candidate diagnostics; it does not
return a normal standard error.
Let the frame contain `G` groups and `M=sum_g n_g` rows, with bounded
outcomes `Y_i` and fully known proxies `F_i`. Write `Y_g` and `F_g` for
group totals. Draw exactly `k` groups uniformly without replacement and
observe every outcome in those groups. For a fixed coefficient,

```text
mu_C = sum_g Y_g/M
R_g(lambda) = Y_g - lambda F_g
theta_hat(lambda) = lambda sum_g F_g/M + G/(k M) sum_{g in S} R_g(lambda)
theta_hat(lambda) - mu_C = (G/M) [mean_S R(lambda) - mean_frame R(lambda)].
```

Group inclusion probability `k/G` makes each fixed-coefficient estimate
design-unbiased. At zero coefficient this is the Horvitz–Thompson expansion
of group totals; unequal group sizes do not permit replacing its denominator
with the number of sampled rows. The realized row-label cost is random at
fixed `k`. The full proxy total includes sampled groups, so there is no
independent prediction-pool variance or error allocation. Within-group
outcomes may be dependent because the finite corpus is held fixed.

Known support is `R_g(lambda) in [-lambda F_g, n_g-lambda F_g]`. Let
`L_lambda` be the width of the common enclosing interval over the complete
frame, and let `v_hat_lambda` be the audited group residual variance with
divisor `k` (`ddof=0`). Define

```text
rho(k,G) = 1-(k-1)/G                 if k <= G/2
           (1-k/G)(1+1/k)           if k > G/2.

r_H(lambda) = (G/M) L_lambda sqrt[rho(k,G) log(2K/alpha)/(2k)]

r_E(lambda) = (G/M) {
    sqrt[2 rho(k,G) v_hat_lambda log(10K/alpha)/k]
    + (7/3 + 3/sqrt(2)) L_lambda log(10K/alpha)/k
}.
```

These apply Bardenet and Maillard's Hoeffding–Serfling Corollary 2.5 and
empirical Bernstein–Serfling Theorem 4.3 to group residual totals. The latter
has one-sided failure `5 delta`; two signs and `K` fixed candidates require
`delta=alpha/(10K)`. The empirical variance convention is Eq. (26), not the
`ddof=1` convention of the package's iid normal estimator. See
[Bardenet and Maillard (2015), arXiv v2, reprint pp. 10, 19, 21–23](https://arxiv.org/abs/1309.4029v2),
*Concentration inequalities for sampling without replacement*,
Bernoulli 21(3), 1361–1385,
[DOI 10.3150/14-BEJ605](https://doi.org/10.3150/14-BEJ605).

A union bound guarantees simultaneous containment for the prespecified
coefficient grid within one method. Selecting its smallest untruncated
radius therefore preserves coverage for this corpus, but does not imply
that the selected point is design-unbiased. No cross-family selection,
continuous coefficient search, simultaneous guarantee across reported
cells, or superpopulation guarantee follows. When `k=G`, compute the
observed corpus mean and zero radius directly: the empirical-Bernstein
range term does not vanish merely because `rho` vanishes.

For these support bounds, `L_lambda >= max_g n_g = L_0`. Thus a range-only
radius-minimizing grid containing zero cannot improve on its zero candidate
at equal allocation. Variance-sensitive selection may help but has no
guaranteed efficiency advantage. The study compares agreement with the
constant row proxy `F_i=1`, whose group totals are known sizes. This control
uses no auxiliary judge information and can reveal benefits attributable
to group-size adjustment. Correct frame membership, uniform fixed-size
sampling and fully observed sampled groups are essential; general survey
weights, missing labels and arbitrary sampling designs remain outside scope.
