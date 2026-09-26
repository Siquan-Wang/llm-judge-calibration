# Methods and interpretation

`judgecal` estimates a population human-preference rate using an imperfect,
frozen judge and a smaller human audit. It implements established mean
correction and power-tuning ideas, with an explicit cluster-variance
adaptation. The package and experiments do not introduce a new PPI estimator
or establish general annotation savings.

## Target and sampling units

For a consistently oriented comparison, let the human score `Y` and judge
score `F` equal 1 for a target-model win, 0 for a loss, and 0.5 for a tie.
The population target is `theta = E[Y]` under that convention. This is a
human-preference target; it is not automatically factual correctness, an
individual annotator's preference, or latent consensus.

The labeled audit contains aligned `(Y_L, F_L)` values, and a separate
prediction-only pool contains `F_U`. Human labels in the prediction-only
pool are unavailable to the estimator. The pools must be independent and
represent the same relevant population, and both use the same frozen judge.
When the audit human mean equals the target human mean and judge means are
equal across pools, the correction is unbiased for a fixed power. The
documented same-population condition is a stronger, convenient sampling
assumption. Changing prompts, model versions, or audit selection can violate
it.

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
the human audit mean. The original framework is due to Angelopoulos,
Bates, Fannjiang, Jordan, and Zrnic; efficient power tuning is developed by
Angelopoulos, Duchi, and Zrnic in PPI++. [Original PPI paper](https://arxiv.org/abs/2301.09633),
[PPI++ paper](https://arxiv.org/abs/2311.01453).

Write `V_L(X)` and `V_U(X)` for estimated variances of sample means, and
`C_L(Y,F)` for the paired covariance of the audit means. This implementation
minimizes

```text
Q(lambda) = lambda^2 * V_U(F) + V_L(Y - lambda * F)
          = V_L(Y) - 2 * lambda * C_L(Y,F)
            + lambda^2 * [V_L(F) + V_U(F)].

lambda_hat = projection_to_[0,1](C_L(Y,F) / [V_L(F) + V_U(F)]).
```

If the denominator is zero, the implementation selects zero. The bounded
coefficient is a deliberate choice: an empirically nonpositive covariance
selects the human-only endpoint. It does not exploit a negatively correlated
judge by using a negative coefficient. An unconstrained minimum above one
selects one. The point estimate and interval endpoints themselves are never
clipped to `[0,1]`.

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
in `ppi_py` uses pooled judge variance in its automatic coefficient, with
its own finite-sample covariance conventions. Its normal-interval variance
also uses different finite-sample moment divisors. Consequently numerical
equality at finite sample sizes is not expected. This package uses separate
pool variances so its selected coefficient minimizes exactly the variance
formula it reports; the cluster version applies the same principle to
cluster moments. This is an implementation choice, not a claimed statistical
innovation or a claim of superiority to `ppi_py`.
[Authors' code](https://github.com/aangelopoulos/ppi_py/blob/main/ppi_py/ppi.py).

## What same-audit tuning does and does not justify

`power='auto'` uses the audit labels and both judge pools, without accessing
evaluation human labels. It estimates one coefficient; it does not fit or
fine-tune the underlying judge. To see why same-audit scalar tuning can be
first-order valid, compare a consistent coefficient with its deterministic
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

PPI and PPI++ supply the statistical foundation; wrapping them for judge
labels, adding tests, or using cluster sandwiches is not by itself a novelty
claim. The authors' repository also contains cross-fitting, bootstrap,
nonuniform-sampling, and distribution-shift methods. [PPI author repository](https://github.com/aangelopoulos/ppi_py).

Adaptive evaluation with fallback and reliable model selection already has
direct prior work in R-AutoEval+. That method addresses a sequential
model-selection problem, so comparing it with these fixed-sample mean
intervals requires aligning the decision target and protocol.
[Park, Zecchin, and Simeone](https://arxiv.org/abs/2505.18659).

Misclassification correction, calibration/test uncertainty, and audit
allocation are studied specifically for LLM judges by Lee and colleagues.
Their transfer analysis assumes invariant conditional judge behavior;
it is not a guarantee under arbitrary changes in judge error.
[How to Correctly Report LLM-as-a-Judge Evaluations](https://arxiv.org/abs/2511.21140).

Judge reliability also involves the validity of the reference outcome.
JudgeBench supplies response pairs labeled for objective correctness,
which differs from human stylistic preference; a benchmark adapter must
preserve that distinction. Position bias and balanced-order mitigation also
predate this project. These are relevant external benchmarks and baselines,
not implemented contributions of the current study.
[JudgeBench](https://arxiv.org/abs/2410.12784),
[FairEval](https://arxiv.org/abs/2305.17926).
