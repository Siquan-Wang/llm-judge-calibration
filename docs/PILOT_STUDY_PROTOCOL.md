# Independent-pilot tuning at a fixed label budget

This study compares complete prediction-assisted mean-estimation procedures
that spend the same total number of reference labels. An independent pilot
can choose a coefficient before the final inference observations are used.
That separation supports exact conditional unbiasedness and existing
finite-sample bounds, but leaves fewer labels for the final residual mean.
Whether it improves point error or interval width is an empirical question.

This is a retrospective extension motivated by earlier studies and
development checks. The specified design is frozen before the final full
run; it is not preregistered. All configured cells, methods and repetitions
are retained. Sample splitting, control variates, PPI tuning and the
concentration inequalities are established methods. Neither the pilot rule
nor the algebra below is claimed as a new estimator or theorem.

Observed results and artifacts belong in the
[pilot-study report](../reports/pilot-study/REPORT.md). Reproduce with:

```bash
python examples/independent_pilot_study.py --output reports/reproduced/pilot-study --plot
```

## Population target and fixed design

The target is the population mean `mu=E[Y]` of a specified binary outcome,
not a realized held-out mean or latent human consensus. A frozen binary
proxy `F` has sensitivity `P(F=1|Y=1)` and specificity `P(F=0|Y=0)`.
Within each law, observations are iid and both pools have the same proxy
marginal. The estimator receives no population truth or oracle moments.

| Profile | `P(Y=1)` | Sensitivity | Specificity |
|---|---:|---:|---:|
| Balanced strong | .50 | .95 | .90 |
| Near-boundary strong | .95 | .95 | .90 |
| Balanced uninformative | .50 | .60 | .40 |
| Balanced perfect | .50 | 1.00 | 1.00 |

Each profile uses total label budgets `B in {20,200,1000}` and prediction
pool size `N=10*B`: twelve scenarios. The full study uses 2,000 independent
repetitions per scenario, master seed 2031 and `alpha=.05`. Eight methods
produce 192,000 trial rows and 96 method/scenario summaries. These are
twelve simulation laws at their respective budgets, not twelve empirical
datasets. The perfect proxy is a structural control; the estimator is not
told to replace the known residual range by zero.

For each scenario and repetition, generate one bank of `B` iid paired
observations `(Y,F)` and a separate pool of `N` iid proxy observations.
For pilot fraction `f` equal to .20 or .50, the first `P=floor(f*B)` bank
rows form the pilot, and the remaining `m=B-P` rows form the final residual
sample. Fixed positions make the split independent of observed values.

| Total labels `B` | 20% pilot / final rows | 50% pilot / final rows | `N` |
|---:|---:|---:|---:|
| 20 | 4 / 16 | 10 / 10 | 200 |
| 200 | 40 / 160 | 100 / 100 | 2,000 |
| 1,000 | 200 / 800 | 500 / 500 | 10,000 |

Every pilot is independent of its own final residual and prediction pools.
Across procedures, pilots are nested and some rows change roles. This
overlap is intentional pairing for method comparisons, not within-method
reuse. Full-audit procedures use the entire bank, and all methods use the
same prediction pool. For each pilot procedure, all `B` labels are charged,
including those used only for tuning. A zero pilot coefficient does not
recover the full-`B` audit mean: it leaves a final mean based on `m` labels.

Draws use NumPy Philox with addressable SeedSequence inputs
`[master_seed, stable_profile_code, B, repetition, stream_code]`.
Profile codes are `1=balanced_strong`, `2=boundary_strong`,
`3=uninformative`, and `4=perfect`; stream `0` identifies the labeled bank
and stream `1` the prediction pool. Predictions in the latter are drawn
directly from `Bernoulli(q)`, where
`q=mu*sensitivity+(1-mu)*(1-specificity)`; no target outcomes are generated.
The configuration records this mapping and the NumPy version; Python's
process-randomized `hash` is not used. Reordering methods
or scenarios, or requesting additional repetitions, must preserve existing
keyed draws. There is no stateful dependence on method enumeration.

## Eight procedures and the comparison they identify

| Method ID | Final labeled rows | Coefficient and interval |
|---|---:|---|
| `human_normal` | `B` | Zero coefficient; normal audit interval |
| `human_eb` | `B` | Zero coefficient; scalar empirical Bernstein (EB) |
| `same_audit_normal` | `B` | Existing automatic coefficient; normal interval |
| `same_audit_grid_eb` | `B` | Prespecified grid; simultaneous EB selection |
| `pilot20_normal` | `.80B` | 20% pilot coefficient; fixed-power normal |
| `pilot20_eb` | `.80B` | Same 20% pilot coefficient and point; scalar EB |
| `pilot50_normal` | `.50B` | 50% pilot coefficient; fixed-power normal |
| `pilot50_eb` | `.50B` | Same 50% pilot coefficient and point; scalar EB |

The grid is `(0,.25,.50,.75,1)`. Its existing API chooses the minimum
untruncated radius after simultaneous calibration, with smaller power
breaking exact ties. The pilot variants pass a single coefficient to
the existing `prediction_powered_mean` and `finite_sample_mean` APIs.
The study adds no statistical public API or duplicate bound implementation.

Using pilot moments with matching sample divisors, set

```text
lambda_P = projection_to_[0,1]( N/(m+N) * s_YF,P / s_F,P^2 ).
```

Exactly constant pilot predictions select zero. All such draws, including
four-row constant pilots, remain in the results. Clip the complete
coefficient after multiplying the size factor. The pilot selector may use
`m,N` as known design counts, but no final outcomes, final proxy values,
population moments or evaluation error. Pilot rows are not recycled into
the final residual or prediction mean.

For binary pilot pairs, moments use integer counts before final division.
With `k=P`, positive counts `n_Y,n_F` and joint-positive count `n_11`, the
covariance numerator is `k*n_11-n_Y*n_F` and the variance numerator is
`n_F*(k-n_F)`, both over `k*(k-1)`. The coefficient is formed directly from
`N*covariance_numerator / [(m+N)*variance_numerator]`, with integer sign and
boundary comparisons before division. Thus exactly zero sample covariance
selects exactly zero power and the standalone zero-power EB allocation.
The generic fractional-score path uses scaled moments without an arbitrary
epsilon cutoff. This arithmetic distinction changes no sampling assumption
or candidate grid.

The legacy same-audit normal rule instead selects

```text
lambda_auto = projection_to_[0,1](
    s_YF,B / [s_F,B^2 + (B/N)*s_F,U^2]
).
```

These finite-sample moment functionals differ: the pilot rule estimates
the common proxy variance from the pilot alone, while the legacy rule uses
separate audit and prediction variances. Their corresponding population
oracles also use different final labeled counts. The experiment compares
complete fixed-budget procedures; it is not an isolated causal test of
independence. A performance difference cannot be attributed solely to
removing same-audit dependence. The comparator's existing behavior is
preserved and the difference is disclosed.

## Conditional point and variance calculation

Condition on the independent pilot, so `lambda_P` is fixed. For the final
residual sample `L` and prediction pool `U`,

```text
theta_P = mean(Y_L) + lambda_P * [mean(F_U) - mean(F_L)]
E[theta_P | pilot] = mu
Q(lambda) = Var(Y-lambda*F)/m + lambda^2*Var(F)/N
Var(theta_P | pilot) = Q(lambda_P)
Var(theta_P) = E[Q(lambda_P)].
```

The final equality follows because the conditional mean is always `mu`.
Pilot randomness still affects `E[Q(lambda_P)]`; it has not been removed
from unconditional risk. Sample residual and prediction variances with
`ddof=1` give a conditionally unbiased variance estimate on fresh final
data. Neither this fact nor exact point unbiasedness establishes normal
coverage at small sample sizes. Normal intervals remain asymptotic and
can be zero width or inaccurate near the outcome boundary.

For `Var(F)>0`, define the unrestricted oracle
`lambda_0=N/(m+N)*Cov(Y,F)/Var(F)` and
`A=(1/m+1/N)*Var(F)`. Completing the square gives

```text
Q(lambda) = Var(Y)/m
            - N*Cov(Y,F)^2 / [m*(m+N)*Var(F)]
            + A*(lambda-lambda_0)^2.
```

This is elementary variance accounting. The last term includes estimation
error and any clipping loss. When the oracle lies outside `[0,1]`, the
first two terms are only an unrestricted lower bound. Even an interior
known oracle beats the full-budget human variance `Var(Y)/B` only if

```text
[N/(m+N)] * Corr(Y,F)^2 > P/B,
```

assuming positive outcome variance. A fitted pilot pays the additional
expected squared-coefficient term. At zero covariance its variance is at
least `Var(Y)/m`, which exceeds the full-budget audit variance when `P>0`.
These identities are diagnostic, not new thresholds claimed from a paper,
and oracle values never select a fitted method or coefficient.

## Finite EB coverage and its limits

Conditional on the pilot, residuals lie in the known range
`[-lambda_P,1]` and remain iid. For a sample of `k>=2` values in a known
interval of length `L`, the existing two-sided EB radius is

```text
B_EB(Z,delta,L) = sqrt(2*s_Z^2*log(4/delta)/k)
                  + 7*L*log(4/delta)/(3*(k-1)).
```

For positive pilot power, scalar EB uses

```text
radius = B_EB(Y_L-lambda_P*F_L,alpha/2,1+lambda_P)
         + lambda_P*B_EB(F_U,alpha/2,1).
```

The two components use their own sample counts. Conditional failure
probability is at most `alpha`; averaging over the pilot preserves that
bound. A continuous range of possible pilot coefficients requires no
same-data grid penalty because only one coefficient is fixed conditional
on the pilot. If pilot power is zero, the existing scalar API uses the
sharper standalone `B_EB(Y_L,alpha,1)`. Both branches are selected before
the final data are observed.

The grid procedure instead protects selection on the same inference
observations: each residual event receives `alpha/(2*K)`, and one shared
prediction event receives `alpha/2`, with `K=5`. Its zero candidate retains
the grid allocation. Details and constants are in the
[finite-sample protocol](FINITE_SAMPLE_PROTOCOL.md).

Each declared EB procedure has its own guarantee. Choosing the narrower
pilot fraction, or the narrower grid/pilot interval, after inspecting final
data does not inherit those individual guarantees. No such selector is
reported. Observed residual ranges cannot replace the known ranges.
Intervals remain untruncated, including widths exceeding one.

Coverage is for a population mean under the stated iid sampling law,
conditional on an independent pilot where applicable. It is not conditional
on final proxy composition and does not cover a frozen empirical target
pool by this argument. There are no cross-fit, grouped, finite-population
without-replacement, shift-robust, sequential-stopping or latent-truth
claims in this study.

## Records and interpretation

Record pilot size, final labeled count and total label cost separately.
Trial fields are `n_pilot`, `n_labeled` (final `m`, or `B` for full-audit
methods), `n_unlabeled`, `total_budget`, and `total_labels_used`.
The last two both equal `B` for every method. The field
`unique_prediction_scores_used` is zero for `human_normal` and `human_eb`,
and `B+N` for the other procedures. Separately, `proxy_scores_supplied`
records the common generated bank of `B+N` values for every method.
Human-only procedures need no proxy scores mathematically, even when
arrays are supplied to the shared wrapper. Logical observation counts are not
annotation dollars or measured model-inference cost. Method-specific
utilization and the common generated data budget must remain distinguishable.

Retain points, errors, coefficients, pilot degeneracy diagnostics, interval
endpoints, untruncated widths, coverage and out-of-domain/zero-width events
for every draw. Normal and EB versions of each pilot must have identical
points and coefficients. `estimated_variance` is the normal estimated SE
squared and is unavailable for EB. The known-law `conditional_variance`
diagnostic is available for human-only and independently tuned procedures;
it is not assigned to the same-audit auto/grid methods or used for tuning.
Report bias and Monte Carlo standard error (MCSE),
MSE/RMSE, widths, and coverage with pointwise exact-binomial Monte Carlo
intervals. Observed 100% coverage does not mean the coverage probability is
known to be one. Finite-theorem applicability is an assumption-based flag
for EB procedures, never a property inferred from observed coverage.

Matched-draw MSE contrasts use full-budget `human_normal` and
`same_audit_normal` as baselines. Interval contrasts distinguish the normal
family from EB and its full-budget human/grid comparators. Paired contrast
MCSE uses the standard deviation of within-draw differences, not an
independent-method approximation. No cells, pilot fractions or targets are
pooled into a general efficiency score. All uncertainty summaries are
pointwise simulation precision, not simultaneous claims across the matrix.

Coverage scoring uses the existing numerical convention for every method:
`low-tolerance <= mu <= high+tolerance`, where
`tolerance=8*float64_epsilon*max(1,abs(low),abs(high),abs(mu))`.
This guards floating-point boundary comparisons only; it does not change
points, endpoints, widths, variances or losses. Strict containment remains
reconstructible from the saved endpoints and population truth. The
mathematical coverage statements concern the underlying real-valued bounds.

## Prior work

Control-variate correction for NLP evaluation and the bias of estimating
its coefficient from the same data predate this study: Chaganty, Mussmann
and Liang (ACL 2018), Sections 3.2–3.3 and Proposition 3.1,
[author paper](https://nlp.stanford.edu/pubs/chaganty2018price.pdf).

The finite-prediction-pool population oracle follows PPI++ Example 6.1:
Angelopoulos, Duchi and Zrnic, *PPI++: Efficient Prediction-Powered
Inference*, [v2, 2024](https://arxiv.org/html/2311.01453v2).
The pilot-only moment estimate here is not the authors' pooled-variance
implementation, and their normal limit is asymptotic.

Mani, Xu, Lipton and Oberst explicitly study split/cross-fit coefficient
tuning, unbiasedness and finite-sample efficiency costs in *No Free Lunch:
Non-Asymptotic Analysis of Prediction-Powered Inference*,
[v2, 2026](https://arxiv.org/html/2505.20178v2), Definition 3.5,
Proposition 4.3 and Theorem 4.1. Their principal two-fold construction and
effectively infinite prediction pool differ from this clipped one-way
finite-pool protocol. Its results are not imported as guarantees here.

The bounded-mean ingredient is Maurer and Pontil (2009), *Empirical Bernstein
Bounds and Sample Variance Penalization*, Theorem 4,
[v1, PDF p. 2](https://arxiv.org/pdf/0907.3740v1#page=2).
Two-tail rescaling, event allocation and conditioning on an independent
pilot apply existing results; they do not establish a new theorem or a
guaranteed label saving.
