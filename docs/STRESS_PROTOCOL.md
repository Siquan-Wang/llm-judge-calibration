# Follow-up stress grid: protocol v2

This is a retrospective follow-up to the seven-scenario v1 study. The v1
results, including strong-judge improvements, few-cluster undercoverage and
shift failures, informed the choice to broaden the experiment. This is not a
preregistration or evidence that these scenarios occur with equal frequency.

## Question

Do v1 conclusions persist when human preference prevalence, judge quality and
audit size change? How much does treating repeated turns as independent distort
reported coverage?

## IID factorial grid

- Human preference prevalence: 0.2, 0.5, 0.8.
- Audit size: 20, 50, 200 independent observations.
- Judge sensitivity/specificity: strong (0.95, 0.90), moderate (0.90, 0.60),
  uninformative (0.60, 0.40), anticorrelated (0.20, 0.20).
- Target judge-only pool: ten times audit size, independent of the audit.
- Four methods on every identical draw: raw judge, human-only, coefficient-one
  PPI, tuned PPI. No scenario is dropped based on its result.
- 500 replications per cell, master seed 2027, for 36 cells.

The uninformative judge has a constant positive probability of 0.6. Varying the
human prevalence prevents accidental agreement of marginal rates at a single
truth from being mistaken for general judge usefulness.

Record the analytic oracle bounded coefficient as a diagnostic only. It must
never replace the sample-fitted coefficient in the methods being evaluated.

## Dependence ablation

Use 8, 30 and 100 independent audit clusters, ten times as many target clusters,
four exact repeated turns per cluster, prevalence 0.6, and the strong judge.
For each generated draw, run both correct cluster inference and deliberately
misspecified iid inference on the same repeated rows. All costs and effective
unit counts remain explicit. Do not count these six scenarios as independent
datasets: each correct/naive pair shares its generated observations.

## Outcomes

Retain every trial, including boundary coefficients and zero-width intervals.
Report bias, RMSE, interval width, coverage, Monte Carlo error, oracle and fitted
power. Method comparisons to human-only use paired per-replication losses.
Paired RMSE-difference uncertainty uses the multivariate delta method; separate
method MCSEs must not be subtracted as though methods used independent samples.
At zero observed coverage, plug-in MCSE zero is not evidence of certainty.

Plot each factorial cell separately. Counts of favorable cells are descriptive
of this chosen grid, not probabilities of success on deployment distributions.
Do not claim that minimizing fitted variance guarantees finite-sample MSE or
coverage improvements. The raw interval targets judge-positive rate, and
human-only intervals target audit prevalence; target-human coverage remains
a separate diagnostic in the record format.

## Reproduction and preservation

Keep v1 configuration and substantive results available. Commit deterministic
compressed full trials, summaries, exact generation settings and artifact hashes.
Use no model calls or private data. Review formula, shared-draw pairing and
claims before publication. Any result-driven change to this grid should be
labeled as a subsequent follow-up.
