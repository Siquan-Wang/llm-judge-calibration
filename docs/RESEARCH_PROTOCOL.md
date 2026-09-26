# Human-audit reliability study: protocol v1

## Research question

When does an imperfect LLM judge improve human-preference estimation at a fixed
human-audit budget, and when does relying on the judge hurt?

The contribution is a reproducible empirical comparison and failure analysis,
not a new prediction-powered estimator. This is a versioned retrospective
analysis plan: the original data and development-time exploratory results were
available when choices were made. It is not a prospective preregistration on an
unseen dataset. The committed configuration records the final analysis choices.

## Methods and estimands

Compare four methods on identical samples: raw judge, human audit only,
coefficient-one PPI, and power-tuned PPI mean correction. For human score Y and
judge score F, the latter estimates `mean(Y_L) + lambda*(mean(F_U)-mean(F_L))`.
Wins, losses and ties score 1, 0 and 0.5. The target is observation-weighted
human preference under this convention, not objective correctness.

The tuning coefficient minimizes the estimated per-pool variance over [0, 1].
It uses audit human labels and both pools' judge labels only. There is no tuning
on evaluation human labels, no selection of successful splits and no clipping
of point estimates or confidence bounds. The same audit is used for tuning and
correction, so intervals rely on asymptotic plug-in reasoning. The cluster
sandwich extension is explicitly distinguished from the iid author software.

Primary foundations: [PPI](https://arxiv.org/abs/2301.09633),
[PPI++](https://arxiv.org/abs/2311.01453) and
[author code](https://github.com/aangelopoulos/ppi_py).

## Fixed-target MT-Bench experiment

- Reuse the immutable, attributed public labels at `reports/mtbench/labels.csv`.
  Validate its SHA-256 against its source manifest before running.
- Include all canonically oriented model pairs with at least 40 questions.
- For each pair and seed 0 through 29, permute sorted question IDs once.
  Reserve the first floor(0.25 G) questions for evaluation at every budget.
- Take nested audit prefixes of floor(0.2 G), floor(0.4 G), floor(0.6 G)
  questions from the remaining bank. Unused human labels never enter inference.
- All turns from a question stay together. Pair-specific partitions may differ;
  comparisons sharing a question across pairs are not independent replicates.
- The raw judge estimate and the realized evaluation human reference remain
  fixed across budgets. Equal pair/split weighting defines the summary.
- Primary outcomes: mean absolute error and root mean squared error against
  the realized heldout reference. Also retain signed error, interval widths,
  selected power, per-pair results, every trial and all split IDs.
- Report actual audited questions, comparison labels and underlying votes.
  Reused public vote counts are workload proxies, not measured annotation cost.
- Evaluation human labels may be validated upfront but are used numerically
  only after inference, for error scoring.
  Population confidence intervals do not target a fixed subset's realized mean;
  do not report MT-Bench interval containment as population coverage.

The historical complementary-pool experiment remains available unchanged in
`reports/mtbench`. Its evaluation subset changes with budget; it answers a
different question and is not directly pooled with this study.

## Known-truth simulation

Use the explicit, versioned scenario configuration exported by
`judgecal.simulation`. Run 1,000 independent replications per scenario with a
fixed master seed. Scenarios cover strong, weak and anticorrelated iid judges,
repeated turns within questions, few independent questions, prevalence shift,
and conditional judge-error shift. All methods receive the same generated data.

Report bias, RMSE, mean interval width, coverage of the known human population
mean, Monte Carlo standard error and mean selected power, retaining every
individual power for distributional inspection. The raw
judge's interval estimates its own mean; its containment of human truth is a
bias diagnostic. Shift scenarios deliberately violate transport assumptions;
they are stress tests, not settings covered by the PPI validity claim. Every
draw is retained, including zero-width intervals and failures of nominal
coverage. Synthetic findings do not establish performance on real distributions.

## Interpretation and exclusions

- No guaranteed label-saving percentage, universal efficiency improvement,
  finite-sample validity or new statistical novelty will be inferred.
- MT-Bench uses older responses and 80 selected questions. A result here does
  not establish reliability on modern judge families, tasks or rubrics.
- Human plurality is a noisy reference. Vote aggregation and judge-order
  inconsistency remain potential sensitivity-analysis directions.
- Repeated-split variability is descriptive. No naive standard errors across
  correlated pair/split rows or cherry-picked significance tests are used.
- Exclude private/company data, new paid model calls and invented observations.

## Reproducibility and review

Commit code, configuration, source checksums, environment versions, every trial,
summary tables and plots. Independently review estimator algebra, leakage and
claims. Test limiting cases, reversal, dependence, deterministic splits and
hidden-label isolation. Each change to this protocol after the published v1
results must be labeled as a follow-up analysis.
