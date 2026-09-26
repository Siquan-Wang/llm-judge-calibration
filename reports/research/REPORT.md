# Human-audit reliability study

A fixed-target MT-Bench case study and known-truth stress tests.

## Question and scope

When does an imperfect judge help estimate human preference at a fixed audit budget? This study implements established PPI/PPI++ mean correction and examines its failure modes. It does not introduce a new estimator or establish a general label-saving guarantee.

[Versioned analysis plan](../../docs/RESEARCH_PROTOCOL.md) | [Methods](../../docs/METHODS.md) | [Data attribution](../mtbench/DATA_LICENSE.md)

## Fixed-target MT-Bench experiment

All 15 eligible model pairs; 30 deterministic splits per pair. A fixed 25% of each pair's questions form evaluation; nested audits use 20%, 40% and 60% of its original questions. Every turn of a question stays together. The evaluation human reference and raw judge estimate are identical across budgets within a split.

Errors compare to the realized heldout human plurality mean. They are not population-confidence-interval coverage measurements. Pairs and repeated splits share questions; their rows are not independent replications.

| Audit questions | Method | MAE (pp) | RMSE (pp) | Mean power |
|---:|---|---:|---:|---:|
| 20% | Human audit | 7.842 | 9.712 | 0.000 |
| 20% | PPI (power 1) | 8.610 | 10.957 | 1.000 |
| 20% | Tuned PPI | 7.480 | 9.319 | 0.258 |
| 20% | Raw judge | 9.132 | 11.429 | -- |
| 40% | Human audit | 6.506 | 8.307 | 0.000 |
| 40% | PPI (power 1) | 7.127 | 9.129 | 1.000 |
| 40% | Tuned PPI | 6.267 | 7.983 | 0.201 |
| 40% | Raw judge | 9.132 | 11.429 | -- |
| 60% | Human audit | 6.321 | 7.850 | 0.000 |
| 60% | PPI (power 1) | 7.112 | 8.946 | 1.000 |
| 60% | Tuned PPI | 6.113 | 7.613 | 0.169 |
| 60% | Raw judge | 9.132 | 11.429 | -- |

Tuned PPI has lower aggregate MAE than human-only at 3/3 budgets and lower MAE than coefficient-one PPI at 3/3 budgets. These are descriptive comparisons on this fixed dataset; tuning need not improve every split or model pair.

| Audit questions | Pairs where tuned PPI has lower mean MAE than human-only |
|---:|---:|
| 20% | 10/15 |
| 40% | 9/15 |
| 60% | 11/15 |

![Fixed-target budget results](fixed_target_budget.svg)

The full `trials.csv`, `per_pair.csv` and `split_manifest.json` retain all outcomes and sampling choices. `human_comparisons_used` and `human_votes_used` count the audited comparison labels and underlying public votes; these are workload proxies, not a priced annotation study.

## Known-truth stress tests

Each scenario uses 1,000 independent replications from a fixed master seed (2026). See `simulation_config.json` for exact generation mechanisms, population truths and validity scopes. All four methods share the same draws. `simulation_trials.csv` retains every replication.

| Scenario | Method | Bias | RMSE | Width | Human-truth coverage | MC SE |
|---|---|---:|---:|---:|---:|---:|
| iid_strong | Raw judge | 0.0105 | 0.0151 | 0.0427 | 0.846 | 0.011 |
| iid_strong | Human audit | 0.0018 | 0.0348 | 0.1357 | 0.952 | 0.007 |
| iid_strong | PPI (power 1) | 0.0015 | 0.0211 | 0.0844 | 0.952 | 0.007 |
| iid_strong | Tuned PPI | 0.0017 | 0.0198 | 0.0784 | 0.951 | 0.007 |
| iid_weak | Raw judge | -0.0401 | 0.0416 | 0.0435 | 0.049 | 0.007 |
| iid_weak | Human audit | -0.0003 | 0.0338 | 0.1358 | 0.954 | 0.007 |
| iid_weak | PPI (power 1) | 0.0028 | 0.0467 | 0.1884 | 0.960 | 0.006 |
| iid_weak | Tuned PPI | 0.0000 | 0.0335 | 0.1349 | 0.950 | 0.007 |
| iid_anticorrelated | Raw judge | -0.1997 | 0.2000 | 0.0429 | 0.000 | 0.000 |
| iid_anticorrelated | Human audit | -0.0019 | 0.0360 | 0.1359 | 0.942 | 0.007 |
| iid_anticorrelated | PPI (power 1) | -0.0036 | 0.0675 | 0.2578 | 0.946 | 0.007 |
| iid_anticorrelated | Tuned PPI | -0.0019 | 0.0360 | 0.1359 | 0.942 | 0.007 |
| clustered_repeated_turns | Raw judge | 0.0100 | 0.0249 | 0.0855 | 0.914 | 0.009 |
| clustered_repeated_turns | Human audit | 0.0018 | 0.0739 | 0.2708 | 0.929 | 0.008 |
| clustered_repeated_turns | PPI (power 1) | 0.0017 | 0.0434 | 0.1663 | 0.952 | 0.007 |
| clustered_repeated_turns | Tuned PPI | 0.0023 | 0.0415 | 0.1538 | 0.935 | 0.008 |
| few_clusters | Raw judge | 0.0110 | 0.0555 | 0.2136 | 0.938 | 0.008 |
| few_clusters | Human audit | 0.0003 | 0.1792 | 0.6630 | 0.822 | 0.012 |
| few_clusters | PPI (power 1) | 0.0015 | 0.1043 | 0.3754 | 0.966 | 0.006 |
| few_clusters | Tuned PPI | 0.0081 | 0.1196 | 0.3298 | 0.901 | 0.009 |
| prevalence_shift | Raw judge | -0.0249 | 0.0269 | 0.0391 | 0.300 | 0.014 |
| prevalence_shift | Human audit | -0.3507 | 0.3523 | 0.1358 | 0.000 | 0.000 |
| prevalence_shift | PPI (power 1) | -0.1048 | 0.1088 | 0.1152 | 0.042 | 0.006 |
| prevalence_shift | Tuned PPI | -0.1975 | 0.2003 | 0.1016 | 0.000 | 0.000 |
| conditional_error_shift | Raw judge | -0.0602 | 0.0613 | 0.0437 | 0.000 | 0.000 |
| conditional_error_shift | Human audit | 0.0009 | 0.0351 | 0.1357 | 0.949 | 0.007 |
| conditional_error_shift | PPI (power 1) | -0.0706 | 0.0742 | 0.0848 | 0.098 | 0.009 |
| conditional_error_shift | Tuned PPI | -0.0545 | 0.0585 | 0.0786 | 0.239 | 0.013 |

![Known-truth simulation](simulation.svg)

Additional Monte Carlo errors are in `simulation_summary.csv`. Coverage uses known target human truth; the raw interval estimates the judge-positive rate, so its human-truth containment is a bias diagnostic. Native-parameter containment is also retained separately.

Prevalence shift and conditional-error shift are separate, deliberate assumption failures. Tuning controls estimated variance; it does not fix audit-to-target bias. Few independent clusters and fitted tuning weights can impair finite-sample coverage. No failed or zero-width trial is discarded.

## What can and cannot be concluded

- The tuned method is an implementation of established mean-estimation ideas, with an explicitly documented one-way cluster sandwich adaptation.
- Human plurality is a reference, not infallible ground truth. MT-Bench has only 80 selected questions and older model outputs; generalization to new judges and tasks is untested.
- These budget curves hold the evaluation set fixed. Historical complementary-pool results remain separately available in `reports/mtbench` and are not pooled with this study.
- Asymptotic normal intervals are neither distribution-free nor guaranteed at small sample sizes. Synthetic coverage is evidence only for the specified generating process.
- No new human annotation, model API calls, private data or externally priced compute were used.

## Reproduce offline

```bash
pip install -e ".[dev,report]"
python examples/research_study.py --output reports/reproduced/research --plot
```

Default arguments reproduce this complete protocol. `--repetitions` and `--seeds` support explicitly smaller smoke runs and are recorded in `config.json`; do not present those as the full study. Source checksums, package versions and artifact hashes are in `reproducibility.json`. The input CSV is checked against the pinned provenance before inference.
