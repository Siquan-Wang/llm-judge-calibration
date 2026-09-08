# MT-Bench human-label-budget benchmark

A reproducible offline-label experiment. No new LLM calls or human annotations were purchased.

## Protocol

- Dataset: `lmsys/mt_bench_human_judgments`, revision `f7d2896d2cc5d80f8b55c2bbc722613555233c25`.
- Aligned comparisons: 1814; unique questions: 80; eligible model pairs: 15.
- Every pair with at least 40 questions is included; no pair is selected by its result.
- Frozen budgets: 20%, 40%, 60% of questions; seeds 0 through 29. Within each model pair, all turns of a question stay together. Separate pairs can use different partitions.
- A is the alphabetically first model in each pair; ties receive half a win.
- Human evaluation labels are withheld from inference and used only to score estimates afterward.
- PPI uses the untuned, coefficient-one mean correction. Both uncertainty terms use question-cluster variance estimates.
- Each pair/seed receives equal weight in the summary. Repeated splits are correlated and describe this dataset, not independent replications.

## Results

Error is measured against the realized human mean on the held-out questions. It is not a population-interval coverage test.

| Human-audit questions | Estimator | Mean absolute error (pp) | Pair/split trials |
|---:|---|---:|---:|
| 20% | human_only | 6.714 | 450 |
| 20% | ppi | 7.293 | 450 |
| 20% | raw_judge | 8.293 | 450 |
| 40% | human_only | 5.492 | 450 |
| 40% | ppi | 5.769 | 450 |
| 40% | raw_judge | 8.398 | 450 |
| 60% | human_only | 5.538 | 450 |
| 60% | ppi | 5.772 | 450 |
| 60% | raw_judge | 8.559 | 450 |

PPI has lower aggregate error than the raw judge at 3/3 budgets and lower error than human-only at 0/3 budgets. These results do not establish annotation savings.

Per-pair results, every trial and explicit question split IDs are committed alongside this report.

## Known-truth simulation

Each iid scenario uses 1,000 replications, seed 2026, a true human win rate of 0.6, 200 audited labels and 2,000 prediction-only labels. The original moderate-judge scenario is retained; strong and uninformative judges illustrate dependence on judge quality.

| Scenario (sensitivity, specificity) | Raw RMSE | Human-only RMSE | PPI RMSE | PPI 95% coverage | Monte Carlo SE |
|---|---:|---:|---:|---:|---:|
| moderate (0.9, 0.6) | 0.1011 | 0.0338 | 0.0358 | 0.926 | 0.008 |
| strong (0.95, 0.9) | 0.0153 | 0.0338 | 0.0222 | 0.944 | 0.007 |
| uninformative (0.6, 0.4) | 0.0104 | 0.0338 | 0.0508 | 0.938 | 0.008 |

The original moderate-judge case covers 92.6%, below the nominal 95% in this finite-sample experiment. These are specified synthetic models, not evidence of coverage on MT-Bench or under distribution shift. Undercoverage and cases worse than human-only are retained. Normal intervals are asymptotic.

## Limits and interpretation

- MT-Bench contains 80 selected questions and older model outputs. This is a small case study, not a current model leaderboard.
- Human plurality labels are a reference, not infallible truth. Tied vote counts and judge order inconsistencies are retained as ties. The estimand averages aggregated comparison labels, observation-weighted across retained turns, not individual votes or equally weighted questions.
- Randomly sampled, disjoint audit/evaluation questions must represent the same population. Biased auditing, changing judges or distribution shift can invalidate the correction.
- Cluster-normal intervals are asymptotic; some audit budgets have few questions. Estimates and intervals are intentionally not clipped to [0, 1].
- Original PPI can be less efficient than human-only estimation when the judge is weak. No budget or pair was tuned to hide such cases.
- The report compares point-estimation errors, not paid annotation cost or a guaranteed label-saving percentage.

## Reproduce

```bash
pip install -e ".[data,dev,report]"
python examples/benchmark_mtbench.py --plot
# Reuse committed labels without network access:
python examples/benchmark_mtbench.py --input reports/mtbench/labels.csv --output /tmp/judgecal-report
```

See `dataset.json` for the pinned source and transformation counts, `environment.json` for package versions, and `DATA_LICENSE.md` for attribution.
