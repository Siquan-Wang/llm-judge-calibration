# Sensitivity to the human reference

The same public comparisons, judge labels, question splits and audit costs; two different human outcomes.

[Retrospective protocol](../../docs/REFERENCE_SENSITIVITY.md) | [Methods](../../docs/METHODS.md) | [Data attribution](../mtbench/DATA_LICENSE.md)

## The two targets

- **Plurality outcome:** score the unique modal human category A/B/tie as 1/0/0.5; a tie for the most votes becomes 0.5.
- **Mean vote score:** `(A votes + 0.5 * tie votes) / total votes` for each comparison, then average comparisons equally.

Neither is latent ground truth. Mean vote scores preserve the average recorded score; the retained vote counts distinguish disagreement patterns that a mean alone cannot reveal. Both references depend on the observed annotator mix and number of votes. This is not a pooled-vote-weighted target or an estimate of inter-rater reliability. Single-vote comparisons provide no within-comparison disagreement information.

The snapshot contains 1,814 comparisons and 3,354 unique votes; 854 comparisons have one vote. The score changes for 305 comparisons, with an average absolute change of 3.852 percentage points across all comparisons. `reference_changes.csv` retains every comparison, including unchanged scores.

## Identical sampling, separate scoring

All 15 eligible model pairs, 30 deterministic splits per pair, fixed 25% evaluation questions, and nested audits of 20%, 40% and 60% of each pair's total questions. Every turn of a question stays together. Both definitions use exactly the original audit-reliability study's split algorithm. Each estimator sees only audit outcomes; hidden evaluation outcomes score its error afterwards.

The judge remains a frozen categorical A/B/tie score under both definitions. Errors compare each estimator to its own realized heldout reference mean. Changing the definition can change the audit correction, selected power, and evaluation target. A smaller error against one reference does not establish that reference as better.

| Reference | Audit questions | Method | MAE (pp) | RMSE (pp) | Mean power |
|---|---:|---|---:|---:|---:|
| Mean vote score | 20% | Human audit | 7.413 | 9.282 | 0.000 |
| Mean vote score | 20% | PPI (power 1) | 8.454 | 10.696 | 1.000 |
| Mean vote score | 20% | Tuned PPI | 7.072 | 8.900 | 0.246 |
| Mean vote score | 20% | Raw judge | 8.653 | 10.881 | -- |
| Mean vote score | 40% | Human audit | 6.359 | 8.113 | 0.000 |
| Mean vote score | 40% | PPI (power 1) | 7.067 | 8.967 | 1.000 |
| Mean vote score | 40% | Tuned PPI | 6.123 | 7.818 | 0.192 |
| Mean vote score | 40% | Raw judge | 8.653 | 10.881 | -- |
| Mean vote score | 60% | Human audit | 6.141 | 7.682 | 0.000 |
| Mean vote score | 60% | PPI (power 1) | 6.920 | 8.796 | 1.000 |
| Mean vote score | 60% | Tuned PPI | 5.937 | 7.469 | 0.161 |
| Mean vote score | 60% | Raw judge | 8.653 | 10.881 | -- |
| Plurality outcome | 20% | Human audit | 7.842 | 9.712 | 0.000 |
| Plurality outcome | 20% | PPI (power 1) | 8.610 | 10.957 | 1.000 |
| Plurality outcome | 20% | Tuned PPI | 7.480 | 9.319 | 0.258 |
| Plurality outcome | 20% | Raw judge | 9.132 | 11.429 | -- |
| Plurality outcome | 40% | Human audit | 6.506 | 8.307 | 0.000 |
| Plurality outcome | 40% | PPI (power 1) | 7.127 | 9.129 | 1.000 |
| Plurality outcome | 40% | Tuned PPI | 6.267 | 7.983 | 0.201 |
| Plurality outcome | 40% | Raw judge | 9.132 | 11.429 | -- |
| Plurality outcome | 60% | Human audit | 6.321 | 7.850 | 0.000 |
| Plurality outcome | 60% | PPI (power 1) | 7.112 | 8.946 | 1.000 |
| Plurality outcome | 60% | Tuned PPI | 6.113 | 7.613 | 0.169 |
| Plurality outcome | 60% | Raw judge | 9.132 | 11.429 | -- |

![Reference definition sensitivity](reference_sensitivity.svg)

## Does the relative method comparison change?

Negative gaps favor tuned PPI over human-only for that reference. These are descriptive gaps; repeated splits and model pairs share questions, so no independent-trial significance test is reported.

| Reference | Audit questions | Tuned minus human MAE (pp) | Pairs favoring tuned PPI |
|---|---:|---:|---:|
| Mean vote score | 20% | -0.340 | 11/15 |
| Mean vote score | 40% | -0.236 | 11/15 |
| Mean vote score | 60% | -0.203 | 11/15 |
| Plurality outcome | 20% | -0.362 | 10/15 |
| Plurality outcome | 40% | -0.238 | 9/15 |
| Plurality outcome | 60% | -0.208 | 11/15 |

Tuned PPI has lower aggregate MAE than human-only in 6/6 reference/budget combinations. The sign of the pair-level MAE gap changes in 5/45 matched pair/budget comparisons when the reference changes. Thus aggregate direction and pair-level sensitivity must be assessed separately; these signs are descriptive and do not imply statistical significance.

`paired_reference_differences.csv` aligns each method/pair/seed/budget across definitions; all changes are mean-vote minus plurality. It retains changes in predictions and evaluation targets separately. `per_pair.csv` and `paired_method_gaps.csv` retain heterogeneous pair-level outcomes. The original plurality results are preserved in their earlier report.

## Limits

- These data contain selected older models, 80 questions and unequal annotation counts. Neither definition establishes a population-wide preference distribution.
- One-way question clustering does not model shared annotator effects across questions. The derived snapshot lacks per-vote annotator identities, so this study cannot estimate those effects or resample individual annotators.
- The intervals target a population mean under representative independent pools and sufficient independent clusters. Heldout-reference errors are not interval coverage tests; existing synthetic stress tests document undercoverage.
- Vote counts are observed workload proxies. No new labels were acquired and no annotation-saving claim follows from this comparison.
- This is a retrospective robustness analysis, with both definitions reported regardless of outcome. It does not select a preferred reference by the resulting method ranking.

## Reproduce offline

```bash
pip install -e ".[dev,report]"
python examples/reference_sensitivity.py --output reports/reproduced/sensitivity --plot
```

The default uses 30 split seeds. Smaller `--seeds` runs are recorded as such in `config.json`. Input provenance, exact question manifests, package versions, source hashes and artifact hashes are included. No model API, private data or new human annotation is used.
