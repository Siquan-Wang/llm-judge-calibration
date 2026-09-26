# Reproducible research evidence

Eleven retrospective findings, each tied to pinned CSV bytes, an exact selector and a named target. The [structured crosswalk](research_evidence.json) retains metric names, exact values, denominators, uncertainty and dataset lineage. These studies do not introduce a new estimator or prove general label savings.

Rates/errors below use percentages or percentage points (pp) as labeled. Paired MSE contrasts use squared fractions. Simulation Monte Carlo uncertainty applies within the declared scenario; fixed-corpus split averages have no iid MCSE.

## E01: Tuned correction improves aggregate fixed-target MAE at all three budgets

20% budget: 7.842 to 7.480 pp; 40% budget: 6.506 to 6.267 pp; 60% budget: 6.321 to 6.113 pp. Tuned MAE is lower at 3/3 budgets versus both reference-only audit and fixed-power PPI.

**Target:** MAE against each model-pair/split realized held-out plurality A-score; categorical A/tie/B scores 1/.5/0.

[Complete source report](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/research/REPORT.md). **Selectors and metrics:**

- [reports/research/summary.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/research/summary.csv): `{}`; 12 rows; metric columns and exact values in `E01` of the JSON.
- [reports/research/per_pair.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/research/per_pair.csv): `{"method":{"in":["human_only","ppi_tuned"]}}`; 90 rows; metric columns and exact values in `E01` of the JSON.

**Dependence:** Same questions recur across model pairs, both turns, nested budgets and 30 splits. Sensitivity is a reanalysis of identical comparisons/splits, not independent replication. Historical reports/mtbench has a different complementary-pool design and must not be pooled with fixed-target results.

## E02: Reference definition changes 305 comparison scores and 5 of 45 pair/budget method-gap signs

305/1,814 comparison scores change; mean absolute score change 3.852 pp. Pair-level tuned-minus-audit MAE changes sign in 5/45 cells; neither reference is thereby superior.

**Target:** Separate realized held-out means of human plurality scores versus equally comparison-weighted mean human vote scores; neither is a pooled-vote target or latent truth.

[Complete source report](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/sensitivity/REPORT.md). **Selectors and metrics:**

- [reports/sensitivity/reference_changes.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/sensitivity/reference_changes.csv): `{}`; 1814 rows; metric columns and exact values in `E02` of the JSON.
- [reports/sensitivity/paired_method_gaps.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/sensitivity/paired_method_gaps.csv): `{}`; 90 rows; metric columns and exact values in `E02` of the JSON.
- [reports/sensitivity/summary.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/sensitivity/summary.csv): `{"method":{"in":["human_only","ppi_tuned"]}}`; 12 rows; metric columns and exact values in `E02` of the JSON.

**Dependence:** Same questions recur across model pairs, both turns, nested budgets and 30 splits. Sensitivity is a reanalysis of identical comparisons/splits, not independent replication. Historical reports/mtbench has a different complementary-pool design and must not be pooled with fixed-target results.

## E03: Near-boundary lower point RMSE coexists with 52.4% normal coverage

At p=.95,n=20,N=200: tuned normal RMSE 4.128 pp, coverage 52.4% (pointwise Monte Carlo interval 49.25–55.53%), with 353/1,000 zero-width intervals. Point accuracy does not validate coverage.

**Target:** Population binary outcome mean p=.95; IID frozen proxy sensitivity .95/specificity .90; nominal 95% population interval.

[Complete source report](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/finite-sample/REPORT.md). **Selectors and metrics:**

- [reports/finite-sample/summary.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/finite-sample/summary.csv): `{"method":{"in":["human_normal","ppi_tuned_normal","human_hoeffding","ppi_eb_grid"]},"scenario":{"eq":"iid_rare_strong_n0020"}}`; 4 rows; metric columns and exact values in `E03` of the JSON.

**Dependence:** Synthetic data, not corpus observations. Shared draws across 8 methods within each scenario. The 12 IID cells and 2 intentional assumption-violation cells; only specified IID cells support finite-theorem applicability. The same mechanism family recurs in earlier/later studies with distinct experiment seeds; do not pool as new domains.

## E04: Conservative finite intervals cover frequently but grid EB beats human Hoeffding width in only 1 of 12 IID settings

Observed coverage is 99.3–100.0% across 60 finite-theorem-applicable method/scenario cells. Grid EB has smaller untruncated mean width than human Hoeffding in 1/12 IID settings.

**Target:** Known population binary mean in 12 prespecified IID settings; interval raw untruncated width, not clipped domain width or deployment label efficiency.

[Complete source report](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/finite-sample/REPORT.md). **Selectors and metrics:**

- [reports/finite-sample/summary.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/finite-sample/summary.csv): `{"theorem_applicable":{"eq":true}}`; 60 rows; metric columns and exact values in `E04` of the JSON.
- [reports/finite-sample/summary.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/finite-sample/summary.csv): `{"method":{"in":["ppi_eb_grid","human_hoeffding","human_eb"]},"scenario_family":{"eq":"iid"}}`; 36 rows; metric columns and exact values in `E04` of the JSON.

**Dependence:** Synthetic data, not corpus observations. Shared draws across 8 methods within each scenario. The 12 IID cells and 2 intentional assumption-violation cells; only specified IID cells support finite-theorem applicability. The same mechanism family recurs in earlier/later studies with distinct experiment seeds; do not pool as new domains.

## E05: Signed tuning exploits inverse signal at a small audit budget

At p=.5,n=20,N=200, positive-to-signed RMSE changes from 10.981 to 6.826 pp. Paired signed-minus-positive MSE is -0.007398 with MCSE 0.000439 across 1,000 shared-draw replications.

**Target:** Population binary outcome prevalence p=.5; paired comparison of population-variance tuning over [0,1] versus [-1,1].

[Complete source report](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/signed-power/REPORT.md). **Selectors and metrics:**

- [reports/signed-power/simulation_summary.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/signed-power/simulation_summary.csv): `{"method":{"in":["human_only","ppi_tuned","ppi_signed"]},"scenario":{"eq":"iid_p050_inverse_n0020"}}`; 3 rows; metric columns and exact values in `E05` of the JSON.
- [reports/signed-power/llmbar_summary.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/signed-power/llmbar_summary.csv): `{"method":{"in":["ppi_tuned","ppi_signed"]}}`; 54 rows; metric columns and exact values in `E05` of the JSON.

**Dependence:** Primary contrast uses one of 18 synthetic settings with shared method draws. The supplementary 11/27 LLMBar comparison repeats the historical panel and splits; it is exploratory, not external replication. No synthetic and empirical losses are pooled.

## E06: Allowing negative powers can increase observed error when the proxy carries no signal

At p=.5,n=20,N=200, positive-to-signed RMSE changes from 11.077 to 11.247 pp. Paired signed-minus-positive MSE is +0.000381 with MCSE 0.000122 across 1,000 shared-draw replications.

**Target:** Population binary outcome prevalence p=.5; paired comparison of population-variance tuning over [0,1] versus [-1,1].

[Complete source report](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/signed-power/REPORT.md). **Selectors and metrics:**

- [reports/signed-power/simulation_summary.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/signed-power/simulation_summary.csv): `{"method":{"in":["human_only","ppi_tuned","ppi_signed"]},"scenario":{"eq":"iid_p050_uninformative_n0020"}}`; 3 rows; metric columns and exact values in `E06` of the JSON.

**Dependence:** One scenario from 18 synthetic settings; all 4 methods share each draw. Design motivated by earlier LLMBar negative association; empirical LLMBar extension is repeated evidence, while this finding uses synthetic outcomes only. Mechanism family overlaps other simulations; no cross-study pooling.

## E07: The same switch worsens population RMSE while improving realized-pool RMSE

At p=.5,n=200,N=100 with a positive proxy: population RMSE 3.094 to 4.619 pp; realized-pool RMSE 4.718 to 3.228 pp. These are two within-target comparisons, with opposite directions.

**Target:** Two explicitly separate targets: analytic population prevalence p=.5, and each draw's realized held-out binary mean. Compare methods within each target, never compare unlike-target RMSE as a superiority metric.

[Complete source report](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/estimand/REPORT.md). **Selectors and metrics:**

- [reports/estimand/simulation_summary.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/estimand/simulation_summary.csv): `{"method":{"in":["population_tuned","pool_tuned"]},"scenario":{"eq":"iid_p050_positive_n0200_r005"}}`; 2 rows; metric columns and exact values in `E07` of the JSON.
- [reports/estimand/simulation_summary.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/estimand/simulation_summary.csv): `{"method":{"in":["population_tuned","pool_tuned"]},"scenario":{"eq":"iid_p095_positive_n0020_r005"}}`; 2 rows; metric columns and exact values in `E07` of the JSON.

**Dependence:** All 4 point rules and both targets use identical draws; target scores are correlated, not independent experiments. The 36 settings reuse earlier proxy-law families at additional pool ratios. Empirical LLMBar extension elsewhere repeats the original corpus and is not included in this synthetic finding.

## E08: ChatGPT agrees across orders on 176 reference-wrong adversarial comparisons

ChatGPT on 319 Adversarial comparisons: original-order reference accuracy 28.2%, agreement 64.3%; 176/205 valid agreements choose the reference-incorrect response.

**Target:** Historical ChatGPT original-order canonical choice correctness against curated LLMBar instruction-following reference; valid two-order agreement is a separate gold-free proxy.

[Complete source report](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/llmbar-audit/REPORT.md). **Selectors and metrics:**

- [reports/llmbar-audit/corpus_diagnostics.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/llmbar-audit/corpus_diagnostics.csv): `{"cohort":{"eq":"Adversarial"},"judge":{"eq":"ChatGPT"}}`; 1 rows; metric columns and exact values in `E08` of the JSON.
- [reports/llmbar-audit/summary.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/llmbar-audit/summary.csv): `{"cohort":{"eq":"Adversarial"},"judge":{"in":["ChatGPT","LLaMA2"]}}`; 24 rows; metric columns and exact values in `E08` of the JSON.

**Dependence:** Signed and estimand empirical extensions reuse the same LLMBar panel and historical splits. RewardBench includes the same 419-comparison source component with different cached judges. Cohorts, methods and repeated splits are not independent evidence.

## E09: One shared RewardBench agreement signal corresponds to two different reference accuracies

On 2,566 NonLLMBar comparisons, agreement is 89.95% for both directions; target accuracies are 90.80% and 86.67%. Both judges agree incorrectly on 160 comparisons.

**Target:** Equal-comparison-weighted correctness relative to heterogeneous RewardBench references, versus cross-model canonical-choice agreement; not official weighted leaderboard score.

[Complete source report](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/rewardbench-audit/REPORT.md). **Selectors and metrics:**

- [reports/rewardbench-audit/corpus_diagnostics.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/rewardbench-audit/corpus_diagnostics.csv): `{"cohort":{"eq":"NonLLMBar"}}`; 2 rows; metric columns and exact values in `E09` of the JSON.

**Dependence:** NonLLMBar excludes 419 known LLMBar comparisons and shares no exact prompt with that component; All overlaps both. Remaining sources still include MT-Bench-derived comparisons, so NonLLMBar is not certified as wholly independent benchmark lineage. Both target directions use the same two judgments/reference labels; 30 splits and budgets overlap.

## E10: Tuning improves primary audit-baseline MAE descriptively, but not every alternative or cell

Lower-MAE cells versus reference-only audit: ppi 1/6, ppi_tuned 6/6, ppi_signed 6/6, audit_residual 5/6. Adverse examples remain: signed-minus-raw MAE +0.237 pp for GPT-4o at 20%; residual-minus-audit MAE +0.025 pp for mini at 40%. For GPT-4o at 40%, residual tuning changes MAE 1.179 to 1.115 pp, but RMSE 1.421 to 1.432 pp.

**Target:** Per-target error against realized held-out strict reference accuracy; MAE averages 30 dependent fixed-corpus splits within each target/budget cell.

[Complete source report](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/rewardbench-audit/REPORT.md). **Selectors and metrics:**

- [reports/rewardbench-audit/summary.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/rewardbench-audit/summary.csv): `{"cohort":{"eq":"NonLLMBar"}}`; 36 rows; metric columns and exact values in `E10` of the JSON.

**Dependence:** NonLLMBar excludes 419 known LLMBar comparisons and shares no exact prompt with that component; All overlaps both. Remaining sources still include MT-Bench-derived comparisons, so NonLLMBar is not certified as wholly independent benchmark lineage. Both target directions use the same two judgments/reference labels; 30 splits and budgets overlap.

## E11: Grouping and transport are separate requirements

On 500 shared draws with 8 audit clusters repeated 4 times, tuned coverage is 89.0% with grouped inference and 59.6% with naive row independence. Prevalence-shift tuned bias/coverage are -0.1975/0.0%; conditional-error-shift values are -0.0545/23.9% across 1,000 replications per shift law. These scenarios are not pooled.

**Target:** Population outcome mean under an explicit synthetic law; intended-target coverage under correct clusters versus deliberate row-IID misuse, or under deliberate audit-to-target shift.

[Complete source report](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/stress/REPORT.md). **Selectors and metrics:**

- [reports/stress/summary.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/stress/summary.csv): `{"method":{"eq":"ppi_tuned"},"scenario":{"in":["cluster_g008_correct","cluster_g008_naive_iid"]}}`; 2 rows; metric columns and exact values in `E11` of the JSON.
- [reports/research/simulation_summary.csv](https://github.com/Siquan-Wang/llm-judge-calibration/blob/b4bdeab08481a25b529d81fa909f96aab10f368f/reports/research/simulation_summary.csv): `{"method":{"eq":"ppi_tuned"},"scenario":{"in":["prevalence_shift","conditional_error_shift"]}}`; 2 rows; metric columns and exact values in `E11` of the JSON.

**Dependence:** Grouped and naive variants share identical generated draws, but variance/tuning calculations differ. Shift scenarios are distinct declared laws, not empirical corpora or independent domain replications.

## Verify or rebuild

```bash
python examples/research_evidence.py --check
python examples/research_evidence.py --output-directory docs
```

`--check` verifies inputs and compares both generated files without writing. Any CSV, cited report/configuration, report-manifest or lineage-input byte mismatch fails; changing a self-declared sidecar hash cannot bypass the independent pins. Rebuilds are offline and deterministic.

Historical report source-code fingerprints describe their original runs, not the current evolving source tree. Immutable source-report links refer to commit `b4bdeab08481a25b529d81fa909f96aab10f368f`.
