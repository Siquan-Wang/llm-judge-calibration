# Follow-up factorial and dependence study

This retrospective follow-up broadens the first reliability study; it is a specified stress grid, not a representative sample of real deployments.

## Factorial settings

Actual run: 500 replications per setting; master seed 2027.

Human prevalence 0.2/0.5/0.8; audit size 20/50/200; target pool ten times the audit size; four strong/moderate/uninformative/anticorrelated judges. All four estimators receive identical draws. Every cell and replication is retained.

Tuned PPI has lower observed RMSE than human-only in 19/36 iid cells. This count is descriptive of this chosen grid, not a probability of improvement in deployment. Paired Monte Carlo errors and per-cell results are retained in `summary.csv`.

| Scenario | Human RMSE | PPI RMSE | Tuned RMSE | Tuned-human RMSE difference | Paired MC SE | Tuned coverage |
|---|---:|---:|---:|---:|---:|---:|
| iid_p020_anti_n020 | 0.0842 | 0.1619 | 0.0843 | 0.0001 | 0.0001 | 0.926 |
| iid_p020_anti_n050 | 0.0577 | 0.1134 | 0.0577 | 0.0000 | 0.0000 | 0.936 |
| iid_p020_anti_n200 | 0.0281 | 0.0559 | 0.0281 | 0.0000 | 0.0000 | 0.944 |
| iid_p020_moderate_n020 | 0.0874 | 0.1201 | 0.0824 | -0.0050 | 0.0018 | 0.918 |
| iid_p020_moderate_n050 | 0.0612 | 0.0746 | 0.0554 | -0.0058 | 0.0010 | 0.912 |
| iid_p020_moderate_n200 | 0.0292 | 0.0348 | 0.0267 | -0.0026 | 0.0004 | 0.946 |
| iid_p020_strong_n020 | 0.0942 | 0.0740 | 0.0655 | -0.0287 | 0.0027 | 0.868 |
| iid_p020_strong_n050 | 0.0585 | 0.0444 | 0.0400 | -0.0186 | 0.0015 | 0.914 |
| iid_p020_strong_n200 | 0.0283 | 0.0225 | 0.0193 | -0.0090 | 0.0008 | 0.952 |
| iid_p020_uninformative_n020 | 0.0796 | 0.1445 | 0.0809 | 0.0013 | 0.0005 | 0.924 |
| iid_p020_uninformative_n050 | 0.0574 | 0.0902 | 0.0578 | 0.0004 | 0.0003 | 0.932 |
| iid_p020_uninformative_n200 | 0.0279 | 0.0459 | 0.0279 | 0.0001 | 0.0001 | 0.958 |
| iid_p050_anti_n020 | 0.1082 | 0.1941 | 0.1082 | 0.0000 | 0.0000 | 0.966 |
| iid_p050_anti_n050 | 0.0712 | 0.1292 | 0.0712 | 0.0000 | 0.0000 | 0.928 |
| iid_p050_anti_n200 | 0.0359 | 0.0643 | 0.0359 | 0.0000 | 0.0000 | 0.930 |
| iid_p050_moderate_n020 | 0.1083 | 0.1089 | 0.0962 | -0.0121 | 0.0021 | 0.932 |
| iid_p050_moderate_n050 | 0.0695 | 0.0676 | 0.0603 | -0.0093 | 0.0014 | 0.942 |
| iid_p050_moderate_n200 | 0.0359 | 0.0378 | 0.0325 | -0.0035 | 0.0008 | 0.936 |
| iid_p050_strong_n020 | 0.1077 | 0.0683 | 0.0625 | -0.0451 | 0.0034 | 0.942 |
| iid_p050_strong_n050 | 0.0727 | 0.0430 | 0.0409 | -0.0318 | 0.0023 | 0.960 |
| iid_p050_strong_n200 | 0.0353 | 0.0222 | 0.0207 | -0.0146 | 0.0010 | 0.942 |
| iid_p050_uninformative_n020 | 0.1128 | 0.1639 | 0.1140 | 0.0012 | 0.0008 | 0.956 |
| iid_p050_uninformative_n050 | 0.0702 | 0.0970 | 0.0702 | -0.0000 | 0.0003 | 0.950 |
| iid_p050_uninformative_n200 | 0.0346 | 0.0481 | 0.0347 | 0.0001 | 0.0001 | 0.956 |
| iid_p080_anti_n020 | 0.0950 | 0.1757 | 0.0953 | 0.0003 | 0.0001 | 0.918 |
| iid_p080_anti_n050 | 0.0573 | 0.1074 | 0.0573 | 0.0000 | 0.0000 | 0.932 |
| iid_p080_anti_n200 | 0.0283 | 0.0533 | 0.0283 | 0.0000 | 0.0000 | 0.942 |
| iid_p080_moderate_n020 | 0.0877 | 0.0953 | 0.0797 | -0.0080 | 0.0020 | 0.912 |
| iid_p080_moderate_n050 | 0.0566 | 0.0608 | 0.0517 | -0.0049 | 0.0012 | 0.920 |
| iid_p080_moderate_n200 | 0.0293 | 0.0294 | 0.0255 | -0.0038 | 0.0005 | 0.940 |
| iid_p080_strong_n020 | 0.0916 | 0.0601 | 0.0608 | -0.0308 | 0.0027 | 0.882 |
| iid_p080_strong_n050 | 0.0559 | 0.0398 | 0.0366 | -0.0193 | 0.0016 | 0.928 |
| iid_p080_strong_n200 | 0.0280 | 0.0178 | 0.0161 | -0.0119 | 0.0008 | 0.958 |
| iid_p080_uninformative_n020 | 0.0896 | 0.1498 | 0.0901 | 0.0004 | 0.0006 | 0.916 |
| iid_p080_uninformative_n050 | 0.0565 | 0.0941 | 0.0567 | 0.0002 | 0.0002 | 0.938 |
| iid_p080_uninformative_n200 | 0.0302 | 0.0459 | 0.0303 | 0.0000 | 0.0001 | 0.918 |

![Factorial results](factorial_grid.svg)

## Dependence ablation

Four exact repeated turns per independent question. The correct and naive analyses share every generated label. Treating rows as independent is deliberately misspecified. Few clusters can impair even correctly grouped asymptotic intervals.

| Questions | Variance assumption | Method | Coverage | MC SE | Mean width |
|---:|---|---|---:|---:|---:|
| 8 | cluster | Raw judge | 0.936 | 0.011 | 0.2138 |
| 8 | cluster | Human audit | 0.824 | 0.017 | 0.6592 |
| 8 | cluster | PPI (power 1) | 0.958 | 0.009 | 0.3742 |
| 8 | cluster | Tuned PPI | 0.890 | 0.014 | 0.3265 |
| 8 | naive_iid | Raw judge | 0.706 | 0.020 | 0.1064 |
| 8 | naive_iid | Human audit | 0.714 | 0.020 | 0.3132 |
| 8 | naive_iid | PPI (power 1) | 0.630 | 0.022 | 0.1813 |
| 8 | naive_iid | Tuned PPI | 0.596 | 0.022 | 0.1578 |
| 30 | cluster | Raw judge | 0.944 | 0.010 | 0.1104 |
| 30 | cluster | Human audit | 0.922 | 0.012 | 0.3499 |
| 30 | cluster | PPI (power 1) | 0.936 | 0.011 | 0.2118 |
| 30 | cluster | Tuned PPI | 0.928 | 0.012 | 0.1954 |
| 30 | naive_iid | Raw judge | 0.670 | 0.021 | 0.0551 |
| 30 | naive_iid | Human audit | 0.638 | 0.021 | 0.1727 |
| 30 | naive_iid | PPI (power 1) | 0.634 | 0.022 | 0.1049 |
| 30 | naive_iid | Tuned PPI | 0.632 | 0.022 | 0.0967 |
| 100 | cluster | Raw judge | 0.900 | 0.013 | 0.0604 |
| 100 | cluster | Human audit | 0.960 | 0.009 | 0.1918 |
| 100 | cluster | PPI (power 1) | 0.942 | 0.010 | 0.1190 |
| 100 | cluster | Tuned PPI | 0.938 | 0.011 | 0.1103 |
| 100 | naive_iid | Raw judge | 0.558 | 0.022 | 0.0302 |
| 100 | naive_iid | Human audit | 0.636 | 0.022 | 0.0955 |
| 100 | naive_iid | PPI (power 1) | 0.664 | 0.021 | 0.0593 |
| 100 | naive_iid | Tuned PPI | 0.654 | 0.021 | 0.0550 |

![Dependence ablation](dependence_ablation.svg)

## Interpretation

- Tuning minimizes fitted variance, not finite-sample risk. Small audit samples and near-boundary human rates remain difficult cases.
- The bounded coefficient discards negative covariance; it does not invert anticorrelated judges.
- Oracle powers are generator diagnostics only. No oracle enters the fitted-method results.
- Method differences are paired across identical replications. RMSE-difference MCSE uses a delta approximation; it is not a guarantee or a multiple-comparison-adjusted claim.
- Raw judge intervals target their own positive rate. Their human-truth containment is a bias diagnostic.
- Simulation cells do not establish accuracy or annotation savings on real LLM evaluation data.

## Reproduce

```bash
pip install -e ".[dev,report]"
python examples/stress_grid.py --output reports/reproduced/stress --plot
```

`config.json` records repetitions, seed and every generating mechanism. `trials.csv.gz` is deterministic gzip containing every trial; `reproducibility.json` binds source and artifact hashes. Defaults use 500 replications and master seed 2027. A smaller `--repetitions` value is an explicitly recorded smoke run.
