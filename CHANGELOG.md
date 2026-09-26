# Changelog

## 0.3.0

### Research methods and experiments

- Bounded power tuning for the PPI mean estimator, attributed to PPI++.
  `power=0` gives human-only inference, `power=1` preserves the existing
  estimator, and `power="auto"` minimizes estimated per-pool variance.
  Results expose the selected coefficient, tuning method and estimated
  variance ratio. One-way cluster adaptation and asymptotic limits are explicit.
- A fixed-evaluation MT-Bench study with nested audit budgets, all four
  baselines, complete split manifests and annotation-workload counts.
  Historical complementary-pool results remain separate and unchanged.
- Seven known-truth simulation scenarios covering weak and anticorrelated
  judges, dependence, few clusters and two separate distribution-shift mechanisms.
  Every replication, failure of coverage, Monte Carlo error and zero-width
  interval is retained; native interval targets are distinct from target-human truth.
- Versioned analysis plan, mathematical methods, code/artifact checksums and
  offline reproduction, plus citation metadata. This is an empirical research
  extension of established methods, not a new estimator claim.

### Correctness

- Stable upper-tail calculations for Wilson and Beta intervals at extreme
  confidence levels; stable Hoeffding log calculation at very small alpha.
- Research output directories reject stale or unrelated artifacts and protect
  the original input study; tampered label provenance fails before analysis.

## 0.2.0

### Added

- Original fixed-weight prediction-powered human-preference inference, with
  optional one-way question-cluster variance and disjoint-pool checks.
- A reproducible MT-Bench human-label-budget benchmark: all 15 model pairs,
  three budgets, 30 splits per budget, full trial/split records, plots and
  three known-truth simulations. Public data provenance and its separate
  CC-BY-4.0 license are recorded.
- Joint resampling of independent calibration pairs and evaluation labels
  for binary Rogan–Gladen correction, with failure diagnostics.
- A paired agreement-gap confidence interval and offline benchmark CI checks.

### Corrected behavior

- Legacy `compare_models` fields describe the raw judge result. Explicit
  `raw_*` aliases and separate `corrected_ci` / `corrected_significant`
  fields prevent a raw interval from being reported as corrected inference.
- Rogan–Gladen correction rejects ties and summaries fitted after dropping
  ties. Use tie-inclusive PPI for the half-win human-preference estimand.
- `target="B"` correctly reverses sensitivity/specificity orientation.
- Nonfinite rates, malformed inputs and failed Platt optimization no longer
  silently produce misleading results. Isotonic calibration pools equal scores.
- Raw binary win-rate intervals use Clopper–Pearson; tie-containing samples
  use percentile bootstrap with a conservative constant-sample fallback.
- The MT-Bench loader pins its source, validates response alignment and
  deduplicates identical votes from the same annotator. Its aggregate is a
  plurality label, with tied vote counts mapped to `tie`.
- Documentation removes general unbiasedness, human-noise-ceiling and
  interval-overlap-equivalence claims. Unstable corrected intervals are
  unavailable, with an explicit diagnostic, rather than silently filtered.

## 0.1.0

Initial agreement, score calibration, measurement-error correction and
MT-Bench demonstration APIs.
