# Changelog

## 0.11.0

- An independent-pilot simulation compares eight established procedures at the
  same total label budget. Twelve configured law/budget cells retain all pilot
  costs, degenerate draws, paired losses and normal/finite-bound diagnostics.
- Addressable random streams preserve earlier draws when replication counts
  grow. The protocol distinguishes conditional unbiasedness from normal
  coverage and discloses the coefficient-formula differences between rules.

- A retrospective technical report with three figures, an offline PDF builder,
  and eleven machine-readable claim-to-evidence records. The synthesis preserves
  target distinctions, overlapping data, negative results and method attribution;
  it creates no new experiment or estimator.
- A concise research README and a separate API guide preserving the detailed
  examples. CI verifies evidence pins, PDF content, links and repeatability.

## 0.10.0

- Reproducible RewardBench cross-judge accuracy audits from two dated caches:
  all 2,985 comparisons, exact prompt grouping, both target directions and six
  estimators. A non-LLMBar primary component separates the verified overlap.
- Gold-independent agreement extraction, complete shared split records,
  explicit source terms and lossless indexed manifest encoding.

## 0.9.0

- Separate population-mean and random held-out-mean estimands, an iid marginal
  pool prediction interval, and grouped audit-residual point estimation.
- Shared-draw simulations and a repeated LLMBar analysis preserve target-specific
  errors and limitations of unequal-group fixed-corpus inference.

## 0.8.0

- Explicit signed coefficient bounds for numeric inference and a study of
  inverse proxies, fitted noise and repeated LLMBar outcomes. Negative powers
  are attributed to the existing PPI++ framework.

## 0.7.0

- Cached LLMBar judge-correctness audits retain all 419 comparisons and invalid
  judgments. Order agreement is a proxy, not a reference label; failures and
  exact-instruction split identities remain in the report.

## 0.6.0

- Fixed-coefficient and finite-grid Hoeffding/empirical-Bernstein intervals for
  bounded iid means, with an explicit simultaneous error allocation.
- A finite-sample study reports coverage and untruncated width separately,
  including deliberate assumption violations and degenerate normal intervals.

## 0.5.0

- Bounded numeric outcome means and MT-Bench reference sensitivity comparing
  plurality with mean recorded votes. Changed targets are not treated as
  interchangeable truth, and annotator uncertainty remains explicit.

## 0.4.0

### Research follow-up

- A 36-cell prevalence/judge-quality/audit-size grid and paired cluster-versus-
  naive-independence ablation, with complete deterministic gzip trial records,
  paired loss-difference Monte Carlo errors and diagnostic oracle coefficients.
- Optional, pinned numerical comparisons to `ppi-python==0.2.3`, distinguishing
  fixed-weight algebra from intentional finite-sample variance/tuning differences.
- Explicit reports of small-sample undercoverage and cases where tuning fails
  to improve error, alongside the favorable scenarios.

### Calibration correctness

- Platt scaling now optimizes and predicts in centered/scaled coordinates while
  preserving the original raw-slope L2 objective. Large score offsets no longer
  trigger optimizer failure or silently constant predictions. Public `params_`
  retain original score units. Regression tests include independent objective
  verification, translation invariance and finite extreme-score handling.

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
