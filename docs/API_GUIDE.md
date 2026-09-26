# API guide

This guide preserves the detailed usage examples behind the research overview.
Choose the outcome, target and sampling unit before selecting an interval.

| Target or question | API | Scope |
|---|---|---|
| Population bounded-outcome mean | `prediction_powered_mean` | Plug-in normal interval; optional one-way clusters; explicit coefficient bounds. |
| Population categorical preference rate | `prediction_powered_win_rate` | Fixed tie score; default nonnegative tuning; same sampling assumptions. |
| Random held-out mean under iid sampling | `predict_heldout_mean` | Marginal prediction interval over both pools; no cluster arguments or conditional fixed-pool guarantee. |
| Grouped audit-residual candidate | `audit_residual_mean` | Point estimate and audit criterion only; no prediction interval. |
| Population bounded mean with conservative finite-sample inference | `finite_sample_mean` | Independent iid pools; fixed power or prespecified finite grid; no continuous automatic tuning or groups. |
| Full fixed-corpus row mean | `finite_corpus_mean` | Uniform fixed-size sampling of complete groups without replacement; known frame and proxy totals; HS/EBS bounds, fixed coefficient or finite grid. |

The [technical report](TECHNICAL_REPORT.md) explains why these targets differ.
For full formulas, attribution and assumptions, see [Methods](METHODS.md).
The small examples below illustrate signatures, not adequate audit sizes.

## Full fixed-corpus mean

```python
from judgecal import finite_corpus_mean

result = finite_corpus_mean(
    group_sizes=[1, 2, 1, 3],
    group_prediction_totals=[0.2, 1.4, 0.9, 2.1],
    audited_group_indices=[0, 3],
    audited_outcome_totals=[1, 2],
    power=(0, 0.25, 0.5, 0.75, 1),
    method="empirical_bernstein_serfling",
)
print(result.point, result.interval, result.audited_rows)
```

Here the supplied indices must be one realized uniform sample of exactly two
complete groups from the four-group frame. Arrays cannot verify randomization.
The target is all seven rows' outcome mean, with known group membership, sizes
and full proxy totals. Only the two audited groups' outcome totals are supplied.
Each row outcome and proxy lies in [0,1], so group totals lie in [0,group_size].

The number of audited rows varies with the sampled groups. Do not stop at a row
budget or pass partial groups while claiming the same coverage guarantee.
There is no independent prediction pool, group-outcome independence assumption,
population target or held-out target. Fixed coefficients are design-unbiased;
the finite-grid selection preserves simultaneous interval coverage within that
declared family but can bias the selected point. Select the grid and bound
family before seeing audit outcomes. A full census returns the exact mean and
zero width. [Methods and source attribution](METHODS.md) and the
[study protocol](FINITE_CORPUS_PROTOCOL.md) state the remaining conditions.

## Prediction-powered win rates


Let `Y_L` be human preference scores on an audit, `F_L` the matching judge scores
and `F_U` judge scores on a separate target pool. A target win scores 1, a loss 0
and a tie 0.5. The power-weighted mean estimator is:

```text
human preference rate = mean(Y_L) + power * (mean(F_U) - mean(F_L))
SE² = Var(mean(Y_L - power * F_L)) + power² * Var(mean(F_U))

power=0       human audit only
power=1       original PPI (backward-compatible default)
power="auto"  minimize the estimated per-pool variance over [0, 1]
```

```python
from judgecal import prediction_powered_win_rate

result = prediction_powered_win_rate(
    judge_labeled=["A", "A", "B", "tie", "B", "A"],
    human_labeled=["A", "B", "B", "tie", "A", "A"],
    judge_unlabeled=["A", "B", "A", "tie", "A", "B", "B", "A"],
    labeled_groups=[1, 1, 2, 2, 3, 3],
    unlabeled_groups=[4, 4, 5, 5, 6, 6, 7, 7],
    power="auto",
)
print(result.point, result.interval.low, result.interval.high)
print(result.selected_power, result.estimated_variance_ratio)
print(result.as_dict())
```

This tiny input illustrates the API, not adequate sample size for valid normal inference. Group IDs enable question-cluster variance and reject overlapping audit/evaluation questions. Unequal clusters retain an observation-weighted estimand. Without groups, rows are treated as independent and the caller must ensure disjoint pools.

Both pools must represent the same target population and use the same frozen
judge. Normal intervals need sufficiently many independent observations or
clusters. Estimates and intervals are not clipped to [0, 1].

The method follows [PPI](https://arxiv.org/abs/2301.09633) and the
[PPI++ mean-estimation framework](https://arxiv.org/abs/2311.01453).
This implementation minimizes a **per-pool sample/sandwich variance**, rather
than the pooled prediction variance in the authors' `ppi_py` software. Its
optional one-way cluster adaptation is documented explicitly. The fitted scalar
power can use audit labels, while the underlying judge must remain frozen.
Estimated-variance reduction is not a finite-sample MSE or coverage guarantee.
See [Methods](METHODS.md) for equations, attribution and assumptions.

For bounded numeric predictions or fractional human outcomes, use the
same inference through `prediction_powered_mean`:

```python
from judgecal import prediction_powered_mean

result = prediction_powered_mean(
    predictions_labeled=[0.2, 0.8, 0.6, 0.1],
    outcomes_labeled=[1/3, 1.0, 2/3, 0.0],
    predictions_unlabeled=[0.3, 0.7, 0.5, 0.9, 0.2],
    power="auto",
)
print(result.point, result.prediction_mean, result.outcome_mean)
```

This is another small API illustration. Inputs must be finite numeric
scores in `[0,1]`; fractional values are preserved. The result estimates
the population mean of the explicitly defined outcome, with the same
sampling assumptions and untruncated normal intervals as above.

To study inverse proxy signals, pass `power="auto", power_bounds=(-1,1)`
to this numeric API. The result records the chosen bounds and coefficient.
Choose the range before examining evaluation outcomes; same-audit scalar
tuning is asymptotic and can introduce finite-sample bias. The categorical
win-rate and finite-sample APIs keep their existing coefficient behavior.

## Other tools

| Question | API | Interpretation |
|---|---|---|
| Does the judge agree with humans? | `agreement_rate`, `agreement_with_ci`, `cohens_kappa` | Explicit tie handling; Bayesian intervals assume independent agreement observations. |
| What is the raw model win rate? | `win_rate_with_ci`, `compare_models` | Ties count as half a win; raw uncertainty is labeled separately. |
| Can binary judge error be corrected? | `judge_confusion`, `rogan_gladen_correction`, `compare_models` | Rogan–Gladen requires a binary estimand and transferable sensitivity/specificity. |
| How uncertain is the correction? | `compare_models(..., calibration_judge=..., calibration_human=...)` | Resamples calibration pairs and independent evaluation judgments; reports invalid draws. |
| Are numeric judge scores calibrated? | `platt_scaling`, `isotonic_calibration` | Fit on calibration data, evaluate separately; duplicate isotonic scores are pooled. |
| Which judge agrees more with humans? | `paired_agreement_gap` | The paired difference has its own bootstrap interval. |

Rogan–Gladen uses `p_true = (p_observed + specificity - 1) / (sensitivity + specificity - 1)`. Plugging in estimated error rates and clipping is not generally unbiased. The correction does not repair selection bias or arbitrary distribution shift. Binary correction rejects ties rather than silently changing its estimand. Summary confusion statistics alone do not supply calibration-uncertainty intervals.

Inter-annotator agreement is a reference, not a universal ceiling. Overlapping confidence intervals do not establish equivalence. Repeated annotations or comparisons sharing a question are not independent trials.
