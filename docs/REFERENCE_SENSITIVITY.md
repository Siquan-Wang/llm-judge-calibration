# Human-reference sensitivity: retrospective follow-up protocol

This follow-up asks whether conclusions about raw judging, human-only
estimation, fixed-power PPI, and tuned PPI depend on how the recorded human
votes are summarized. It compares two legitimate but distinct observed
reference definitions on identical data partitions. It does not identify
which definition is latent truth or claim that the lower-error definition
is intrinsically better.

The original data and earlier experimental results were available when
this analysis was designed. This is a versioned retrospective sensitivity
analysis, not a preregistered experiment on unseen data. Numerical PPI and
power tuning remain established methods; accepting fractional outcomes is
an implementation extension, not a new statistical estimator.

## Verified source and vote construction

The source is the already committed, attributed public snapshot at
`reports/mtbench/labels.csv`, with its checksum and transformation record in
`reports/mtbench/dataset.json`. It derives from
[`lmsys/mt_bench_human_judgments`, revision f7d2896d2cc5d80f8b55c2bbc722613555233c25](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/tree/f7d2896d2cc5d80f8b55c2bbc722613555233c25).
The dataset concerns the older MT-Bench responses described by
[Zheng et al.](https://arxiv.org/abs/2306.05685); no new judgments or model
responses are collected in this analysis.

The loader's canonical comparison key is `(question_id, turn, model_a,
model_b)`, after alphabetically ordering the two models and reorienting
their winners. Human votes are unique within a comparison and annotator ID.
Identical duplicates from one annotator are removed; conflicting labels
from the same annotator and comparison are rejected. Both `author_*` and
`expert_*` annotators are included. These are the recorded annotators, not
a probability sample of all potential users or judges.

For each comparison the snapshot retains `n_human_a`, `n_human_b`,
`n_human_ties`, and their total `n_human_votes`. Upstream `tie` and
`tie (inconsistent)` map to tie before aggregation. The GPT-4 label remains
the same cached, combined judgment under both human-reference definitions.
Its source-order diagnostics do not reconstruct separate order-specific
votes or supply a position-bias estimator.

Source-first checks for this follow-up verified the existing SHA-256,
recomputed every count total, and reconstructed every plurality label:

- All 1,814 aligned comparisons have consistent vote counts, totaling
  3,354 unique recorded votes.
- The number of comparisons with 1 through 7 votes is respectively
  854, 560, 271, 92, 25, 10, and 2.
- Reconstructing the documented plurality rule exactly reproduces every
  stored `human_winner`.
- The two score definitions below differ on 305 comparisons in the full
  snapshot. These descriptive counts precede eligible-pair filtering and
  are not independent experimental replications.

The derived data remain CC BY 4.0; attribution and transformations are
documented in [DATA_LICENSE.md](../reports/mtbench/DATA_LICENSE.md). No
prompt, response, or individual annotator identity needs to be added to
the sensitivity artifacts.

## Two reference definitions and three weighting schemes

For comparison `i`, write its recorded vote counts as `a_i`, `b_i`, `t_i`
and `K_i = a_i + b_i + t_i`.

**Plurality score** (`reference_definition=plurality`): choose the unique most frequent
label among A, B, and tie. Any tie for the largest count becomes tie. Map
A, B, tie to 1, 0, 0.5. This is a plurality convention, not a strict
majority-vote requirement.

**Comparison-average vote score** (`reference_definition=mean_vote`):

```text
Y_i = (a_i + 0.5 * t_i) / K_i.
```

Each unique recorded annotator contributes one score inside its comparison.
The score is retained as a bounded numeric value; values such as 2/3 are
not rounded to A, B, or tie. Two A votes and one B vote give plurality
score 1 and vote score 2/3. Two A votes and two tie votes give plurality
score 0.5 and vote score 0.75. A plurality tie therefore need not imply
an exactly balanced average vote score. The mean does not retain the full
disagreement pattern: unanimous ties and equally split A/B votes both
have mean 0.5. Their original counts remain necessary to distinguish them.

Both references retain **equal comparison weighting**: for `M` retained
comparisons their mean is `sum_i Y_i / M`. A question with multiple retained
turns contributes multiple comparisons. Question-level grouping controls
the variance/split unit; it does not change the point estimate to an equally
weighted question mean.

The mean-vote definition also differs from **pooled-vote weighting**:

```text
equal comparison mean = mean_i[(a_i + 0.5*t_i) / K_i]
pooled vote mean      = sum_i(a_i + 0.5*t_i) / sum_i(K_i).
```

The latter weights comparisons by their number of recorded annotators and
is not the target of this follow-up. Counts must not be used to expand rows
into individual votes, as inferential weights, or as extra independent
observations. The study changes the outcome definition, not comparison
weights, split units, or annotation costs.

## Fixed partitions, estimators, and costs

Use `benchmark.reference_definition_sensitivity` with the same configuration
as the existing fixed-evaluation experiment: every canonical model pair
with at least 40 questions; seeds 0 through 29; a fixed evaluation pool of
`floor(0.25 * G)` questions per pair/seed; and nested audit banks containing
`floor(f * G)` questions for `f` in 0.2, 0.4, 0.6. Fractions refer to the
pair's total question count. Unused bank questions supply no labels or
predictions. All retained turns of a question stay together.

Reuse the existing plurality results and the same split IDs for mean-vote
outcomes. Partitions are pair-specific, as before; sharing questions across
pairs does not produce independent replicates. Record the explicit audit,
evaluation, and unused question IDs. Held-out human references are specific
to `reference_definition` and must be stored alongside that definition rather than treated
as common split metadata.

For both definitions, judge scores remain 1, 0, or 0.5. Human-only uses the
audit outcome mean. The corrected methods estimate
`mean(Y_L) + lambda * (mean(F_U) - mean(F_L))`, with fixed `lambda=1` or
audit-selected bounded power. The generic numeric mean API handles the
fractional outcomes. Grouped covariance/variance is computed at question
level under the same convention as the existing study.

Each definition is fitted using only its audit human outcomes. Its own
evaluation outcomes enter post-inference scoring. No evaluation human
labels choose coefficients, successful seeds, or reported model pairs.
Both configured definitions are reported regardless of the resulting
method ranking. Deterministic validation of labels and vote totals may
precede splitting; it is not coefficient fitting.

Changing the reference can legitimately change the audit covariance,
selected power, correction, and interval width. The raw judge estimate,
split IDs, audit/evaluation sizes, comparison counts, and recorded human
vote costs must remain identical for matched trials. Both definitions use
the same available votes: computing a mean does not create new annotations
or save annotation work. Raw judging uses zero audit labels. Vote counts
remain workload proxies, not paid annotation time or dollar costs.

## Observed reference uncertainty and interpretation

The held-out scoring target is the realized mean of each reference's
recorded comparison scores. Neither is a known population truth. With
few selected annotators, the plurality can change when another vote is
added, and the observed mean-vote score need not equal a comparison's
underlying preference probability.

The PPI variance uses variation in the observed audit scores and their
question-level clusters. It does not fit a latent-label model, annotator
random effects, or a posterior for each comparison's preference. It does
not separately estimate uncertainty about a new panel of annotators or
propagate a posterior distribution for the finite held-out reference.
Recorded annotators may also recur across questions; question-only
clustering does not establish independence for every crossed annotator
effect. No inter-rater noise ceiling or objective correctness claim follows.

As in the original study, the API's intervals concern an intended
population mean under sampling assumptions. They are not intervals for
the fixed evaluation subset's realized reference. Do not report their
containment of that subset's mean as population coverage. The original
[PPI](https://arxiv.org/abs/2301.09633) and
[PPI++](https://arxiv.org/abs/2311.01453) foundations do not resolve the
substantive choice of human-reference definition.

## Outcomes, falsifiable checks, and reporting

Retain every eligible pair/budget/seed/method/reference trial, with its
reference mean, signed error, absolute error, squared error, interval
width, selected power, and available comparison/vote costs. Report
definition-specific MAE/RMSE and within-definition method contrasts. Keep
per-pair results and all outcomes, including a changed ranking or worse
tuned-PPI error.

A lower MAE under mean-vote than under plurality is not, by itself, an
improvement by the estimator: the target changed. The relevant sensitivity
question is whether the direction and magnitude of method comparisons
are stable when each method is assessed against the same reference
definition. Changes in the held-out reference and matched method errors
can be shown descriptively. They are not evidence that one human reference
is more valid or that disagreement has been corrected.

Validation must check count integrity, reconstructed plurality, strict
numeric bounds, exact plurality-result compatibility, shared splits and
costs, hidden-label isolation, score preservation, and target reversal.
With one recorded vote per comparison, the two outcome definitions coincide
and all matched estimates and intervals should agree. Reversing A/B maps
both human scores to `1-Y`, preserves ties, and reverses estimates while
preserving standard errors and selected powers. These are implementation
checks, not generalization or coverage evidence.

The [sensitivity report and artifacts](../reports/sensitivity/REPORT.md)
are generated by `examples/reference_sensitivity.py` under `reports/sensitivity`:

```text
python examples/reference_sensitivity.py --output reports/sensitivity --plot
```

Preserve
configuration, input checksum, environment/source fingerprints, all split
IDs, tables, and plots. The final report must state the retrospective scope,
distinct estimands, reuse of the same votes, and any unresolved evidence
gaps. Repeated seeds and shared questions remain correlated: no naive
independent-row significance test is introduced by this sensitivity study.
