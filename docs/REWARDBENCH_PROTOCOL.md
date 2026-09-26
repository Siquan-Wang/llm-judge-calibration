# RewardBench cross-judge accuracy audit: retrospective protocol

This study asks whether the choice agreement between two cached generative
judges helps estimate either judge's benchmark-reference correctness with a
limited gold audit. It evaluates established correction rules on shared
audit and held-out pools. The proxy is identical in both judge directions;
the correctness outcome and its relationship to that proxy can differ.

The design is retrospective, not preregistered. Source inspection and earlier
project experiments informed it. It introduces no new estimator or inference
theorem. The main analysis uses the non-LLMBar portion of RewardBench, with
the full mixture and the already studied LLMBar component reported as
separate sensitivity analyses.

## Sources and retained comparisons

The benchmark is the official
[allenai/reward-bench snapshot at 168d848cdbbea9764fae4a544dc9ca1e6cca4931](https://huggingface.co/datasets/allenai/reward-bench/tree/168d848cdbbea9764fae4a544dc9ca1e6cca4931).
Cached judgments come from
[allenai/reward-bench-results at 96302cc604e4f257dc526272c803c4eff27e3469](https://huggingface.co/datasets/allenai/reward-bench-results/tree/96302cc604e4f257dc526272c803c4eff27e3469).
Fix these two models and analyze both target–auxiliary directions:

- `openai/gpt-4o-2024-08-06`
- `openai/gpt-4o-mini-2024-07-18`

These are actual per-comparison caches labeled `Generative RM`, rather than
scalar reward-model scores or inputs awaiting model calls. Each selected
cache has a binary score for every one of the same 2,985 comparisons. Retain
all comparisons and preserve their 23 source subsets. No fresh model
inference or new annotation is required.

The loader must match the exact prompt, chosen response and rejected
response to the pinned official benchmark, validate subset and ID metadata,
and match both caches by content rather than row position. Raw ID `3692`
occurs twice on distinct comparisons; it is not a unique join key. A stable
comparison ID hashes the prompt and the sorted pair of answer strings,
excluding reference orientation. A prompt-group ID hashes the exact prompt
without normalization or gold information. The 2,985 complete comparisons
are unique and form 2,733 exact prompt groups.

The inspected official
[runner at 47512a010f748dcdacab422654b43c51c91fa63d, lines 188–225](https://github.com/allenai/reward-bench/blob/47512a010f748dcdacab422654b43c51c91fa63d/scripts/run_generative.py#L188)
makes one randomly ordered comparison and scores the parsed choice as 1 for
the benchmark's chosen answer, 0 for its rejected answer, or .5 for an
unrecognized/error result or unsupported case. The matching
[model registry and helper](https://github.com/allenai/reward-bench/blob/47512a010f748dcdacab422654b43c51c91fa63d/rewardbench/generative.py)
include both selected model identifiers. This implementation documents the
scoring semantics; its commit is not recorded as the exact cached execution
revision. Raw verdicts, shuffle states and complete invocations are absent.

The candidate `google/gemini-1.5-flash-001` cache contains two .5 fallback
scores. The bounded study excludes that entire candidate under its strict
binary reconstruction contract; it neither deletes those rows from a shared
panel nor coerces the fallbacks to choices. Half-scores are not evidence of
ties, half-correct responses or two-order averaging. Both selected OpenAI
caches are complete and binary. Restricting the study to one model family
limits the diversity of evidence.

## Outcome and gold-free agreement

For each comparison, let `C_A` and `C_B` be the two cached binary correctness
bits. For designated target judge `j`, define

```text
Y_A = C_A                     Y_B = C_B
F_AB = 1{C_A == C_B} = F_BA.
```

This equality reconstructs canonical choice agreement. If both judges are
correct, they chose the same reference answer. If both are incorrect, they
chose the same sole alternative. Different correctness bits mean different
choices. Reversing the common reference complements both correctness bits
and leaves `F` unchanged. The stable comparison ID is also invariant to
this reversal. Thus this particular function reconstructs an observable
choice relation that does not require a gold label, even though the cache
stores gold-dependent scores.

The binary, common-answer and common-reference checks are essential. An
auxiliary judge's correctness bit by itself is not a gold-free proxy.
Equality involving nonbinary fallbacks would not identify choice agreement.
Missing raw judgments limit independent recovery of parsing details; the
study does not claim same-model order consistency or estimate position bias.
Agreement may be wrong and is not a calibrated correctness probability.
Dependence between audit `Y` and `F` is expected and does not itself violate
the correction's assumptions.

The outcome is correctness relative to heterogeneous benchmark references:
response quality, instruction following, safety/refusal conventions, and
code/math criteria. It is not a uniform human-preference vote or proof of
latent truth. Each comparison has equal weight, including repeated prompts.
The resulting row-average correctness is not RewardBench's official
weighted leaderboard score.

## Cohorts and deterministic splits

| Cohort | Role | Comparisons | Exact prompt groups | Subsets |
|---|---|---:|---:|---:|
| `NonLLMBar` | Primary | 2,566 | 2,315 | 18 |
| `All` | Full-mixture sensitivity | 2,985 | 2,733 | 23 |
| `LLMBar` | Historical-overlap sensitivity | 419 | 418 | 5 |

The two component cohorts share no exact prompt string, but `All` overlaps
both. The LLMBar component reuses the earlier corpus with different cached
judges; it is not an independent new-domain replication. Excluding this
known overlap does not certify absence of all shared benchmark lineage:
the remaining sources include MT-Bench-derived comparisons. Exact prompt
grouping prevents identical prompts from crossing a split; it does not
establish semantic independence or eliminate related multilingual tasks.

For each cohort and seed 0 through 29:

1. Sort its complete prompt IDs and permute them using
   `numpy.random.default_rng(seed).permutation`.
2. Reserve the first `floor(0.25 * G)` groups for evaluation, where `G` is
   the total number of groups in that cohort.
3. Take nested prefixes from the remaining bank with
   `floor(b * G)` audit groups for `b` in `{0.20, 0.40, 0.60}`.
4. Leave every other group unused. Use the same pools for both designated
   judges and all methods.

Both audit and evaluation fractions refer to **total cohort groups**, not
the remaining bank or comparison rows. Actual row counts vary with group
sizes and are exported alongside group counts. The default minimum cohort
size is ten groups; require at least two groups in each inference pool.
Audit, evaluation and unused groups are disjoint;
the evaluation pool stays fixed as the nested audit budget grows. Export
the exact group/sample IDs and split hashes.

The agreement proxy is fixed before fitting. Audit outcomes alone enter
correction and power tuning. Held-out outcomes enter post-inference scoring;
source-integrity checks do not select rules using them. Unused outcomes
never enter estimation. Neither a favorable auxiliary nor favorable
cohorts are selected after observing held-out errors.

## Point rules and target-specific interpretation

All six methods use the same evaluation pool and, where needed, the same
gold audit. Corrected points take the form
`mean(Y_L) + lambda * (mean(F_U) - mean(F_L))`.

| Method | Rule |
|---|---|
| `raw_proxy` | Evaluation agreement mean, an uncorrected surrogate for accuracy |
| `human_only` | Reference-only audit correctness mean; existing method identifier |
| `ppi` | Fixed coefficient `lambda=1` |
| `ppi_tuned` | Existing population-variance tuning in `[0,1]` |
| `ppi_signed` | Existing population-variance tuning in `[-1,1]` |
| `audit_residual` | Existing signed audit-residual criterion, point only |

The last three coefficients use only the permitted audit outcomes and
cached proxies. No oracle or held-out label is used in fitting. All points,
including values outside `[0,1]`, are retained without clipping. Zero-power
fallbacks and worse outcomes remain in the exports.

The scored truth is each designated judge's realized correctness mean on
the fixed held-out pool. Population-variance tuning and audit-residual
tuning optimize different criteria, as explained in the
[estimand protocol](ESTIMAND_PROTOCOL.md). Comparing their point errors
against this common target does not redefine the population API or make a
new claim about its optimality.

Existing population PPI intervals use the prompt-cluster sandwich and their
documented asymptotic sampling assumptions. They concern a hypothetical
population mean, not the realized held-out target scored here. Widths,
standard errors and variance ratios are working-model diagnostics. **No
empirical interval coverage statistic or coverage guarantee is reported.**
The iid finite-sample bounds are not applied to this grouped corpus. The
`audit_residual` method has no interval, standard error or population
variance ratio; its criterion is not an uncertainty estimate. The raw
agreement surrogate also receives no correctness interval. See
[Methods](METHODS.md) for the established formulas and limitations.

## Paired comparisons and logical costs

Report bias, MAE and RMSE separately by cohort, target judge and audit
budget. Export per-split absolute- and squared-error differences against
`human_only` and `ppi_signed`; negative differences favor the compared
method. All cohort/direction/budget cells and every seed are retained.
Source-subset diagnostics keep the mixture visible without creating an
additional family of small-subset fitted experiments.

The 30 overlapping splits repeatedly reuse a fixed corpus. Their averages
and paired differences are descriptive; they do not receive an iid Monte
Carlo standard error or establish significance. The two judge directions
also share the agreement proxy and gold references. They are different
targets, not independent additional experiments. Avoid selecting only
favorable cells or ranking losses against different estimands.

The `human_only` identifier is retained for compatibility; RewardBench
references are not uniformly new human annotations. Cost fields use
`reference_labels_used` and its summary counterpart
`mean_reference_labels_used`.

One revealed gold reference scores both designated judges. Label counts
are shared and must not be summed as independent annotation purchases.
Corrected agreement methods use two cached judgments per audit and
evaluation comparison. Raw agreement uses the two evaluation judgments;
audit-only uses the designated judge's audited choice and the reference.
Record these logical inputs, not fresh calls, token usage, dollar costs,
annotation time or measured deployment savings. This study reveals already
available benchmark labels and purchases none.

## Provenance, publication and limits

The dated 2024 caches support reproducible historical reanalysis, not claims
about current 2026 endpoints or exact reproduction of fresh model calls.
The dataset's curated source mixture, model-family restriction, missing
raw verdicts and incomplete invocation provenance remain part of every
interpretation. Full-corpus diagnostics do not enter model selection or
power fitting.

Derived artifacts contain numeric scores, source categories, model IDs and
content/group hashes; they do not republish raw prompts, responses or model
completions. Record the immutable source URLs and byte hashes. The benchmark
has ODC-BY database terms and source-specific content notices; the code has
Apache-2.0 terms. The separately hosted results card states research release
but supplies no named license at the pinned revision. Preserve attribution
and that qualification rather than assigning the project's code license to
cached data or assuming that omission of text resolves every reuse term.

Required verification includes complete source joins, duplicate-ID
preservation, strict binary validation, reference-flip invariance of IDs and
the proxy, held-out-outcome isolation, shared splits/costs, and independent
recalculation of paired errors. Results should state what happened for each
specified target under this protocol, retain failures, and avoid claims of
new PPI, latent truth, universal consensus reliability or broad independent
external validation.
