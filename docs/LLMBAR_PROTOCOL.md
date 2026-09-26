# LLMBar judge-accuracy audit: retrospective protocol

This study estimates a cached judge's accuracy against LLMBar's supplied
instruction-following references using a limited gold audit and a proxy
computed from response-order agreement. It asks whether that proxy helps
after calibration, and when tuning should reduce reliance on it. Accuracy
against this curated reference is distinct from a response model's win rate,
subjective human preference, or unrestricted population correctness.

The design is retrospective. The benchmark, earlier experiments and
development checks were available during its construction. Neither the
analysis nor the choice of proxy is preregistered. The PPI methods are
established estimators; the contribution here is an auditable application
and assessment of their limitations on a second public benchmark.

## Certified source and outcome construction

The source is [Princeton NLP's official LLMBar repository, revision
900616bff90b6c6c8e1681f7d079250637c55992](https://github.com/princeton-nlp/LLMBar/tree/900616bff90b6c6c8e1681f7d079250637c55992),
associated with [Zeng et al., Evaluating Large Language Models at Evaluating
Instruction Following, ICLR 2024](https://arxiv.org/abs/2310.07641).
The repository's MIT notice is preserved in
[the derived-data attribution](../reports/llmbar/DATA_LICENSE.md).
The label snapshot contains IDs, categories, references and choices; it
does not republish source instructions, responses or model completions.

The certified snapshot has 419 comparison pairs and 418 exact instruction
strings, repeated for each of three cached judges (1,257 rows). Natural has
100 pairs; Adversarial consists of Neighbor 134, GPTInst 92, GPTOut 47 and
Manual 46. The judges are GPT-4, ChatGPT and LLaMA2. We use the source's
`Vanilla` cache, which corresponds to **Vanilla+Rules in the paper**;
`Vanilla_NoRules` is a different condition.

The loader matches each cached record to exactly one reference record using
the complete `(input, output_1, output_2)` tuple, rather than row position.
It checks reference labels separately. `sample_id` hashes that tuple without
gold; `instruction_id` hashes the exact instruction's UTF-8 bytes without
normalization. Instruction groups are shared across judges. Exact source
URLs and byte hashes for all 27 source files, model configurations,
transformation definitions and count checks are recorded in
`reports/llmbar/dataset.json`.

Both cached prediction orders already use the original output IDs 1 and 2.
The upstream evaluator stores `winner` for the original presentation and
`reverse_label(winner)` for the swapped presentation. The loader therefore
does **not** flip the cached reversed winner again. A null parsed winner
becomes explicit prediction 0. Invalid predictions are retained; LLaMA2 has
one invalid original-order and one invalid reversed-order choice, while
the other two judge caches have none.

The complete snapshot's SHA-256 is
`aafff984b410bd1f60e0a3196a10957c5a1f241caf8c85c4c391d8b1e2ef1f97`.
The report runner checks both the certified snapshot and its exact provenance
sidecar. A separately assembled panel belongs in the general benchmark API,
not under this report's fixed-corpus claims.

## Accuracy target and gold-free proxy

For comparison `i` and designated judge `j`, define

```text
Y_ij = 1{forward_prediction_ij == reference_label_i}
F_ij = 1{forward and reverse predictions are both valid
          and their canonical choices are equal}.
```

`Y` is correctness of the designated **original-order** judge choice, not an
order-averaged accuracy. An invalid original prediction counts as incorrect.
`F` uses no gold label. An invalid prediction in either order makes `F=0`.
Agreement can be confidently wrong, and disagreement does not identify
which order is correct. This proxy is not a calibrated correctness
probability merely because it is binary.

Each comparison has equal weight. The All cohort reflects its 100/319
Natural/Adversarial composition, not a 50/50 category macro-average.
The two comparisons with an identical instruction remain separate equally
weighted observations but are always assigned together. Full-corpus
accuracy, reverse accuracy, agreement, invalid counts and consistent-wrong
counts are descriptive diagnostics; they do not enter proxy construction
or estimator fitting.

## Sampling and methods

Analyze All, Natural and Adversarial separately. These cohorts overlap, and
repeated splits reuse a fixed corpus; they are not independent experimental
replications. For each cohort and seed 0 through 29:

1. Sort and shuffle complete instruction IDs using the recorded seeded rule.
2. Fix `floor(0.25 * G)` evaluation instructions, where `G` is the cohort's
   total instruction count.
3. Use nested prefixes of the remaining instruction bank containing
   `floor(b * G)` audit instructions for `b` in `{0.20,0.40,0.60}`.
4. Keep every other instruction unused. All judges share the same split.

The audit, evaluation and unused instruction groups are disjoint. Larger
budgets enlarge the audit while leaving the evaluation pool fixed. Fractions
refer to total cohort instructions, not the remaining bank. Exact instruction
and comparison IDs, group counts and split hashes are exported.

Four methods are evaluated on identical pools:

| Method | Estimate |
|---|---|
| `raw_proxy` | Mean order agreement on the evaluation pool |
| `human_only` | Mean forward correctness on the gold audit |
| `ppi` | Audit mean correctness plus evaluation-minus-audit proxy mean |
| `ppi_tuned` | Same correction with the existing variance-tuned power in `[0,1]` |

The raw proxy is a deliberately uncorrected accuracy surrogate; its own
quantity is agreement. Audit gold alone supplies correctness for calibration
and power tuning. Evaluation gold enters only cross-judge integrity checks
and post-inference scoring. Unused observations never enter estimation.
The benchmark rejects missing judge/sample cells, inconsistent shared
metadata or gold, and unsupported prediction values rather than silently
dropping difficult cases.

## Fixed heldout reference and interval interpretation

The scored reference is each judge's realized forward accuracy on the fixed
heldout pool. Errors, MAE and RMSE compare point estimates with that reference.
They are descriptive fixed-corpus estimation errors.

The corrected methods also return normal intervals with an instruction-cluster
sandwich variance. Those intervals concern a hypothetical population mean
under representative independent-pool and sufficiently many independent-group
assumptions. They are **not confidence intervals for the realized heldout
accuracy being used to score error**. Their widths are retained as method
diagnostics; this study reports no interval coverage statistic or coverage
guarantee. Raw-proxy intervals are omitted because their uncertainty would
concern agreement rather than correctness. The iid finite-sample API is not
applied to this grouped benchmark.

The fitted power uses the same audit as the correction and has the existing
plug-in asymptotic interpretation described in [Methods](METHODS.md).
Estimated variance reduction is not a finite-sample MSE, interval-validity
or annotation-saving guarantee. A selected zero power exactly falls back
to the gold-audit mean; such outcomes are retained rather than treated as
failed tuning.

## Costs and comparisons

One audited comparison's gold reference can determine correctness for all
three cached judges. Human-label counts are shared across judges and must
not be summed as separate annotation purchases. All three corrected methods
receive the same audit-label budget; the raw proxy uses zero audit gold.
Existing labels are revealed for this offline experiment; no new labels
are acquired.

Cached-prediction usage is also recorded. Gold audit only needs original-order
audit choices; PPI methods use both orders in audit and evaluation, while
the raw proxy uses both evaluation orders. These are method-input counts,
not actual new API calls, tokens, monetary costs or annotation time. The
agreement proxy has a second-order prediction requirement even though it
requires no gold.

Report all judge/cohort/budget cells, every seed and all four methods. Paired
error differences subtract gold-audit-only error within the same split.
Negative differences favor the compared method. Aggregate signs and zero-power
frequencies describe this cached panel; they do not establish significance,
universal superiority or label savings. No iid standard error is assigned
to averages across these overlapping corpus splits.

## Provenance limits and reproduction

The references target curated objective instruction following. Natural
examples were filtered or modified, and adversarial examples were constructed
or filtered using ChatGPT evaluators. The cohorts must remain visible;
their selection does not support universal rankings of judge families.

These are historical cached predictions. GPT-4 is an undated `gpt-4` alias;
ChatGPT's config names `gpt-3.5-turbo` with engine `gpt-35-turbo-0613`;
LLaMA2 names `meta-llama/Llama-2-70b-chat-hf` without a weight revision.
Pinning the source reproduces this reanalysis, not fresh inference from an
identical modern endpoint. Prompt rules and parse limitations are part of
the evaluated setup. No runtime model API is called.

The [LLMBar audit report](../reports/llmbar-audit/REPORT.md) retains results,
diagnostics, trials, split manifests, configurations and source/artifact
hashes. Reproduce from the committed certified label snapshot:

```bash
pip install -e ".[dev,report]"
python examples/llmbar_audit.py --output reports/reproduced/llmbar-audit --plot
```

The default uses 30 split seeds; smoke runs must record their smaller count.
Keep original cached-order results, invalid parses and weak-proxy outcomes.
The purpose is to expose when proxy correction is useful or unhelpful under
this specified protocol, without treating self-consistency as correctness.
