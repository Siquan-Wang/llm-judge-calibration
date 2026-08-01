# judgecal — statistically rigorous LLM-as-a-judge evaluation

LLM-as-a-judge is now the default way to evaluate language models, but most
pipelines report a single win-rate with no uncertainty and treat the judge as
if it were ground truth. `judgecal` treats the judge as what it is — a **noisy
measurement instrument** — and applies standard statistical machinery
(measurement-error correction, Bayesian credible intervals, chance-corrected
agreement) to make judge-based evaluations honest.

## What it does

| Question | Tool |
|---|---|
| How much does my judge agree with humans, with uncertainty? | `agreement_with_ci` (Beta-Binomial credible interval), `cohens_kappa` |
| Is my model's win-rate significantly above 50%? | `win_rate_with_ci` (bootstrap), `compare_models` |
| My judge is imperfect — what is the *human-judged* win-rate? | `judge_confusion` + Rogan–Gladen measurement-error correction |
| Are my judge's scalar scores calibrated to human preference? | `platt_scaling`, `isotonic_calibration` |
| Is judge A better than judge B at matching humans? | `paired_agreement_gap` |
| What is the human-annotator-noise ceiling? | inter-annotator agreement via `agreement_with_ci` |

The key idea behind the win-rate correction: if a judge has sensitivity `se`
and specificity `sp` against human labels (estimated on a small human-labeled
calibration set), the observed judge win-rate is biased:

```
p_obs = se * p_true + (1 - sp) * (1 - p_true)
```

Inverting this (the Rogan–Gladen estimator, standard in epidemiology for
imperfect diagnostic tests) recovers an unbiased estimate of the human-judged
win-rate — so you can label 200 items with humans and 20,000 with the judge,
and still report a number that means what people think it means.

## Install

```bash
pip install -e ".[data,dev]"
```

Core dependencies are just numpy / scipy / pandas. The `data` extra adds
Hugging Face `datasets` for the MT-Bench demo.

## Quickstart

```python
from judgecal import agreement_with_ci, judge_confusion, compare_models

# judge / human labels are "A", "B", or "tie" per comparison
ci = agreement_with_ci(judge_labels, human_labels)
print(f"agreement {ci.point:.3f}, 95% CrI [{ci.low:.3f}, {ci.high:.3f}]")

# estimate judge error on the human-labeled subset ...
conf = judge_confusion(judge_labels, human_labels)

# ... then evaluate a model pair on the full judge-labeled set,
# correcting the win-rate for judge measurement error
result = compare_models(judge_labels_full, confusion=conf)
print(result["win_rate"], result["corrected_win_rate"], result["significant"])
```

## Demo: GPT-4 as a judge on MT-Bench

The demo uses [`lmsys/mt_bench_human_judgments`](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
(Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*,
NeurIPS 2023) — 3.3k expert human pairwise votes and matching GPT-4 votes
over six models. It reproduces the paper's judge-human agreement analysis and
goes further:

```bash
python examples/demo_mtbench.py
```

1. **Agreement with uncertainty** — GPT-4 vs. human majority vote, with a
   Beta-Binomial credible interval and Cohen's kappa instead of a bare
   percentage.
2. **Measurement-error-corrected win-rates** — estimates GPT-4's
   sensitivity/specificity against humans, then reports per-model-pair
   win-rates both raw and Rogan–Gladen-corrected, with bootstrap CIs.
3. **The annotator-noise ceiling** — computes human-human inter-annotator
   agreement and asks whether the judge is statistically distinguishable
   from a human annotator.

## Testing

```bash
pytest
```

Tests are fully offline (synthetic data with known ground truth); one
end-to-end test verifies the measurement-error correction recovers a known
true win-rate from deliberately corrupted judge labels.

## Roadmap

- [ ] Position-bias and verbosity-bias estimation from swapped-order judgments
- [ ] Bayesian hierarchical model for per-category judge reliability
- [ ] Sample-size calculator: "how many human labels do I need to calibrate?"
- [ ] Multi-judge ensembling with Dawid–Skene-style latent-truth models
- [ ] Technical report with MT-Bench + Chatbot Arena case studies

## License

MIT
