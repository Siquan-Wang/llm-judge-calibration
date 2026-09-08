# MT-Bench data attribution

The derived `labels.csv` contains labels, model identifiers, question IDs and
vote/ordering metadata from
[`lmsys/mt_bench_human_judgments`](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments),
revision `f7d2896d2cc5d80f8b55c2bbc722613555233c25`.

The upstream dataset is licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
That license continues to apply to this derived data, separately from the
MIT license for the `judgecal` source code.

Attribution: Lianmin Zheng et al., **Judging LLM-as-a-Judge with MT-Bench and
Chatbot Arena**, NeurIPS 2023.
[Paper](https://arxiv.org/abs/2306.05685) ·
[FastChat repository](https://github.com/lm-sys/FastChat).

Changes: model pairs are alphabetically ordered, winners are reoriented,
identical duplicate annotator votes are removed, unique human votes are
aggregated by plurality, tied counts remain ties, and comparisons are joined
to cached GPT-4 judgments. Source `tie (inconsistent)` labels remain ties
with an explicit flag. `dataset.json` records transformation counts and checksum.

No prompt or model response text is redistributed. Reports and plots are
generated analyses; upstream authors do not endorse this analysis or project.
