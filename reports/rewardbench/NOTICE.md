# RewardBench sources and derived research data

The accompanying CSV contains numeric choices, correctness, categories and
content/group hashes for a research reanalysis. It contains no source prompts,
answer text or judge completions. `dataset.json` records exact source revisions,
file digests, reconstruction rules and source limitations.

- Benchmark: [allenai/reward-bench at 168d848cdbbea9764fae4a544dc9ca1e6cca4931](https://huggingface.co/datasets/allenai/reward-bench/tree/168d848cdbbea9764fae4a544dc9ca1e6cca4931).
- Cached results: [allenai/reward-bench-results at 96302cc604e4f257dc526272c803c4eff27e3469](https://huggingface.co/datasets/allenai/reward-bench-results/tree/96302cc604e4f257dc526272c803c4eff27e3469).
- Inspected implementation: [allenai/reward-bench at 47512a010f748dcdacab422654b43c51c91fa63d](https://github.com/allenai/reward-bench/tree/47512a010f748dcdacab422654b43c51c91fa63d).

Attribute the benchmark and cached judgments to the RewardBench authors and
Allen Institute for AI, and the underlying comparisons to the source datasets
listed in the pinned [dataset card](https://huggingface.co/datasets/allenai/reward-bench/blob/168d848cdbbea9764fae4a544dc9ca1e6cca4931/README.md).
The benchmark includes AlpacaEval, MT-Bench, LLMBar, refusal data, XSTest,
Do Not Answer, HumanEvalPack and PRM Math comparisons. References have different
construction and correctness semantics; this reanalysis does not supply new
human judgments.

The benchmark's [ODC-BY notice](https://huggingface.co/datasets/allenai/reward-bench/blob/168d848cdbbea9764fae4a544dc9ca1e6cca4931/LICENSE.md)
distinguishes database rights from individual content rights. The dataset card
also identifies source-specific terms, including noncommercial terms for some
components. The upstream code is Apache-2.0. The independently hosted results
card states that the results are released for further research but provides no
named license at the pinned revision. This project does not infer a score-cache
license from neighboring benchmark or code licenses, or relicense those source
records under the package's MIT license. Derived numeric packaging does not
establish that all upstream reuse permissions are uniform.

The two selected caches identify `openai/gpt-4o-2024-08-06` and
`openai/gpt-4o-mini-2024-07-18`. They reproduce historical saved judgments,
not current model behavior. Their exact executed code revision, raw verdicts,
shuffle state and complete invocation are not recorded. The inspected code
supports the binary scoring interpretation, without recovering those missing
execution details. No model calls or new annotations were purchased.

The inspected but unselected [Gemini cache](https://huggingface.co/datasets/allenai/reward-bench-results/blob/96302cc604e4f257dc526272c803c4eff27e3469/eval-set-scores/google/gemini-1.5-flash-001.json)
has SHA-256 `5a462b155c9c1f9689159a965f3574f069c5703ceb011a7f49209022bc02c0d8`
and two half-score fallback records (IDs 2540 and 3347). It informed binary
schema eligibility only; it is not an extraction input or a fitted competitor.

See the [retrospective protocol](https://github.com/Siquan-Wang/llm-judge-calibration/blob/main/docs/REWARDBENCH_PROTOCOL.md) for the
strict binary reconstruction contract and whole-cache candidate eligibility.
