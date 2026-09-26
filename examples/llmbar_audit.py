"""Audit cached LLMBar judge accuracy using gold-free order agreement."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from judgecal.judge_audit import judge_accuracy_audit
from research_study import ROOT, sha256, write_json


REVISION = "900616bff90b6c6c8e1681f7d079250637c55992"
# Exact derived artifacts reproduced from all 27 independently pinned public sources.
# This runner reports the complete benchmark; custom panels use judge_accuracy_audit.
LABELS_SHA256 = "aafff984b410bd1f60e0a3196a10957c5a1f241caf8c85c4c391d8b1e2ef1f97"
PROVENANCE_SHA256 = "e6ee5d891fe5697a058af1247634ffc6c3d92338c60bbff14521b03a0f68fb98"
COHORTS = ("All", "Natural", "Adversarial")
JUDGES = ("GPT-4", "ChatGPT", "LLaMA2")
METHODS = ("raw_proxy", "human_only", "ppi", "ppi_tuned")
NAMES = {"raw_proxy": "Order-agreement proxy", "human_only": "Gold audit only",
         "ppi": "PPI (power 1)", "ppi_tuned": "Tuned PPI"}
COLORS = dict(zip(METHODS, ["#8a5c32", "#525a66", "#3673b5", "#00876c"]))


def read_verified_labels(path):
    path = Path(path)
    source = json.loads(path.with_name("dataset.json").read_text(encoding="utf-8"))
    if source.get("dataset_id") != "princeton-nlp/LLMBar" or source.get("source_revision") != REVISION:
        raise ValueError("the audit study requires the pinned official LLMBar source")
    if source.get("labels_sha256") != sha256(path):
        raise ValueError("input label checksum does not match the provenance sidecar")
    if sha256(path) != LABELS_SHA256:
        raise ValueError("input differs from the pinned complete LLMBar label snapshot")
    if sha256(path.with_name("dataset.json")) != PROVENANCE_SHA256:
        raise ValueError("provenance differs from the pinned complete LLMBar source manifest")
    return pd.read_csv(path, dtype={"sample_id": str, "instruction_id": str}), source


def cohort_frames(frame):
    yield "All", frame
    yield "Natural", frame.loc[frame.subset == "Natural"]
    yield "Adversarial", frame.loc[frame.subset != "Natural"]


def corpus_diagnostics(frame):
    """Reference-based diagnostics are descriptive, never fitted proxy inputs."""
    records = []
    cohorts = list(cohort_frames(frame)) + list(frame.loc[frame.subset != "Natural"].groupby("subset", sort=True))
    for cohort, rows in cohorts:
        for judge, group in rows.groupby("judge", sort=True):
            forward_valid = group.forward_prediction.isin([1, 2])
            reverse_valid = group.reverse_prediction.isin([1, 2])
            correct = group.forward_prediction == group.reference_label
            reverse_correct = group.reverse_prediction == group.reference_label
            agree = forward_valid & reverse_valid & (group.forward_prediction == group.reverse_prediction)
            records.append({"cohort": cohort, "judge": judge, "comparisons": len(group),
                            "instructions": group.instruction_id.nunique(),
                            "forward_accuracy": float(correct.mean()), "reverse_accuracy": float(reverse_correct.mean()),
                            "order_agreement": float(agree.mean()), "proxy_minus_accuracy": float(agree.mean()-correct.mean()),
                            "forward_invalid": int((~forward_valid).sum()), "reverse_invalid": int((~reverse_valid).sum()),
                            "agree_valid_count": int(agree.sum()), "agree_and_forward_correct": int((agree & correct).sum()),
                            "agree_but_wrong": int((agree & ~correct).sum()),
                            "accuracy_given_agreement": float(correct[agree].mean()) if agree.any() else np.nan})
    return pd.DataFrame(records)


def paired_method_differences(trials):
    keys = ["cohort", "judge", "seed", "labeled_fraction"]
    columns = keys + ["absolute_error", "squared_error"]
    human = trials.loc[trials.method == "human_only", columns]
    result = trials.merge(human, on=keys, suffixes=("", "_human"), validate="many_to_one")
    result["absolute_error_difference_vs_human"] = result.absolute_error-result.absolute_error_human
    result["squared_error_difference_vs_human"] = result.squared_error-result.squared_error_human
    return result[keys + ["method", "split_sha256", "absolute_error_difference_vs_human", "squared_error_difference_vs_human"]]


def make_report(summary, diagnostics, config, links):
    lines = ["# Auditing judge accuracy with an imperfect consistency proxy", "",
             "A second public benchmark with three cached judges, two presentation orders and published instruction-following references.", "",
             f"[Retrospective protocol]({links['protocol']}) | [Data provenance and license]({links['license']})", "",
             "## Target and proxy", "",
             "The outcome is whether the designated judge's **original-order canonical choice** matches "
             "LLMBar's supplied gold preference. An invalid original-order output counts as incorrect and is retained. "
             "This estimates correctness relative to the benchmark's instruction-following labels, not a model's "
             "win rate or a survey of subjective human preference.", "",
             "The proxy is one when both presentation orders yield valid, identical canonical choices, and zero otherwise. "
             "It uses no gold reference. Both orders can consistently choose the wrong output, so consistency "
             "is not accuracy. The original cache already maps reversed-order choices to original response IDs; "
             "the loader does not reverse them again.", "",
             "All 419 comparisons are retained for GPT-4, ChatGPT and LLaMA2 with the source's Vanilla+Rules prompt. "
             "They share 418 exact instruction strings; both comparisons of a repeated instruction remain together. "
             "The Natural set has 100 comparisons and the four Adversarial categories total 319. "
             "All-cohort means weight comparisons equally; they are not a 50/50 Natural/Adversarial macro-average.", "",
             "## Corpus diagnostics", "",
             "These full-corpus reference-based diagnostics describe the cached judges; they never enter proxy construction or fitting. "
             "Both choices are canonicalized in the source, and invalid outputs remain in each denominator.", "",
             "| Cohort | Judge | Comparisons | Original accuracy | Reversed accuracy | Order agreement | Consistent but wrong | Invalid original / reversed |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in diagnostics.itertuples():
        lines.append(f"| {row.cohort} | {row.judge} | {row.comparisons} | {row.forward_accuracy:.3f} | "
                     f"{row.reverse_accuracy:.3f} | {row.order_agreement:.3f} | {row.agree_but_wrong} | "
                     f"{row.forward_invalid} / {row.reverse_invalid} |")
    if config['plots']:
        lines += ["", "![Accuracy and order agreement](accuracy_vs_consistency.svg)"]
    lines += ["", "## Fixed-target audit experiment", "",
              f"For every cohort and each of {config['split_seeds']} deterministic seeds, 25% of instruction groups form "
              "a fixed evaluation pool. Audits use nested 20%, 40% and 60% fractions of total cohort instructions. "
              "Splits are identical across all three judges. Whole instruction groups stay together. "
              "Unused comparisons supply neither audit labels nor proxy predictions.", "",
              "Four estimators see the same pools: raw consistency proxy, gold audit only, fixed-power PPI and tuned PPI. "
              "PPI uses the numeric mean API with instruction-cluster variance. Audit gold fits the correction and "
              "scalar power; evaluation gold only scores the already formed estimates. "
              "Errors target realized heldout original-order accuracy, not a population confidence-interval coverage experiment.", "",
              "| Cohort | Judge | Audit instructions | Method | MAE (pp) | RMSE (pp) | Mean power |",
              "|---|---|---:|---|---:|---:|---:|"]
    for row in summary.itertuples():
        power = "--" if pd.isna(row.mean_selected_power) else f"{row.mean_selected_power:.3f}"
        lines.append(f"| {row.cohort} | {row.judge} | {row.labeled_fraction:.0%} | {NAMES[row.method]} | "
                     f"{100*row.mean_absolute_error:.3f} | {100*np.sqrt(row.mean_squared_error):.3f} | {power} |")
    gaps = summary.pivot(index=["cohort", "judge", "labeled_fraction"], columns="method", values="mean_absolute_error")
    differences = gaps.ppi_tuned - gaps.human_only
    tied = np.isclose(differences, 0, atol=1e-12, rtol=0)
    lines += ["", f"Tuned PPI has lower mean absolute error than the gold-audit-only baseline in "
              f"{int(((differences < 0) & ~tied).sum())}/{len(gaps)} cohort/judge/budget cells, "
              f"ties in {int(tied.sum())}, and has higher error in {int(((differences > 0) & ~tied).sum())} "
              "(absolute tolerance 1e-12 for ties). "
              "This is descriptive of the retained fixed corpus and splits, not a probability of improvement in deployment. "
              "Raw consistency is an accuracy proxy whose errors may be systematic; its mean receives no accuracy confidence interval."]
    adversarial = diagnostics.loc[diagnostics.cohort == "Adversarial"].set_index("judge")
    weak_tuned = summary.loc[(summary.cohort == "Adversarial") & summary.judge.isin(["ChatGPT", "LLaMA2"])
                             & (summary.method == "ppi_tuned")]
    weak_gaps = gaps.loc[[("Adversarial", judge, fraction) for judge in ["ChatGPT", "LLaMA2"]
                          for fraction in [.2, .4, .6]]]
    lines += ["", "### A proxy can be consistently wrong", "",
              f"For ChatGPT on the Adversarial subset, order agreement is {100*adversarial.loc['ChatGPT', 'order_agreement']:.1f}% "
              f"while original-order accuracy is {100*adversarial.loc['ChatGPT', 'forward_accuracy']:.1f}%. "
              f"Of {int(adversarial.loc['ChatGPT', 'agree_valid_count'])} comparisons with valid agreement, "
              f"{int(adversarial.loc['ChatGPT', 'agree_but_wrong'])} consistently choose the reference-incorrect answer. "
              "A high agreement rate alone is therefore insufficient to validate this judge.", "",
              f"Across the {len(weak_tuned)} Adversarial judge/budget cells for ChatGPT and LLaMA2, "
              f"{int(weak_tuned.mean_selected_power.eq(0).sum())} have zero tuned power in every retained split, "
              "recovering the gold-audit-only estimate. "
              f"Fixed-power PPI instead increases MAE in {int((weak_gaps.ppi > weak_gaps.human_only).sum())} of these cells. "
              "This fallback reflects the restricted [0,1] coefficient family and observed audit covariance; "
              "the study does not compare negative coefficients or claim global estimator optimality."]
    if config['plots']:
        lines += ["", "![Gold-audit budget results](judge_accuracy_budget.svg)"]
    lines += ["", "## Costs and retained evidence", "",
              "`human_labels_used` counts audited gold comparison labels and is zero for raw proxy inference. "
              "The same audited gold label reveals correctness for all three judges; do not sum that cost across judges. "
              "Cached judgment counts describe records consumed by each estimator, not new API calls or monetary costs. "
              "The proxy requires both presentation orders; audit-only inference uses original-order audit judgments. "
              "Heldout labels used for scoring are recorded separately from inference labels.", "",
              "Every trial is retained in `trials.csv`; `paired_method_differences.csv` subtracts the same-split "
              "gold-audit-only error. Exact instruction and comparison IDs are in `split_manifest.json`. "
              "Repeated splits, nested cohorts and judges share data, so no independent-row significance test or "
              "naive Monte Carlo standard error is attached to those empirical contrasts.", "",
              "## Limits", "",
              "- These are curated references, not infallible latent truth. Adversarial construction/filtering makes "
              "the benchmark deliberately difficult; the source used ChatGPT evaluators in some filtering steps. "
              "It is not a random sample of deployment tasks, and comparative model results inherit that selection.",
              "- The GPT-4 cache uses an undated `gpt-4` alias; ChatGPT configuration specifies Azure "
              "`gpt-35-turbo-0613`; LLaMA2 is `Llama-2-70b-chat-hf` without a weight revision. "
              "Cache pins reproduce this historical reanalysis, not fresh model calls or 2025/2026 model performance.",
              "- One-way instruction grouping handles the known duplicate but does not prove arbitrary independence "
              "or normal-interval validity. Finite-sample iid bounds from the separate synthetic study are not applied here.",
              "- No new human annotation or model call was purchased. No runtime label-saving or dollar-saving guarantee is claimed.", "",
              "## Reproduce offline", "", "```bash", 'pip install -e ".[dev,report]"',
              "python examples/llmbar_audit.py --output reports/reproduced/llmbar-audit --plot", "```", "",
              "The committed derived labels contain only IDs, categories and choices. Their checksum is checked "
              "against pinned source provenance before analysis. `examples/prepare_llmbar.py` can rebuild that snapshot "
              "from the specified public files without running a model. Defaults use 30 split seeds; smaller "
              "`--seeds` runs are explicitly recorded in configuration.", ""]
    return "\n".join(lines)


def save_plot(fig, output, name):
    fig.savefig(output/f"{name}.png", dpi=180)
    fig.savefig(output/f"{name}.svg", metadata={"Date": None})
    svg = output/f"{name}.svg"
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines())+b"\n")


def make_plots(summary, diagnostics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"svg.hashsalt": "judgecal-llmbar-v1", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True, constrained_layout=True)
    for ax, cohort in zip(axes, COHORTS):
        rows = diagnostics.loc[diagnostics.cohort == cohort].set_index("judge").loc[list(JUDGES)]
        positions = np.arange(len(JUDGES))
        ax.bar(positions-.18, rows.forward_accuracy, .36, color="#3673b5", label="Original-order accuracy")
        ax.bar(positions+.18, rows.order_agreement, .36, color="#ba7631", label="Order agreement")
        ax.set(title=cohort, xticks=positions, xticklabels=JUDGES, ylim=(0, 1.02))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Fraction of cached comparisons")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("LLMBar: agreeing across presentation orders is not the same as being correct", fontsize=12)
    save_plot(fig, output, "accuracy_vs_consistency")
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True, sharey=True, constrained_layout=True)
    for row, cohort in enumerate(COHORTS):
        for column, judge in enumerate(JUDGES):
            ax = axes[row, column]
            for method in METHODS:
                values = summary.loc[(summary.cohort == cohort) & (summary.judge == judge) & (summary.method == method)]
                ax.plot(values.labeled_fraction*100, values.mean_absolute_error*100, marker="o",
                        color=COLORS[method], label=NAMES[method], linewidth=1.8)
            if row == 0:
                ax.set_title(judge)
            if column == 0:
                ax.set_ylabel(f"{cohort}\nMean absolute error (pp)")
            if row == 2:
                ax.set_xlabel("Audit instructions (% of cohort)")
            ax.set_xticks([20, 40, 60])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Estimating cached original-order judge accuracy on fixed evaluation pools", fontsize=12)
    save_plot(fig, output, "judge_accuracy_budget")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"reports/llmbar/labels.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"reports/llmbar-audit")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("seeds must be positive")
    frame, source = read_verified_labels(args.input)
    output = args.output
    if output.resolve() == args.input.resolve().parent:
        raise ValueError("choose a separate output directory to preserve the input snapshot")
    names = ["trials", "summary", "corpus_diagnostics", "paired_method_differences"]
    generated = {f"{name}.csv" for name in names} | {"split_manifest.json", "config.json", "dataset.json", "REPORT.md"}
    if args.plot:
        generated |= {f"{stem}.{ext}" for stem in ["accuracy_vs_consistency", "judge_accuracy_budget"] for ext in ["png", "svg"]}
    if output.exists() and {p.name for p in output.iterdir()} - generated - {"reproducibility.json"}:
        raise ValueError("output contains stale or unrelated artifacts; use a fresh directory")
    trials, summary = judge_accuracy_audit(frame, seeds=tuple(range(args.seeds)))
    split_manifest = trials.attrs["split_manifest"]
    audit_protocol = trials.attrs["audit_protocol"]
    trials.attrs = {}
    diagnostics, paired = corpus_diagnostics(frame), paired_method_differences(trials)
    config = {"protocol": "cached-judge-accuracy-audit-v1", "split_seeds": args.seeds, "plots": args.plot,
              "audit_protocol": audit_protocol, "source_labels_sha256": source["labels_sha256"],
              "source_revision": REVISION, "cohorts": list(COHORTS), "judges": list(JUDGES)}
    output.mkdir(parents=True, exist_ok=True)
    for name, table in zip(names, [trials, summary, diagnostics, paired]):
        table.to_csv(output/f"{name}.csv", index=False, lineterminator="\n")
    write_json(output/"split_manifest.json", split_manifest, compact_records=True)
    write_json(output/"config.json", config)
    write_json(output/"dataset.json", source)
    links = {name: os.path.relpath(ROOT/path, output).replace("\\", "/") for name, path in {
        "protocol": "docs/LLMBAR_PROTOCOL.md", "license": "reports/llmbar/DATA_LICENSE.md"}.items()}
    (output/"REPORT.md").write_bytes(make_report(summary, diagnostics, config, links).encode("utf-8"))
    if args.plot:
        make_plots(summary, diagnostics, output)
    sources = sorted((ROOT/"src/judgecal").glob("*.py")) + [Path(__file__), ROOT/"examples/research_study.py",
                ROOT/"examples/prepare_llmbar.py", ROOT/"docs/LLMBAR_PROTOCOL.md", ROOT/"pyproject.toml"]
    manifest = {"python": platform.python_version(), "platform": platform.system(),
                "packages": {name: version(name) for name in ["judgecal", "numpy", "scipy", "pandas"]},
                "input_labels_sha256": sha256(args.input), "input_provenance_sha256": sha256(args.input.with_name("dataset.json")),
                "source_text_normalization": "CRLF to LF before SHA-256 (matches Git text attributes)",
                "source_sha256": {p.relative_to(ROOT).as_posix():
                                  hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sources},
                "artifacts_sha256": {name: sha256(output/name) for name in sorted(generated)}}
    if args.plot:
        manifest["packages"]["matplotlib"] = version("matplotlib")
    write_json(output/"reproducibility.json", manifest)
    print(f"Wrote {len(trials)} method trials, {len(split_manifest)} common splits and {len(diagnostics)} corpus diagnostics.")


if __name__ == "__main__":
    main()
