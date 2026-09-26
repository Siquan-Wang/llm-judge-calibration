"""Offline sensitivity to human-reference definition on identical audit splits."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from judgecal.benchmark import reference_definition_sensitivity
from research_study import COLORS, METHODS, NAMES, ROOT, read_verified_labels, sha256, write_json


REFERENCES = {"plurality": "Plurality outcome", "mean_vote": "Mean vote score"}
KEYS = ["model_a", "model_b", "labeled_fraction", "evaluation_fraction", "seed", "method"]


def reference_changes(frame):
    """Describe each comparison, without treating votes as independent rows."""
    columns = ["question_id", "turn", "model_a", "model_b", "n_human_votes",
               "n_human_a", "n_human_b", "n_human_ties"]
    result = frame[columns].copy()
    result["plurality_score"] = frame.human_winner.map({"A": 1., "B": 0., "tie": .5})
    result["mean_vote_score"] = (frame.n_human_a + .5*frame.n_human_ties)/frame.n_human_votes
    result["score_change"] = result.mean_vote_score - result.plurality_score
    result["votes_disagree"] = (frame[["n_human_a", "n_human_b", "n_human_ties"]] > 0).sum(axis=1) > 1
    return result.sort_values(["model_a", "model_b", "question_id", "turn"]).reset_index(drop=True)


def paired_reference_differences(trials):
    columns = KEYS + ["split_sha256", "estimate", "heldout_human_reference", "absolute_error",
                      "squared_error", "selected_power", "human_comparisons_used", "human_votes_used"]
    paired = trials.loc[trials.reference_definition == "plurality", columns].merge(
        trials.loc[trials.reference_definition == "mean_vote", columns], on=KEYS,
        suffixes=("_plurality", "_mean_vote"), validate="one_to_one")
    if len(paired)*2 != len(trials):
        raise ValueError("each trial must have both reference definitions")
    for name in ["split_sha256", "human_comparisons_used", "human_votes_used"]:
        if not paired[f"{name}_plurality"].equals(paired[f"{name}_mean_vote"]):
            raise ValueError(f"reference definitions must share {name}")
        paired[name] = paired.pop(f"{name}_plurality")
        paired.drop(columns=f"{name}_mean_vote", inplace=True)
    for name in ["estimate", "heldout_human_reference", "absolute_error", "squared_error", "selected_power"]:
        paired[f"{name}_change"] = paired[f"{name}_mean_vote"] - paired[f"{name}_plurality"]
    return paired


def summarize_pairs(trials):
    return trials.groupby(["reference_definition", "model_a", "model_b", "labeled_fraction", "method"]).agg(
        mean_absolute_error=("absolute_error", "mean"), mean_squared_error=("squared_error", "mean"),
        mean_bias=("signed_error", "mean"), mean_selected_power=("selected_power", "mean"),
        trials=("estimate", "size")).reset_index()


def pair_method_gaps(pairs):
    gaps = pairs.pivot(index=["reference_definition", "model_a", "model_b", "labeled_fraction"],
                       columns="method", values="mean_absolute_error").reset_index()
    gaps["tuned_minus_human_mae"] = gaps.ppi_tuned - gaps.human_only
    return gaps


def make_report(summary, gaps, diagnostics, config, links):
    lines = ["# Sensitivity to the human reference", "",
             "The same public comparisons, judge labels, question splits and audit costs; two different human outcomes.", "",
             f"[Retrospective protocol]({links['protocol']}) | [Methods]({links['methods']}) | "
             f"[Data attribution]({links['license']})", "",
             "## The two targets", "",
             "- **Plurality outcome:** score the unique modal human category A/B/tie as 1/0/0.5; a tie for the most votes becomes 0.5.",
             "- **Mean vote score:** `(A votes + 0.5 * tie votes) / total votes` for each comparison, then average comparisons equally.", "",
             "Neither is latent ground truth. Mean vote scores preserve the average recorded score; the retained vote counts "
             "distinguish disagreement patterns that a mean alone cannot reveal. Both references depend on the observed annotator mix "
             "and number of votes. This is not a pooled-vote-weighted target or an estimate of inter-rater reliability. "
             "Single-vote comparisons provide no within-comparison disagreement information.", "",
             f"The snapshot contains {diagnostics['comparisons']:,} comparisons and {diagnostics['human_votes']:,} unique votes; "
             f"{diagnostics['single_vote_comparisons']:,} comparisons have one vote. "
             f"The score changes for {diagnostics['changed_scores']:,} comparisons, "
             f"with an average absolute change of {100*diagnostics['mean_absolute_score_change']:.3f} percentage points "
             "across all comparisons. `reference_changes.csv` retains every comparison, including unchanged scores.", "",
             "## Identical sampling, separate scoring", "",
             f"All {config['eligible_pairs']} eligible model pairs, {config['split_seeds']} deterministic splits per pair, "
             "fixed 25% evaluation questions, and nested audits of 20%, 40% and 60% of each pair's total questions. "
             "Every turn of a question stays together. Both definitions use exactly the original audit-reliability study's split algorithm. "
             "Each estimator sees only audit outcomes; hidden evaluation outcomes score its error afterwards.", "",
             "The judge remains a frozen categorical A/B/tie score under both definitions. "
             "Errors compare each estimator to its own realized heldout reference mean. "
             "Changing the definition can change the audit correction, selected power, and evaluation target. "
             "A smaller error against one reference does not establish that reference as better.", "",
             "| Reference | Audit questions | Method | MAE (pp) | RMSE (pp) | Mean power |",
             "|---|---:|---|---:|---:|---:|"]
    for row in summary.itertuples():
        power = "--" if pd.isna(row.mean_selected_power) else f"{row.mean_selected_power:.3f}"
        lines.append(f"| {REFERENCES[row.reference_definition]} | {row.labeled_fraction:.0%} | {NAMES[row.method]} | "
                     f"{100*row.mean_absolute_error:.3f} | {100*np.sqrt(row.mean_squared_error):.3f} | {power} |")
    if config['plots']:
        lines += ["", "![Reference definition sensitivity](reference_sensitivity.svg)"]
    lines += ["", "## Does the relative method comparison change?", "",
              "Negative gaps favor tuned PPI over human-only for that reference. "
              "These are descriptive gaps; repeated splits and model pairs share questions, so no independent-trial significance test is reported.", "",
              "| Reference | Audit questions | Tuned minus human MAE (pp) | Pairs favoring tuned PPI |",
              "|---|---:|---:|---:|"]
    for (reference, fraction), table in gaps.groupby(["reference_definition", "labeled_fraction"]):
        lines.append(f"| {REFERENCES[reference]} | {fraction:.0%} | {100*table.tuned_minus_human_mae.mean():+.3f} | "
                     f"{int((table.tuned_minus_human_mae < 0).sum())}/{len(table)} |")
    aggregate = gaps.groupby(["reference_definition", "labeled_fraction"]).tuned_minus_human_mae.mean()
    aligned_gaps = gaps.pivot(index=["model_a", "model_b", "labeled_fraction"],
                              columns="reference_definition", values="tuned_minus_human_mae")
    preference_changes = np.sign(aligned_gaps.mean_vote) != np.sign(aligned_gaps.plurality)
    lines += ["", f"Tuned PPI has lower aggregate MAE than human-only in {int((aggregate < 0).sum())}/{len(aggregate)} "
              "reference/budget combinations. "
              f"The sign of the pair-level MAE gap changes in {int(preference_changes.sum())}/{len(aligned_gaps)} "
              "matched pair/budget comparisons when the reference changes. "
              "Thus aggregate direction and pair-level sensitivity must be assessed separately; "
              "these signs are descriptive and do not imply statistical significance."]
    lines += ["", "`paired_reference_differences.csv` aligns each method/pair/seed/budget across definitions; "
              "all changes are mean-vote minus plurality. It retains changes in predictions and evaluation targets separately. "
              "`per_pair.csv` and `paired_method_gaps.csv` retain heterogeneous pair-level outcomes. "
              "The original plurality results are preserved in their earlier report.", "",
              "## Limits", "",
              "- These data contain selected older models, 80 questions and unequal annotation counts. Neither definition establishes a population-wide preference distribution.",
              "- One-way question clustering does not model shared annotator effects across questions. "
              "The derived snapshot lacks per-vote annotator identities, so this study cannot estimate those effects or resample individual annotators.",
              "- The intervals target a population mean under representative independent pools and sufficient independent clusters. "
              "Heldout-reference errors are not interval coverage tests; existing synthetic stress tests document undercoverage.",
              "- Vote counts are observed workload proxies. No new labels were acquired and no annotation-saving claim follows from this comparison.",
              "- This is a retrospective robustness analysis, with both definitions reported regardless of outcome. "
              "It does not select a preferred reference by the resulting method ranking.", "",
              "## Reproduce offline", "", "```bash", 'pip install -e ".[dev,report]"',
              "python examples/reference_sensitivity.py --output reports/reproduced/sensitivity --plot", "```", "",
              "The default uses 30 split seeds. Smaller `--seeds` runs are recorded as such in `config.json`. "
              "Input provenance, exact question manifests, package versions, source hashes and artifact hashes are included. "
              "No model API, private data or new human annotation is used.", ""]
    return "\n".join(lines)


def plot_results(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"svg.hashsalt": "judgecal-reference-sensitivity-v1", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, constrained_layout=True)
    for ax, reference in zip(axes, REFERENCES):
        for method in METHODS:
            rows = summary.loc[(summary.reference_definition == reference) & (summary.method == method)]
            ax.plot(rows.labeled_fraction*100, rows.mean_absolute_error*100, marker="o", linewidth=2,
                    color=COLORS[method], label=NAMES[method])
        ax.set(title=REFERENCES[reference], xlabel="Audit questions (% of all pair questions)", xticks=[20, 40, 60])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("MAE against the named reference (pp)")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Human-reference sensitivity on identical MT-Bench splits", fontsize=12)
    fig.savefig(output/"reference_sensitivity.png", dpi=180)
    fig.savefig(output/"reference_sensitivity.svg", metadata={"Date": None})
    svg = output/"reference_sensitivity.svg"
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines()) + b"\n")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"reports/mtbench/labels.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"reports/sensitivity")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("seeds must be positive")
    frame, source = read_verified_labels(args.input)
    output = args.output
    if output.resolve() == args.input.resolve().parent:
        raise ValueError("choose a separate output directory to preserve the input study")
    table_names = ["trials", "summary", "per_pair", "paired_reference_differences", "paired_method_gaps", "reference_changes"]
    generated = {f"{name}.csv" for name in table_names} | {
        "split_manifest.json", "config.json", "dataset.json", "reference_diagnostics.json", "REPORT.md"}
    if args.plot:
        generated.update({"reference_sensitivity.png", "reference_sensitivity.svg"})
    if output.exists():
        stale = {p.name for p in output.iterdir()} - generated - {"reproducibility.json"}
        if stale:
            raise ValueError(f"output contains stale or unrelated artifacts: {sorted(stale)}; use a fresh directory")
    trials, summary = reference_definition_sensitivity(frame, seeds=tuple(range(args.seeds)))
    # pandas propagates attrs through selections/groups. Keep the large split
    # manifest separately so table analysis does not repeatedly deep-copy it.
    split_manifest = trials.attrs["split_manifest"]
    trials.attrs = {}
    pairs, changes, paired = summarize_pairs(trials), reference_changes(frame), paired_reference_differences(trials)
    gaps = pair_method_gaps(pairs)
    diagnostics = {"comparisons": len(changes), "human_votes": int(changes.n_human_votes.sum()),
                   "single_vote_comparisons": int((changes.n_human_votes == 1).sum()),
                   "changed_scores": int((changes.score_change != 0).sum()),
                   "comparisons_with_disagreeing_votes": int(changes.votes_disagree.sum()),
                   "mean_absolute_score_change": float(changes.score_change.abs().mean()),
                   "mean_signed_score_change": float(changes.score_change.mean()),
                   "weighting": "equal comparison weights, not pooled individual votes"}
    config = {"protocol": "human-reference-sensitivity-v1", "audit_fractions": [.2, .4, .6],
              "evaluation_fraction": .25, "split_seeds": args.seeds, "plots": args.plot,
              "reference_definitions": list(REFERENCES), "min_pair_questions": 40,
              "eligible_pairs": int(trials[["model_a", "model_b"]].drop_duplicates().shape[0]),
              "source_labels_sha256": source["labels_sha256"]}
    output.mkdir(parents=True, exist_ok=True)
    for name, table in zip(table_names, [trials, summary, pairs, paired, gaps, changes]):
        table.to_csv(output/f"{name}.csv", index=False, lineterminator="\n")
    write_json(output/"split_manifest.json", split_manifest, compact_records=True)
    for name, value in [("config", config), ("dataset", source), ("reference_diagnostics", diagnostics)]:
        write_json(output/f"{name}.json", value)
    links = {name: os.path.relpath(ROOT/path, output).replace("\\", "/") for name, path in {
        "protocol": "docs/REFERENCE_SENSITIVITY.md", "methods": "docs/METHODS.md",
        "license": "reports/mtbench/DATA_LICENSE.md"}.items()}
    (output/"REPORT.md").write_bytes(make_report(summary, gaps, diagnostics, config, links).encode("utf-8"))
    if args.plot:
        plot_results(summary, output)
    sources = sorted((ROOT/"src/judgecal").glob("*.py")) + [Path(__file__), ROOT/"examples/research_study.py",
                ROOT/"docs/REFERENCE_SENSITIVITY.md", ROOT/"docs/METHODS.md", ROOT/"pyproject.toml"]
    manifest = {"python": platform.python_version(), "platform": platform.system(),
                "packages": {name: version(name) for name in ["judgecal", "numpy", "scipy", "pandas"]},
                "source_text_normalization": "CRLF to LF before SHA-256 (matches Git text attributes)",
                "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"):
                                  hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sources},
                "input_labels_sha256": sha256(args.input),
                "artifacts_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.name in generated}}
    if args.plot:
        manifest["packages"]["matplotlib"] = version("matplotlib")
    write_json(output/"reproducibility.json", manifest)
    print(summary.to_string(index=False))
    print(f"Wrote {len(trials)} method/reference trials and {len(paired)} paired reference comparisons.")


if __name__ == "__main__":
    main()
