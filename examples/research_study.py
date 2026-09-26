"""Reproduce the fixed-target MT-Bench study and known-truth stress tests.

No network access or model API is used. See docs/RESEARCH_PROTOCOL.md.
"""
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

from judgecal.benchmark import fixed_evaluation_label_budget
from judgecal.simulation import run_simulation_study


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("raw_judge", "human_only", "ppi", "ppi_tuned")
NAMES = {"raw_judge": "Raw judge", "human_only": "Human audit",
         "ppi": "PPI (power 1)", "ppi_tuned": "Tuned PPI"}
COLORS = {"raw_judge": "#8a5c32", "human_only": "#525a66",
          "ppi": "#3673b5", "ppi_tuned": "#00876c"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, compact_records=False):
    if compact_records:
        content = "[\n" + ",\n".join(json.dumps(record, sort_keys=True, allow_nan=False,
                                              separators=(",", ":")) for record in value) + "\n]"
    else:
        content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    Path(path).write_bytes((content + "\n").encode("utf-8"))


def read_verified_labels(path):
    """Bind the named experiment to the committed public-label revision."""
    path = Path(path)
    sidecar = path.with_name("dataset.json")
    source = json.loads(sidecar.read_text(encoding="utf-8"))
    if source.get("dataset_id") != "lmsys/mt_bench_human_judgments":
        raise ValueError("the research study requires the attributed MT-Bench labels")
    if source.get("dataset_revision") != "f7d2896d2cc5d80f8b55c2bbc722613555233c25":
        raise ValueError("the research study requires the pinned dataset revision")
    if source.get("labels_sha256") != sha256(path):
        raise ValueError("input label checksum does not match the provenance sidecar")
    return pd.read_csv(path), source


def summarize_pairs(trials):
    result = trials.groupby(["model_a", "model_b", "labeled_fraction", "method"]).agg(
        mean_absolute_error=("absolute_error", "mean"),
        mean_squared_error=("squared_error", "mean"),
        mean_bias=("signed_error", "mean"),
        mean_selected_power=("selected_power", "mean"),
        trials=("estimate", "size"),
    ).reset_index()
    result["rmse"] = np.sqrt(result.mean_squared_error)
    return result


def make_report(summary, pairs, simulation_summary, config, links):
    lines = ["# Human-audit reliability study", "",
             "A fixed-target MT-Bench case study and known-truth stress tests.", "",
             "## Question and scope", "",
             "When does an imperfect judge help estimate human preference at a fixed audit budget? "
             "This study implements established PPI/PPI++ mean correction and examines its failure modes. "
             "It does not introduce a new estimator or establish a general label-saving guarantee.", "",
             f"[Versioned analysis plan]({links['protocol']}) | "
             f"[Methods]({links['methods']}) | [Data attribution]({links['license']})", "",
             "## Fixed-target MT-Bench experiment", "",
             f"All {config['eligible_pairs']} eligible model pairs; "
             f"{config['split_seeds']} deterministic splits per pair. "
             "A fixed 25% of each pair's questions form evaluation; nested audits use 20%, 40% and 60% "
             "of its original questions. Every turn of a question stays together. "
             "The evaluation human reference and raw judge estimate are identical across budgets within a split.", "",
             "Errors compare to the realized heldout human plurality mean. "
             "They are not population-confidence-interval coverage measurements. "
             "Pairs and repeated splits share questions; their rows are not independent replications.", "",
             "| Audit questions | Method | MAE (pp) | RMSE (pp) | Mean power |",
             "|---:|---|---:|---:|---:|"]
    for row in summary.itertuples():
        power = "--" if pd.isna(row.mean_selected_power) else f"{row.mean_selected_power:.3f}"
        lines.append(f"| {row.labeled_fraction:.0%} | {NAMES[row.method]} | "
                     f"{100*row.mean_absolute_error:.3f} | {100*np.sqrt(row.mean_squared_error):.3f} | {power} |")
    pivot = summary.pivot(index="labeled_fraction", columns="method", values="mean_absolute_error")
    n = len(pivot)
    vs_human = int((pivot.ppi_tuned < pivot.human_only).sum())
    vs_ppi = int((pivot.ppi_tuned < pivot.ppi).sum())
    lines += ["", f"Tuned PPI has lower aggregate MAE than human-only at {vs_human}/{n} budgets "
              f"and lower MAE than coefficient-one PPI at {vs_ppi}/{n} budgets. "
              "These are descriptive comparisons on this fixed dataset; tuning need not improve every split or model pair.", "",
              "| Audit questions | Pairs where tuned PPI has lower mean MAE than human-only |",
              "|---:|---:|"]
    for fraction, group in pairs.groupby("labeled_fraction"):
        table = group.pivot(index=["model_a", "model_b"], columns="method", values="mean_absolute_error")
        count = int((table.ppi_tuned < table.human_only).sum())
        lines.append(f"| {fraction:.0%} | {count}/{len(table)} |")
    if config.get("plots"):
        lines += ["", "![Fixed-target budget results](fixed_target_budget.svg)"]
    lines += ["", "The full `trials.csv`, `per_pair.csv` and `split_manifest.json` retain all outcomes "
              "and sampling choices. `human_comparisons_used` and `human_votes_used` count the audited "
              "comparison labels and underlying public votes; these are workload proxies, not a priced annotation study.", "",
              "## Known-truth stress tests", "",
              f"Each scenario uses {config['simulation_repetitions']:,} independent replications "
              f"from a fixed master seed ({config['simulation_seed']}). "
              "See `simulation_config.json` for exact generation mechanisms, population truths and validity scopes. "
              "All four methods share the same draws. `simulation_trials.csv` retains every replication.", ""]
    lines += ["| Scenario | Method | Bias | RMSE | Width | Human-truth coverage | MC SE |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in simulation_summary.itertuples():
        lines.append(f"| {row.scenario} | {NAMES[row.method]} | {row.bias:.4f} | "
                     f"{row.rmse:.4f} | {row.mean_width:.4f} | {row.coverage:.3f} | {row.coverage_mcse:.3f} |")
    if config.get("plots"):
        lines += ["", "![Known-truth simulation](simulation.svg)"]
    lines += ["", "Additional Monte Carlo errors are in "
              "`simulation_summary.csv`. Coverage uses known target human truth; the raw interval "
              "estimates the judge-positive rate, so its human-truth containment is a bias diagnostic. "
              "Native-parameter containment is also retained separately.", "",
              "Prevalence shift and conditional-error shift are separate, deliberate assumption failures. "
              "Tuning controls estimated variance; it does not fix audit-to-target bias. "
              "Few independent clusters and fitted tuning weights can impair finite-sample coverage. "
              "No failed or zero-width trial is discarded.", "",
              "## What can and cannot be concluded", "",
              "- The tuned method is an implementation of established mean-estimation ideas, with an explicitly "
              "documented one-way cluster sandwich adaptation.",
              "- Human plurality is a reference, not infallible ground truth. MT-Bench has only 80 selected "
              "questions and older model outputs; generalization to new judges and tasks is untested.",
              "- These budget curves hold the evaluation set fixed. Historical complementary-pool results "
              "remain separately available in `reports/mtbench` and are not pooled with this study.",
              "- Asymptotic normal intervals are neither distribution-free nor guaranteed at small sample sizes. "
              "Synthetic coverage is evidence only for the specified generating process.",
              "- No new human annotation, model API calls, private data or externally priced compute were used.", "",
              "## Reproduce offline", "", "```bash", 'pip install -e ".[dev,report]"',
              "python examples/research_study.py --output reports/reproduced/research --plot", "```", "",
              "Default arguments reproduce this complete protocol. `--repetitions` and `--seeds` support "
              "explicitly smaller smoke runs and are recorded in `config.json`; do not present those as the full study. "
              "Source checksums, package versions and artifact hashes are in `reproducibility.json`. "
              "The input CSV is checked against the pinned provenance before inference.", ""]
    return "\n".join(lines)


def plot_budget(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"svg.hashsalt": "judgecal-research-v1", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for method in METHODS:
        table = summary[summary.method == method]
        for ax, measure in zip(axes, ["mean_absolute_error", "mean_squared_error"]):
            values = table[measure] if measure == "mean_absolute_error" else np.sqrt(table[measure])
            ax.plot(table.labeled_fraction*100, values*100, marker="o", linewidth=2,
                    label=NAMES[method], color=COLORS[method])
    for ax, title in zip(axes, ["Mean absolute error", "Root mean squared error"]):
        ax.set(xlabel="Audit questions (% of all pair questions)", ylabel=f"{title} (pp)",
               title=title, xticks=[20, 40, 60])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("MT-Bench: a fixed evaluation pool across human-audit budgets", fontsize=12)
    fig.savefig(output/"fixed_target_budget.png", dpi=180)
    fig.savefig(output/"fixed_target_budget.svg", metadata={"Date": None})
    svg = output/"fixed_target_budget.svg"
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines()) + b"\n")
    plt.close(fig)


def plot_simulation(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scenarios = list(summary.scenario.drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    positions = np.arange(len(scenarios))
    offsets = {"raw_judge": -.24, "human_only": -.08, "ppi": .08, "ppi_tuned": .24}
    for method in METHODS:
        group = summary[summary.method == method].set_index("scenario").loc[scenarios]
        y = positions + offsets[method]
        axes[0].plot(group.rmse, y, "o", label=NAMES[method], color=COLORS[method], markersize=5)
        if method != "raw_judge":
            axes[1].errorbar(group.coverage, y, xerr=2*group.coverage_mcse,
                             fmt="o", color=COLORS[method], markersize=5, capsize=2)
    axes[1].axvline(.95, color="#555555", linewidth=1, linestyle="--")
    for ax in axes:
        ax.set_yticks(positions, [name.replace("_", " ") for name in scenarios])
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.2)
    axes[0].set(xlabel="RMSE against target human truth", title="Point-estimation error")
    axes[1].set(xlabel="Coverage (bars: ±2 Monte Carlo SE)", xlim=(-.03, 1.03),
                title="Human-target intervals: nominal 95%")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Known-truth stress tests: variance tuning does not repair distribution shift", fontsize=12)
    fig.savefig(output/"simulation.png", dpi=180)
    fig.savefig(output/"simulation.svg", metadata={"Date": None})
    svg = output/"simulation.svg"
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines()) + b"\n")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"reports/mtbench/labels.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"reports/research")
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds < 1 or args.repetitions < 2:
        parser.error("seeds must be positive and repetitions at least 2")
    frame, source = read_verified_labels(args.input)
    output = args.output
    if output.resolve() == args.input.resolve().parent:
        raise ValueError("choose a separate output directory to preserve the input study")
    tables = ["trials", "summary", "per_pair", "simulation_trials", "simulation_summary"]
    generated = {f"{name}.csv" for name in tables} | {
        "split_manifest.json", "config.json", "dataset.json", "simulation_config.json", "REPORT.md"}
    if args.plot:
        generated.update({"fixed_target_budget.png", "fixed_target_budget.svg", "simulation.png", "simulation.svg"})
    if output.exists():
        stale = {p.name for p in output.iterdir()} - generated - {"reproducibility.json"}
        if stale:
            raise ValueError(f"output contains stale or unrelated artifacts: {sorted(stale)}; use a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    trials, summary = fixed_evaluation_label_budget(frame, seeds=tuple(range(args.seeds)))
    pairs = summarize_pairs(trials)
    sim_trials, sim_summary, sim_config = run_simulation_study(args.repetitions, seed=2026)
    config = {"protocol": "human-audit-reliability-v1", "audit_fractions": [.2, .4, .6],
              "evaluation_fraction": .25, "split_seeds": args.seeds,
              "simulation_repetitions": args.repetitions, "simulation_seed": 2026,
              "plots": args.plot,
              "eligible_pairs": int(trials[["model_a", "model_b"]].drop_duplicates().shape[0]),
              "min_pair_questions": 40, "source_labels_sha256": source["labels_sha256"]}
    for name, table in [("trials", trials), ("summary", summary), ("per_pair", pairs),
                        ("simulation_trials", sim_trials), ("simulation_summary", sim_summary)]:
        table.to_csv(output/f"{name}.csv", index=False, lineterminator="\n")
    write_json(output/"split_manifest.json", trials.attrs["split_manifest"], compact_records=True)
    write_json(output/"config.json", config)
    write_json(output/"dataset.json", source)
    write_json(output/"simulation_config.json", sim_config)
    links = {name: os.path.relpath(ROOT/path, output).replace("\\", "/") for name, path in {
        "protocol": "docs/RESEARCH_PROTOCOL.md", "methods": "docs/METHODS.md",
        "license": "reports/mtbench/DATA_LICENSE.md"}.items()}
    (output/"REPORT.md").write_bytes(make_report(summary, pairs, sim_summary, config, links).encode("utf-8"))
    if args.plot:
        plot_budget(summary, output)
        plot_simulation(sim_summary, output)
    sources = sorted((ROOT/"src/judgecal").glob("*.py")) + [Path(__file__), ROOT/"docs/RESEARCH_PROTOCOL.md", ROOT/"docs/METHODS.md"]
    manifest = {"python": platform.python_version(), "platform": platform.system(),
                "packages": {name: version(name) for name in ["judgecal", "numpy", "scipy", "pandas"]},
                "source_text_normalization": "CRLF to LF before SHA-256 (matches Git text attributes)",
                "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"):
                                  hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                  for p in sources},
                "input_labels_sha256": sha256(args.input),
                "artifacts_sha256": {p.name: sha256(p) for p in sorted(output.iterdir())
                                      if p.is_file() and p.name in generated}}
    if args.plot:
        manifest["packages"]["matplotlib"] = version("matplotlib")
    write_json(output/"reproducibility.json", manifest)
    print(summary.to_string(index=False))
    print(f"Wrote {len(trials)} benchmark and {len(sim_trials)} simulation trial rows.")


if __name__ == "__main__":
    main()
