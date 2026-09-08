"""Run a frozen label-budget benchmark using public labels, with no model calls.

Online: python examples/benchmark_mtbench.py --plot
Offline: python examples/benchmark_mtbench.py --input reports/mtbench/labels.csv
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from judgecal import prediction_powered_win_rate
from judgecal.benchmark import evaluate_label_budget
from judgecal.data import load_mt_bench


def json_write(path, value, compact_records=False):
    options = dict(sort_keys=True, allow_nan=False, default=lambda x: x.item())
    if compact_records:
        content = "[\n" + ",\n".join(
            json.dumps(record, separators=(",", ":"), **options) for record in value
        ) + "\n]"
    else:
        content = json.dumps(value, indent=2, **options)
    path.write_text(content + "\n", encoding="utf-8")


def synthetic_check(repetitions=1000, seed=2026, sensitivity=0.90, specificity=0.60):
    """Known-truth iid experiment; independent labeled and unlabeled samples."""
    rng = np.random.default_rng(seed)
    truth = 0.6
    n_labeled, n_unlabeled = 200, 2000
    records = []
    for _ in range(repetitions):
        human = rng.random(n_labeled + n_unlabeled) < truth
        prediction = rng.random(human.size) < np.where(human, sensitivity, 1 - specificity)
        h, j = np.where(human, "A", "B"), np.where(prediction, "A", "B")
        result = prediction_powered_win_rate(j[:n_labeled], h[:n_labeled], j[n_labeled:])
        records.append({"raw": result.raw_rate, "human_only": result.human_only_rate,
                        "ppi": result.point, "width": result.interval.high - result.interval.low,
                        "covered": result.interval.low <= truth <= result.interval.high})
    table = pd.DataFrame(records)
    return {
        "seed": seed, "repetitions": repetitions, "true_human_win_rate": truth,
        "sensitivity": sensitivity, "specificity": specificity,
        "n_labeled": n_labeled, "n_unlabeled": n_unlabeled,
        "bias": {m: float(table[m].mean() - truth) for m in ["raw", "human_only", "ppi"]},
        "rmse": {m: float(np.sqrt(np.mean((table[m] - truth) ** 2))) for m in ["raw", "human_only", "ppi"]},
        "ppi_empirical_95pct_coverage": float(table.covered.mean()),
        "coverage_monte_carlo_se": float(np.sqrt(table.covered.mean() * (1 - table.covered.mean()) / repetitions)),
        "ppi_mean_interval_width": float(table.width.mean()),
        "scope": "One specified synthetic iid setting; not a distribution-free or small-sample guarantee.",
    }


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["svg.hashsalt"] = "judgecal-mtbench-v0.2"
    fig, ax = plt.subplots(figsize=(7, 4.3), constrained_layout=True)
    names = {"raw_judge": "Raw GPT-4 judge", "human_only": "Human audit only", "ppi": "Prediction-powered"}
    for method, group in summary.groupby("method"):
        ax.plot(group.labeled_fraction * 100, group.mean_absolute_error * 100,
                marker="o", linewidth=2, label=names[method])
    ax.set(xlabel="Questions allocated to the human audit (%)",
           ylabel="Mean absolute error (percentage points)",
           title="MT-Bench: error against hidden human judgments")
    ax.set_xticks([20, 40, 60])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output / "label_budget.svg", metadata={"Date": None})
    svg = output / "label_budget.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8")
    fig.savefig(output / "label_budget.png", dpi=180)
    plt.close(fig)


def markdown_report(summary, trials, source, simulation):
    lines = ["# MT-Bench human-label-budget benchmark", "",
             "A reproducible offline-label experiment. No new LLM calls or human annotations were purchased.", "",
             "## Protocol", "",
             f"- Dataset: `{source.get('dataset_id', 'local CSV')}`, revision `{source.get('dataset_revision', 'unspecified')}`.",
             f"- Aligned comparisons: {source['aligned_comparisons']}; unique questions: {source['unique_questions']}; eligible model pairs: {trials[['model_a', 'model_b']].drop_duplicates().shape[0]}.",
             "- Every pair with at least 40 questions is included; no pair is selected by its result.",
             "- Frozen budgets: 20%, 40%, 60% of questions; seeds 0 through 29. Within each model pair, all turns of a question stay together. Separate pairs can use different partitions.",
             "- A is the alphabetically first model in each pair; ties receive half a win.",
             "- Human evaluation labels are withheld from inference and used only to score estimates afterward.",
             "- PPI uses the untuned, coefficient-one mean correction. Both uncertainty terms use question-cluster variance estimates.",
             "- Each pair/seed receives equal weight in the summary. Repeated splits are correlated and describe this dataset, not independent replications.", "",
             "## Results", "",
             "Error is measured against the realized human mean on the held-out questions. It is not a population-interval coverage test.", "",
             "| Human-audit questions | Estimator | Mean absolute error (pp) | Pair/split trials |",
             "|---:|---|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.labeled_fraction:.0%} | {r.method} | {r.mean_absolute_error * 100:.3f} | {r.trials} |")
    pivot = summary.pivot(index="labeled_fraction", columns="method", values="mean_absolute_error")
    better_raw = int((pivot["ppi"] < pivot["raw_judge"]).sum())
    better_human = int((pivot["ppi"] < pivot["human_only"]).sum())
    lines += ["", f"PPI has lower aggregate error than the raw judge at {better_raw}/{len(pivot)} budgets and lower error than human-only at {better_human}/{len(pivot)} budgets. These results do not establish annotation savings.", "",
              "Per-pair results, every trial and explicit question split IDs are committed alongside this report.", "",
              "## Known-truth simulation", "",
              "Each iid scenario uses 1,000 replications, seed 2026, a true human win rate of 0.6, 200 audited labels and 2,000 prediction-only labels. The original moderate-judge scenario is retained; strong and uninformative judges illustrate dependence on judge quality.", "",
              "| Scenario (sensitivity, specificity) | Raw RMSE | Human-only RMSE | PPI RMSE | PPI 95% coverage | Monte Carlo SE |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, case in simulation["scenarios"].items():
        lines.append(f"| {name} ({case['sensitivity']}, {case['specificity']}) | {case['rmse']['raw']:.4f} | {case['rmse']['human_only']:.4f} | {case['rmse']['ppi']:.4f} | {case['ppi_empirical_95pct_coverage']:.3f} | {case['coverage_monte_carlo_se']:.3f} |")
    lines += ["", "The original moderate-judge case covers 92.6%, below the nominal 95% in this finite-sample experiment. These are specified synthetic models, not evidence of coverage on MT-Bench or under distribution shift. Undercoverage and cases worse than human-only are retained. Normal intervals are asymptotic.", "",
              "## Limits and interpretation", "",
              "- MT-Bench contains 80 selected questions and older model outputs. This is a small case study, not a current model leaderboard.",
              "- Human plurality labels are a reference, not infallible truth. Tied vote counts and judge order inconsistencies are retained as ties. The estimand averages aggregated comparison labels, observation-weighted across retained turns, not individual votes or equally weighted questions.",
              "- Randomly sampled, disjoint audit/evaluation questions must represent the same population. Biased auditing, changing judges or distribution shift can invalidate the correction.",
              "- Cluster-normal intervals are asymptotic; some audit budgets have few questions. Estimates and intervals are intentionally not clipped to [0, 1].",
              "- Original PPI can be less efficient than human-only estimation when the judge is weak. No budget or pair was tuned to hide such cases.",
              "- The report compares point-estimation errors, not paid annotation cost or a guaranteed label-saving percentage.", "",
              "## Reproduce", "", "```bash", 'pip install -e ".[data,dev,report]"',
              "python examples/benchmark_mtbench.py --plot",
              "# Reuse committed labels without network access:",
              "python examples/benchmark_mtbench.py --input reports/mtbench/labels.csv --output /tmp/judgecal-report",
              "```", "", "See `dataset.json` for the pinned source and transformation counts, `environment.json` for package versions, and `DATA_LICENSE.md` for attribution.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Use aligned labels CSV without downloading")
    parser.add_argument("--output", type=Path, default=Path("reports/mtbench"))
    parser.add_argument("--plot", action="store_true", help="Requires the report extra")
    args = parser.parse_args()
    if args.input:
        frame = pd.read_csv(args.input)
        sidecar = args.input.with_name("dataset.json")
        if not sidecar.exists():
            raise ValueError("the named MT-Bench report requires a dataset.json provenance sidecar")
        source = json.loads(sidecar.read_text(encoding="utf-8"))
        if source.get("dataset_id") != "lmsys/mt_bench_human_judgments" or source.get("dataset_revision") != "f7d2896d2cc5d80f8b55c2bbc722613555233c25":
            raise ValueError("the report requires the pinned MT-Bench dataset revision")
        actual = hashlib.sha256(args.input.read_bytes()).hexdigest()
        if source.get("labels_sha256") != actual:
            raise ValueError("input labels do not match their provenance checksum")
    else:
        frame = load_mt_bench()
        source = dict(frame.attrs)
    args.output.mkdir(parents=True, exist_ok=True)
    allowed = ["question_id", "turn", "model_a", "model_b", "human_winner", "n_human_votes",
               "gpt4_winner", "n_human_a", "n_human_b", "n_human_ties", "gpt4_raw_winner",
               "gpt4_inconsistent_order", "gpt4_input_reversed"]
    frame = frame[[c for c in allowed if c in frame]].sort_values(["question_id", "turn", "model_a", "model_b"]).reset_index(drop=True)
    labels = args.output / "labels.csv"
    frame.to_csv(labels, index=False, lineterminator="\n")
    source.update(aligned_comparisons=len(frame), unique_questions=frame.question_id.nunique(),
                  labels_sha256=hashlib.sha256(labels.read_bytes()).hexdigest())
    source["contains"] = "Derived public labels and IDs only; no prompts, responses or private user material."
    if "gpt4_inconsistent_order" in frame:
        source["aligned_order_inconsistent_comparisons"] = int(frame.gpt4_inconsistent_order.sum())
    json_write(args.output / "dataset.json", source)
    trials, summary = evaluate_label_budget(frame)
    json_write(args.output / "split_manifest.json", trials.attrs["split_manifest"], compact_records=True)
    trials.to_csv(args.output / "trials.csv", index=False, lineterminator="\n")
    summary.to_csv(args.output / "summary.csv", index=False, lineterminator="\n")
    pair_summary = trials.groupby(["model_a", "model_b", "labeled_fraction", "method"]).absolute_error.mean().reset_index()
    pair_summary.to_csv(args.output / "per_pair.csv", index=False, lineterminator="\n")
    simulation = {"scenarios": {
        "moderate": synthetic_check(),
        "strong": synthetic_check(sensitivity=0.95, specificity=0.90),
        "uninformative": synthetic_check(sensitivity=0.60, specificity=0.40),
    }}
    json_write(args.output / "simulation.json", simulation)
    environment = {p: version(p) for p in ["numpy", "scipy", "pandas", "judgecal"]}
    environment["python"] = platform.python_version()
    for optional in ["datasets", "matplotlib"]:
        try:
            environment[optional] = version(optional)
        except PackageNotFoundError:
            environment[optional] = "not installed"
    json_write(args.output / "environment.json", environment)
    (args.output / "REPORT.md").write_text(markdown_report(summary, trials, source, simulation), encoding="utf-8")
    if args.plot:
        plot_summary(summary, args.output)
    print(summary.to_string(index=False))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
