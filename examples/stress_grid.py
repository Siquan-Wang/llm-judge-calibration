"""Run the retrospective factorial and dependence-ablation follow-up offline."""
from __future__ import annotations

import argparse
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from judgecal.stress_grid import run_stress_grid


ROOT = Path(__file__).resolve().parents[1]
NAMES = {"raw_judge": "Raw judge", "human_only": "Human audit", "ppi": "PPI (power 1)", "ppi_tuned": "Tuned PPI"}
COLORS = {"human_only": "#525a66", "ppi": "#3673b5", "ppi_tuned": "#00876c"}


def write_json(path, value):
    path.write_bytes((json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode())


def save_plot(fig, output, name):
    fig.savefig(output/f"{name}.png", dpi=180)
    fig.savefig(output/f"{name}.svg", metadata={"Date": None})
    svg = output/f"{name}.svg"
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines())+b"\n")


def make_plots(summary, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    matplotlib.rcParams.update({"svg.hashsalt": "judgecal-grid-v2", "font.size": 10})
    iid = summary[summary.inference_mode == "iid"].copy()
    # Metadata are exported beside every scenario by the simulation module.
    info = {s["name"]: s for s in config["scenarios"]}
    iid["prevalence"] = iid.scenario.map(lambda name: info[name]["audit_prevalence"])
    iid["judge_quality"] = iid.scenario.map(lambda name: info[name]["judge_quality"])
    qualities = ["strong", "moderate", "uninformative", "anticorrelated"]
    sizes = [20, 50, 200]
    baseline = iid[iid.method == "human_only"].set_index("scenario").rmse
    tuned = iid[iid.method == "ppi_tuned"].copy()
    tuned["rmse_ratio"] = tuned.rmse / tuned.scenario.map(baseline)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7), constrained_layout=True)
    for column, prevalence in enumerate([.2, .5, .8]):
        group = tuned[tuned.prevalence == prevalence]
        for row, measure in enumerate(["rmse_ratio", "coverage"]):
            table = group.pivot(index="judge_quality", columns="audit_units", values=measure).reindex(index=qualities, columns=sizes)
            values = table.to_numpy()
            kwargs = {"cmap": "RdBu_r", "norm": TwoSlopeNorm(vmin=.4, vcenter=1., vmax=1.2)} if row == 0 else {"cmap": "YlGnBu", "vmin": .75, "vmax": 1.}
            axes[row, column].imshow(values, aspect="auto", **kwargs)
            for y in range(4):
                for x in range(3):
                    axes[row, column].text(x, y, f"{values[y,x]:.3f}", ha="center", va="center",
                                          color="white" if (row == 1 and values[y,x] > .91) else "black")
            axes[row, column].set(xticks=range(3), xticklabels=sizes,
                                  yticks=range(4), yticklabels=qualities,
                                  xlabel="Independent audited labels",
                                  title=f"Human prevalence {prevalence:.1f}")
    axes[0, 0].set_ylabel("Tuned / human-only RMSE\n(lower than 1 is better)")
    axes[1, 0].set_ylabel("Tuned 95% interval coverage")
    fig.suptitle("Power tuning across 36 specified iid settings", fontsize=13)
    save_plot(fig, output, "factorial_grid")
    plt.close(fig)
    cluster = summary[summary.inference_mode != "iid"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    for ax, mode, title in zip(axes, ["cluster", "naive_iid"], ["Correct question-cluster variance", "Misspecified independent-row variance"]):
        for method in COLORS:
            table = cluster[(cluster.inference_mode == mode) & (cluster.method == method)].sort_values("audit_units")
            ax.errorbar(table.audit_units, table.coverage, yerr=2*table.coverage_mcse,
                        marker="o", color=COLORS[method], label=NAMES[method], capsize=3)
        ax.axhline(.95, color="#666666", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="Independent audit questions (4 repeated turns each)",
               ylabel="Coverage of target human mean", ylim=(0, 1.02), xticks=[8, 30, 100])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Same generated data, different independence assumptions", fontsize=13)
    save_plot(fig, output, "dependence_ablation")
    plt.close(fig)


def make_report(summary, config):
    iid = summary[summary.inference_mode == "iid"]
    pivot = iid.pivot(index="scenario", columns="method", values="rmse")
    n = len(pivot)
    better = int((pivot.ppi_tuned < pivot.human_only).sum())
    lines = ["# Follow-up factorial and dependence study", "",
             "This retrospective follow-up broadens the first reliability study; "
             "it is a specified stress grid, not a representative sample of real deployments.", "",
             "## Factorial settings", "",
             f"Actual run: {config['repetitions']:,} replications per setting; master seed {config['seed']}.", "",
             "Human prevalence 0.2/0.5/0.8; audit size 20/50/200; target pool ten times "
             "the audit size; four strong/moderate/uninformative/anticorrelated judges. "
             "All four estimators receive identical draws. Every cell and replication is retained.", "",
             f"Tuned PPI has lower observed RMSE than human-only in {better}/{n} iid cells. "
             "This count is descriptive of this chosen grid, not a probability of improvement in deployment. "
             "Paired Monte Carlo errors and per-cell results are retained in `summary.csv`.", "",
             "| Scenario | Human RMSE | PPI RMSE | Tuned RMSE | Tuned-human RMSE difference | Paired MC SE | Tuned coverage |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for scenario, row in pivot.iterrows():
        tuned = iid[(iid.scenario == scenario) & (iid.method == "ppi_tuned")].iloc[0]
        lines.append(f"| {scenario} | {row.human_only:.4f} | {row.ppi:.4f} | {row.ppi_tuned:.4f} | "
                     f"{tuned.rmse_difference_vs_human:.4f} | {tuned.rmse_difference_vs_human_mcse:.4f} | {tuned.coverage:.3f} |")
    if config.get("plots"):
        lines += ["", "![Factorial results](factorial_grid.svg)"]
    lines += ["", "## Dependence ablation", "",
              "Four exact repeated turns per independent question. The correct and naive analyses "
              "share every generated label. Treating rows as independent is deliberately misspecified. "
              "Few clusters can impair even correctly grouped asymptotic intervals.", "",
              "| Questions | Variance assumption | Method | Coverage | MC SE | Mean width |",
              "|---:|---|---|---:|---:|---:|"]
    for row in summary[summary.inference_mode != "iid"].itertuples():
        lines.append(f"| {row.audit_units} | {row.inference_mode} | {NAMES[row.method]} | "
                     f"{row.coverage:.3f} | {row.coverage_mcse:.3f} | {row.mean_width:.4f} |")
    if config.get("plots"):
        lines += ["", "![Dependence ablation](dependence_ablation.svg)"]
    lines += ["", "## Interpretation", "",
              "- Tuning minimizes fitted variance, not finite-sample risk. Small audit samples "
              "and near-boundary human rates remain difficult cases.",
              "- The bounded coefficient discards negative covariance; it does not invert anticorrelated judges.",
              "- Oracle powers are generator diagnostics only. No oracle enters the fitted-method results.",
              "- Method differences are paired across identical replications. RMSE-difference MCSE uses "
              "a delta approximation; it is not a guarantee or a multiple-comparison-adjusted claim.",
              "- Raw judge intervals target their own positive rate. Their human-truth containment is a bias diagnostic.",
              "- Simulation cells do not establish accuracy or annotation savings on real LLM evaluation data.", "",
              "## Reproduce", "", "```bash", 'pip install -e ".[dev,report]"',
              "python examples/stress_grid.py --output reports/reproduced/stress --plot", "```", "",
              "`config.json` records repetitions, seed and every generating mechanism. "
              "`trials.csv.gz` is deterministic gzip containing every trial; `reproducibility.json` "
              "binds source and artifact hashes. Defaults use 500 replications and master seed 2027. "
              "A smaller `--repetitions` value is an explicitly recorded smoke run.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reports/stress")
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    output = args.output
    artifacts = {"trials.csv.gz", "summary.csv", "config.json", "REPORT.md"}
    if args.plot:
        artifacts |= {"factorial_grid.png", "factorial_grid.svg", "dependence_ablation.png", "dependence_ablation.svg"}
    if output.exists() and {p.name for p in output.iterdir()} - artifacts - {"reproducibility.json"}:
        raise ValueError("output contains stale or unrelated artifacts; use a fresh directory")
    trials, summary, config = run_stress_grid(args.repetitions, args.seed)
    config["plots"] = args.plot
    output.mkdir(parents=True, exist_ok=True)
    with (output/"trials.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(trials.to_csv(index=False, lineterminator="\n").encode())
    summary.to_csv(output/"summary.csv", index=False, lineterminator="\n")
    write_json(output/"config.json", config)
    (output/"REPORT.md").write_bytes(make_report(summary, config).encode())
    if args.plot:
        make_plots(summary, config, output)
    files = sorted((ROOT/"src/judgecal").glob("*.py")) + [Path(__file__), ROOT/"docs/STRESS_PROTOCOL.md"]
    manifest = {"python": platform.python_version(), "platform": platform.system(),
                "packages": {p: version(p) for p in ["judgecal", "numpy", "scipy", "pandas"]},
                "source_text_normalization": "CRLF to LF before SHA-256",
                "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest() for p in files},
                "artifacts_sha256": {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in sorted(artifacts)}}
    if args.plot:
        manifest["packages"]["matplotlib"] = version("matplotlib")
    write_json(output/"reproducibility.json", manifest)
    print(f"Wrote {len(trials)} trial rows and {len(summary)} method/scenario summaries.")


if __name__ == "__main__":
    main()
