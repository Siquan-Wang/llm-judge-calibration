"""Reproduce iid finite-sample coverage/width tradeoffs and assumption failures."""
from __future__ import annotations

import argparse
import gzip
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import platform

import numpy as np

from judgecal.finite_study import run_finite_sample_study
from research_study import ROOT, sha256, write_json


NAMES = {"human_normal": "Human normal", "ppi_normal": "PPI normal",
         "ppi_tuned_normal": "Tuned PPI normal", "human_hoeffding": "Human Hoeffding",
         "ppi_hoeffding": "PPI Hoeffding", "human_eb": "Human EB", "ppi_eb": "PPI EB",
         "ppi_eb_grid": "Grid PPI EB"}
PLOT_METHODS = ("human_normal", "ppi_tuned_normal", "human_hoeffding", "human_eb", "ppi_eb", "ppi_eb_grid")
COLORS = dict(zip(PLOT_METHODS, ["#737373", "#00876c", "#111111", "#c98120", "#377eb8", "#9c4e9b"]))


def iid_plot_data(summary, config):
    """Restrict the main curves to iid scenarios, using recorded profiles."""
    table = summary.loc[summary.scenario_family == "iid"].copy()
    scenarios = {row["name"]: row for row in config["scenarios"]}
    table["profile"] = table.scenario.map(lambda name: scenarios[name]["profile"])
    return table


def make_report(summary, config, protocol_link):
    iid = summary.loc[summary.scenario_family == "iid"]
    finite = iid.loc[iid.theorem_applicable]
    comparison = iid.pivot(index="scenario", columns="method", values="mean_width")
    narrower = int((comparison.ppi_eb_grid < comparison.human_hoeffding).sum())
    boundary = iid.loc[(iid.profile == "rare_strong") & (iid.n_labeled == 20)].set_index("method")
    tuned, human = boundary.loc["ppi_tuned_normal"], boundary.loc["human_normal"]
    hoeffding, grid = boundary.loc["human_hoeffding"], boundary.loc["ppi_eb_grid"]
    lines = ["# Finite-sample coverage and the cost of conservative intervals", "",
             "Established bounded-mean inequalities applied to prediction-powered inference, "
             "with all simulation outcomes retained.", "",
             f"[Retrospective protocol and derivation]({protocol_link})", "",
             "## Question and scope", "",
             "Earlier experiments exposed small-audit undercoverage for normal PPI intervals. "
             "This follow-up asks how finite-sample bounded-score intervals trade width for coverage. "
             "It implements established Hoeffding and Maurer–Pontil empirical Bernstein (EB) bounds, "
             "not a new PPI estimator or a claim of improved practical efficiency.", "",
             "Finite-sample coverage claims require independent iid audit pairs and an independent "
             "prediction pool, bounded outcomes/predictions, fixed sample sizes and a frozen predictor. "
             "The audit outcome and predictor marginals must represent the intended target. "
             "These assumptions cannot be verified from score arrays alone.", "",
             "Fixed-power methods use powers 0 or 1. Grid EB chooses the smallest untruncated radius "
             "over the prespecified powers `[0, 0.25, 0.5, 0.75, 1]`, with a simultaneous error budget "
             "across all residual candidates and one shared prediction-mean bound. "
             "This is different from inserting an unrestricted fitted coefficient into a fixed-power guarantee. "
             "No data-dependent choice between bound families is made.", "",
             "## Design", "",
             f"Actual run: {config['repetitions']:,} independent replications per scenario; master seed {config['seed']}. "
             "All eight methods share every draw. The 12 iid scenarios combine four profiles "
             "(balanced strong judge, near-boundary strong judge, uninformative judge, and perfect-judge control) "
             "with 20, 200 and 1,000 audit observations. Prediction pools have ten times as many observations. "
             "Exact generation parameters and candidate powers are in `config.json`.", "",
             "Every interval is untruncated, including those extending beyond [0,1]. "
             "A known bounded parameter allows intersection with [0,1], but clipping would obscure "
             "the raw width cost, so it is not used here. Zero observed variance does not justify zero "
             "finite-sample width. Perfect sampled predictions do not shrink the known residual range.", "",
             "## IID coverage and width", "",
             f"Across the {len(finite)} theorem-applicable method/scenario cells, observed coverage ranges "
             f"from {finite.coverage.min():.3f} to {finite.coverage.max():.3f}. "
             "These Monte Carlo observations do not prove the theorem or guarantee performance outside its assumptions. "
             f"Grid EB is narrower than the standalone human Hoeffding interval in {narrower}/{len(comparison)} iid scenarios. "
             "All widths and underperforming cases remain visible.", "",
             "The near-boundary case (human prevalence 0.95, audit size 20) separates point accuracy from interval validity. "
             f"Tuned PPI has RMSE {tuned.rmse:.4f}, versus {human.rmse:.4f} for human-only, "
             f"but its nominal 95% normal interval covers only {tuned.coverage:.1%} of replications; "
             f"{int(tuned.zero_width_draws):,}/{config['repetitions']:,} intervals have zero width. "
             f"Human Hoeffding coverage is {hoeffding.coverage:.1%}, with mean width {hoeffding.mean_width:.4f}, "
             f"versus {tuned.mean_width:.4f} for tuned normal. "
             f"Grid EB is wider still ({grid.mean_width:.4f}), containing the entire [0,1] parameter range "
             f"in {grid.fraction_full_domain_covered:.1%} of replications. "
             "Improving point RMSE does not validate a narrow interval, and conservative coverage can be uninformative.", "",
             "Coverage uncertainty below is a pointwise 95% exact-binomial Monte Carlo interval. "
             "Even a run with no misses has a lower endpoint below 1. These intervals describe the "
             "simulation's coverage probability; they are not the estimator's own confidence interval "
             "and are not simultaneous across all table rows.", "",
             "| Scenario | Method | Coverage | MC interval | Mean width | Full [0,1] containment | Mean power |",
             "|---|---|---:|---|---:|---:|---:|"]
    for row in iid.itertuples():
        lines.append(f"| {row.scenario} | {NAMES[row.method]} | {row.coverage:.3f} | "
                     f"[{row.coverage_mc_low:.3f}, {row.coverage_mc_high:.3f}] | {row.mean_width:.4f} | "
                     f"{row.fraction_full_domain_covered:.3f} | {row.mean_selected_power:.3f} |")
    if config["plots"]:
        lines += ["", "![IID coverage and width](finite_sample_tradeoff.svg)", "",
                  "The plot shows six methods for readability; the table and trial data retain all eight. "
                  "Coverage bars are pointwise 95% exact-binomial Monte Carlo intervals; width uses a log scale. "
                  "Axes share scales across profiles. Normal intervals are asymptotic; the stated finite-sample claims "
                  "apply to the bounded-score methods under the specified iid assumptions."]
    violations = summary.loc[summary.scenario_family != "iid"]
    lines += ["", "## Deliberate assumption failures", "",
              "Repeated rows are analyzed as iid to demonstrate misspecification. The shift scenario "
              "changes human prevalence between audit and target with the conditional judge law held fixed. "
              "Neither situation carries the finite-sample guarantee for the intended target human mean. "
              "Intervals can remain numerically wide or contain truth without restoring the violated assumptions.", "",
              "| Scenario | Method | Target coverage | MC interval | Mean width | Bias |",
              "|---|---|---:|---|---:|---:|"]
    for row in violations.itertuples():
        lines.append(f"| {row.scenario} | {NAMES[row.method]} | {row.coverage:.3f} | "
                     f"[{row.coverage_mc_low:.3f}, {row.coverage_mc_high:.3f}] | {row.mean_width:.4f} | {row.bias:+.4f} |")
    lines += ["", "Human-only intervals in the iid shift scenario still estimate the audit prevalence. "
              "The exports distinguish that native parameter from target prevalence. Nonhuman shifted "
              "rectified-functionals are not reported as native targets here; this omission does not assert "
              "that component inequalities fail for their own expectations.", "",
              "## What the artifacts retain", "",
              "- `trials.csv.gz`: every draw/method result, population truth, interval, power, candidate "
              "powers/radii and theorem-applicability metadata. No failed-coverage or uninformative interval is removed.",
              "- `summary.csv`: bias, RMSE, coverage, width, Monte Carlo errors and exact coverage intervals; "
              "width greater than one and full-domain containment are separate diagnostics.",
              "- Paired width differences compare each PPI method to the human-only interval in its "
              "own family on the same draw, with paired Monte Carlo errors. Grid selection can reduce "
              "width within its simultaneously calibrated candidates but need not beat a separately calibrated baseline.",
              "- Configuration, package versions and source/artifact hashes make the run auditable. "
              "This synthetic retrospective grid does not establish savings or coverage on real LLM evaluations.", "",
              "## Reproduce offline", "", "```bash", 'pip install -e ".[dev,report]"',
              "python examples/finite_sample_study.py --output reports/reproduced/finite-sample --plot", "```", "",
              "Defaults use 1,000 replications and seed 2028. Smaller `--repetitions` runs are explicitly "
              "recorded smoke checks. No network, model API, private data or purchased annotation is used.", ""]
    return "\n".join(lines)


def plot_results(summary, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"svg.hashsalt": "judgecal-finite-sample-v1", "font.size": 9})
    table = iid_plot_data(summary, config)
    profiles = [("balanced_strong", "Balanced strong judge"), ("rare_strong", "Near-boundary strong judge"),
                ("balanced_uninformative", "Uninformative judge"), ("balanced_perfect", "Perfect-judge control")]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.2), sharey="row", constrained_layout=True)
    for column, (profile, title) in enumerate(profiles):
        for method in PLOT_METHODS:
            rows = table.loc[(table.profile == profile) & (table.method == method)].sort_values("n_labeled")
            style = "--" if "normal" in method else "-"
            axes[0, column].errorbar(
                rows.n_labeled, rows.coverage,
                yerr=[rows.coverage-rows.coverage_mc_low, rows.coverage_mc_high-rows.coverage],
                marker="o", linewidth=1.7, linestyle=style, color=COLORS[method],
                label=NAMES[method], markersize=4, capsize=2)
            axes[1, column].plot(rows.n_labeled, rows.mean_width, marker="o", linewidth=1.7,
                                 linestyle=style, color=COLORS[method], label=NAMES[method], markersize=4)
        axes[0, column].set(title=title, ylim=(0, 1.02))
        axes[0, column].axhline(.95, color="#888888", linestyle=":", linewidth=1)
        axes[1, column].set(yscale="log", xlabel="Independent audit observations")
        axes[1, column].axhline(1, color="#888888", linestyle=":", linewidth=1)
        for ax in axes[:, column]:
            ax.set(xscale="log", xticks=[20, 200, 1000], xticklabels=["20", "200", "1000"])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel("Coverage of target human mean")
    axes[1, 0].set_ylabel("Mean untruncated interval width (log scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle("Finite-sample coverage comes with a width cost", fontsize=13)
    fig.savefig(output/"finite_sample_tradeoff.png", dpi=180)
    fig.savefig(output/"finite_sample_tradeoff.svg", metadata={"Date": None})
    svg = output/"finite_sample_tradeoff.svg"
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines())+b"\n")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reports/finite-sample")
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    output = args.output
    artifacts = {"trials.csv.gz", "summary.csv", "config.json", "REPORT.md"}
    if args.plot:
        artifacts |= {"finite_sample_tradeoff.png", "finite_sample_tradeoff.svg"}
    if output.exists() and {p.name for p in output.iterdir()} - artifacts - {"reproducibility.json"}:
        raise ValueError("output contains stale or unrelated artifacts; use a fresh directory")
    trials, summary, config = run_finite_sample_study(args.repetitions, args.seed)
    config["plots"] = args.plot
    output.mkdir(parents=True, exist_ok=True)
    with (output/"trials.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(trials.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    summary.to_csv(output/"summary.csv", index=False, lineterminator="\n")
    write_json(output/"config.json", config)
    link = os.path.relpath(ROOT/"docs/FINITE_SAMPLE_PROTOCOL.md", output).replace("\\", "/")
    (output/"REPORT.md").write_bytes(make_report(summary, config, link).encode("utf-8"))
    if args.plot:
        plot_results(summary, config, output)
    sources = sorted((ROOT/"src/judgecal").glob("*.py")) + [Path(__file__), ROOT/"examples/research_study.py",
                ROOT/"docs/FINITE_SAMPLE_PROTOCOL.md", ROOT/"pyproject.toml"]
    manifest = {"python": platform.python_version(), "platform": platform.system(),
                "packages": {name: version(name) for name in ["judgecal", "numpy", "scipy", "pandas"]},
                "source_text_normalization": "CRLF to LF before SHA-256 (matches Git text attributes)",
                "source_sha256": {p.relative_to(ROOT).as_posix():
                                  hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sources},
                "artifacts_sha256": {name: sha256(output/name) for name in sorted(artifacts)}}
    if args.plot:
        manifest["packages"]["matplotlib"] = version("matplotlib")
    write_json(output/"reproducibility.json", manifest)
    print(f"Wrote {len(trials)} trial rows and {len(summary)} method/scenario summaries.")


if __name__ == "__main__":
    main()
