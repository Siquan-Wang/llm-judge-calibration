"""Reproduce fixed-corpus sampling-design inference from pinned public labels."""
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

from judgecal.finite_corpus_study import finite_corpus_study, METHODS


ROOT = Path(__file__).resolve().parents[1]
LABELS_SHA256 = "7dc7d8aad832f470a2831df5011fd7e8e1c670996a225b5d91c81f760350120f"
PROVENANCE_SHA256 = "3ff9bd62f5a939c645b1a18e75cacf83bbe4e37a3d6e26747980894f993062f2"
NAMES = {"ht_hs": "Unadjusted HT + HS", "ht_ebs": "Unadjusted HT + EBS",
         "agreement_grid_ebs": "Agreement grid + EBS", "size_grid_ebs": "Size-only grid + EBS"}
COLORS = dict(zip(METHODS, ("#5c6270", "#a86839", "#2679b1", "#008675")))
SHORT = {"openai/gpt-4o-2024-08-06": "GPT-4o (2024-08-06)",
         "openai/gpt-4o-mini-2024-07-18": "GPT-4o mini (2024-07-18)"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_bytes(path, raw):
    if str(path).endswith(".gz"):
        with Path(path).open("wb") as stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
                compressed.write(raw)
    else:
        Path(path).write_bytes(raw)


def save_json(path, value):
    save_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode("utf-8"))


def make_plot(summary, output, alpha):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "judgecal-finite-corpus-v1"})
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for column, (judge, title) in enumerate(SHORT.items()):
        for method in METHODS:
            values = summary.loc[(summary.target_judge == judge) & (summary.method == method)].sort_values("group_fraction")
            options = {"marker": "o", "color": COLORS[method], "label": NAMES[method]}
            axes[0, column].plot(100*values.group_fraction, 100*values.rmse, **options)
            axes[1, column].errorbar(100*values.group_fraction, values.mean_width,
                                   yerr=1.96*values.width_mcse, capsize=3, **options)
            errors = np.maximum(0, np.array([values.coverage-values.coverage_mc_low,
                                             values.coverage_mc_high-values.coverage]))
            axes[2, column].errorbar(100*values.group_fraction, 100*values.coverage,
                                   yerr=100*errors, capsize=3, **options)
        axes[0, column].set(title=title, ylabel="Full-corpus RMSE (percentage points)", ylim=(0, None))
        axes[1, column].set(ylabel="Mean untruncated interval width", ylim=(0, None))
        axes[2, column].set(ylabel="Design coverage (%)", ylim=(min(90, 100*(1-alpha)-2), 101))
        axes[2, column].axhline(100*(1-alpha), color="#333333", linestyle="--", linewidth=.8)
    for ax in axes.flat:
        ax.set(xlabel="Audited prompt groups (% of fixed frame)", xticks=[20, 60, 90])
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fixed-corpus inference: separate prediction signal from known group-size adjustment")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(output/"finite_corpus_tradeoff.png", dpi=180)
    fig.savefig(output/"finite_corpus_tradeoff.svg", metadata={"Date": None})
    svg = output/"finite_corpus_tradeoff.svg"
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines())+b"\n")
    plt.close(fig)


def make_report(result, plots):
    summary, paired, config = result["summary"], result["paired"], result["configuration"]
    primary = paired.loc[(paired.method == "agreement_grid_ebs") & (paired.baseline == "size_grid_ebs")]
    lines = ["# Can agreement improve a finite-corpus audit beyond known group size?", "",
             "This retrospective study targets the **complete fixed corpus's row-average reference correctness**. "
             "It applies classical difference estimation and Bardenet–Maillard sampling-without-replacement bounds. "
             "It introduces no new estimator or theorem and does not reinterpret the earlier held-out target.", "",
             f"The pinned NonLLMBar frame has {config['n_rows']:,} comparisons in {config['n_groups']:,} exact-prompt "
             f"groups. Two dated judge caches, four procedures and three budgets share {config['repetitions']:,} "
             f"independent random permutations, producing {len(result['trials']):,} retained method-trial rows. "
             "Nested budgets, procedures and target directions are paired, not independent new datasets. "
             "There are no new LLM calls or newly purchased reference labels.", "",
             "## Observed comparison", "",
             "The primary comparison gives the agreement signal and the size-only control without auxiliary judge information the same "
             "coefficient grid and error allocation. Both can use known group sizes; only the agreement rule "
             "uses the cross-judge signal. A lower error than unadjusted HT alone is insufficient to attribute "
             "the gain to the LLM proxy.", "",
             f"Agreement-grid has lower observed MSE than size-only in {int((primary.mse_difference<0).sum())}/{len(primary)} "
             f"cells and lower mean width in {int((primary.width_difference<0).sum())}/{len(primary)}. "
             "These are observed signs, not significance declarations or guarantees for new corpora. "
             "The paired differences and Monte Carlo standard errors below retain every configured comparison.", "",
             "Paired MSE differences use squared outcome fractions; width differences use outcome fractions. "
             "Complete-results RMSE, bias and width also use fractions. Only plotted RMSE uses percentage points.", "",
             f"Observed coverage across all 24 cells is {100*summary.coverage.min():.2f}–{100*summary.coverage.max():.2f}%. "
             "The finite guarantee comes from the assumptions and inequalities, not this empirical coverage. "
             "High coverage can coexist with conservative width and biased grid-selected points.", "",
             "| Target cache | Group fraction | Agreement minus size-only MSE | Paired MCSE | Width difference | Paired MCSE |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in primary.itertuples():
        lines.append(f"| {SHORT[row.target_judge]} | {row.group_fraction:.0%} | {row.mse_difference:.7g} | "
                     f"{row.mse_mcse:.7g} | {row.width_difference:.6g} | {row.width_mcse:.6g} |")
    if plots:
        lines += ["", "![Complete fixed-corpus results](finite_corpus_tradeoff.svg)", "",
                  "RMSE is in percentage points; widths are outcome fractions. Width bars are pointwise "
                  "approximate 95% Monte Carlo intervals, and coverage bars are pointwise exact 95% binomial "
                  "intervals. Overlapping curves may hide methods; all values remain in the table. "
                  "Unadjusted HT's two interval families have identical point estimates."]
    lines += ["", "## Target, sampling and guarantee", "",
              "Every comparison has a frozen reference outcome in [0,1]. Complete prompt groups are sampling "
              "units, with known sizes n_g and known proxy totals F_g. A uniform permutation's fixed k-prefix "
              "is a simple random sample of groups without replacement. Every row in each selected group is audited.", "",
              "```text", "M = sum_g n_g; theta = sum_g Y_g / M",
              "estimate(lambda) = lambda*sum_all F_g/M + G/(k*M)*sum_sample(Y_g-lambda*F_g)", "```", "",
              "Fixed coefficients give a design-unbiased estimate of the row-weighted full-corpus mean. "
              "A sampled-row ratio and an equal average of group accuracies are different estimators. "
              "The same-audit finite-grid selector retains simultaneous interval coverage within its own "
              "family but need not retain unbiasedness. No unsampled outcomes enter fitting, candidate "
              "selection or the residual support range. Full outcomes enter scoring only.", "",
              "The four families are fixed-zero HT with Hoeffding–Serfling (HS), fixed-zero HT with "
              "empirical Bernstein–Serfling (EBS), agreement-grid EBS, and size-only-grid EBS. "
              "Both grids are (0,.25,.5,.75,1); size-only uses F_g=n_g and no auxiliary judge signal. "
              "The common support is [min_g(-lambda F_g),max_g(n_g-lambda F_g)]. EBS uses the "
              "audited residual variance with divisor k and the full known support width, never an "
              "observed residual range. HS uses log(2K/alpha); EBS uses log(10K/alpha) and "
              "kappa=7/3+3/sqrt(2). Census returns the exactly observed mean and zero width.", "",
              "Coverage is marginal over the specified group-sampling design, conditional on this fixed "
              "frame and all its fixed outcomes. It requires neither independent group outcomes nor an "
              "iid superpopulation. It does require complete frame membership, fixed k, uniform group "
              "inclusion and complete audited groups. Convenience samples, stopping at a row-label cap, "
              "nonresponse and post-hoc selection across the four families are not covered. "
              "The reported intervals do not jointly cover all methods, judges and budgets at the nominal level.", "",
              "All intervals are untruncated. The scoring-only eight-epsilon membership margin changes "
              "neither estimates nor endpoints. Coverage MC intervals are 95% regardless of the chosen "
              "inference alpha. The finite target is recorded benchmark correctness, not error-free latent "
              "quality, the unobserved remainder mean or a new deployment population.", "",
              "[Frozen protocol and primary sources](../../docs/FINITE_CORPUS_PROTOCOL.md). "
              "[Original cache provenance and notices](../rewardbench/NOTICE.md). This is a reanalysis of "
              "the same cached panel, not additional external validation. The historical synthesis retains "
              "its own earlier scope and fingerprints.", "", "## Complete results", "",
              "| Target cache | Groups | Procedure | RMSE | Bias (MCSE) | Coverage (MC bounds) | Mean width | Full [0,1] containment |",
              "|---|---:|---|---:|---|---|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {SHORT[row.target_judge]} | {row.n_audited_groups} | {NAMES[row.method]} | {row.rmse:.6f} | "
                     f"{row.bias:.6f} ({row.bias_mcse:.6f}) | {row.coverage:.5f} "
                     f"([{row.coverage_mc_low:.5f}, {row.coverage_mc_high:.5f}]) | {row.mean_width:.6f} | {row.full_domain_fraction:.3f} |")
    lines += ["", "## Cost and census", "",
              "The budget fixes numbers of groups, not labels. Actual reference-label cost is the sum "
              "of sampled group sizes and is shared across all procedures and both target directions. "
              "One full panel materializes two cached choices per row. Agreement requires the corresponding "
              "full-frame cross-judge signal; the size-only rule requires no such signal. These input counts "
              "are not measured compute, latency or dollar savings.", "",
              "| Audited groups | Expected row labels | Mean (MCSE) | Observed min–max | 10/50/90% cost quantiles |",
              "|---:|---:|---|---|---|"]
    cost = summary.loc[(summary.target_judge == config["judges"][0]) & (summary.method == "ht_hs")]
    for row in cost.itertuples():
        lines.append(f"| {row.n_audited_groups} | {row.expected_audited_rows:.3f} | {row.mean_audited_rows:.3f} "
                     f"({row.audit_cost_mcse:.3f}) | {row.minimum_audited_rows}–{row.maximum_audited_rows} | "
                     f"{row.audited_rows_q10:g}/{row.audited_rows_q50:g}/{row.audited_rows_q90:g} |")
    lines += ["", f"All {len(result['census'])} deterministic census checks return the exact full-row mean "
              "and zero width. They are in `census.csv`, outside replication and coverage summaries. "
              "They do not represent additional evidence from thousands of repeated identical censuses.", "",
              "## Reproduction and artifact inventory", "", "```bash",
              f"python examples/finite_corpus_study.py --repetitions {config['repetitions']} --seed {config['seed']} "
              f"--alpha {config['alpha']} --output reports/reproduced/finite-corpus"+(" --plot" if plots else ""), "```", "",
              "- `trials.csv.gz`: every fitted rule, candidate, endpoint, selected power and scored error.",
              "- `summary.csv` and `paired_differences.csv`: all 24 cells and 36 wide paired comparisons with MCSEs.",
              "- `splits.csv`: one shared sample per repetition/budget, actual costs and group/row ID hashes.",
              "- `frame_catalog.json` and `permutations.json.gz`: lossless indexed membership of every draw; "
              "`judgecal.finite_corpus_study.decode_sample` reconstructs the exact IDs.",
              "- `census.csv`: eight exact structural checks, without Monte Carlo error bars.",
              "- `study_config.json` and `reproducibility.json`: randomization, data, source, environment and artifact fingerprints.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"reports/rewardbench/labels.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"reports/finite-corpus")
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2032)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    names = {"trials.csv.gz", "summary.csv", "paired_differences.csv", "splits.csv", "census.csv", "frame_catalog.json",
             "permutations.json.gz", "study_config.json", "REPORT.md", "reproducibility.json"}
    if args.plot:
        names |= {"finite_corpus_tradeoff.png", "finite_corpus_tradeoff.svg"}
    if output.exists() and {p.name for p in output.iterdir()}-names:
        raise ValueError("output directory contains unrelated or stale artifacts")
    if (output/"reproducibility.json").exists():
        previous = json.loads((output/"reproducibility.json").read_text(encoding="utf-8"))
        if set(previous.get("artifacts_sha256", {})) != names-{"reproducibility.json"}:
            raise ValueError("output artifact set differs; use a fresh directory")
    input_paths = {"labels.csv": args.input, "dataset.json": args.input.with_name("dataset.json"),
                   "NOTICE.md": args.input.with_name("NOTICE.md")}
    input_hashes = {name: digest(path) for name, path in input_paths.items()}
    if input_hashes["labels.csv"] != LABELS_SHA256 or input_hashes["dataset.json"] != PROVENANCE_SHA256:
        raise ValueError("input differs from the certified RewardBench panel or provenance")
    labels = pd.read_csv(args.input, dtype={"sample_id": str, "instruction_id": str})
    result = finite_corpus_study(labels, args.repetitions, args.seed, args.alpha)
    result["configuration"]["plots"] = args.plot
    output.mkdir(parents=True, exist_ok=True)
    for key, filename in (("trials", "trials.csv.gz"), ("summary", "summary.csv"), ("paired", "paired_differences.csv"),
                          ("splits", "splits.csv"), ("census", "census.csv")):
        save_bytes(output/filename, result[key].to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8"))
    save_json(output/"frame_catalog.json", result["catalog"])
    # Compact large permutation records, preserving one full draw per replication.
    save_bytes(output/"permutations.json.gz", (json.dumps(result["permutations"], separators=(",", ":"), sort_keys=True)+"\n").encode("utf-8"))
    save_json(output/"study_config.json", result["configuration"])
    (output/"REPORT.md").write_bytes(make_report(result, args.plot).encode("utf-8"))
    if args.plot:
        make_plot(result["summary"], output, args.alpha)
    sources = [*sorted((ROOT/"src/judgecal").glob("*.py")), Path(__file__), ROOT/"docs/FINITE_CORPUS_PROTOCOL.md", ROOT/"pyproject.toml"]
    manifest = {"kind": "fixed-corpus randomized group audit", "python": platform.python_version(), "platform": platform.system(),
                "packages": {name: version(name) for name in ("judgecal", "numpy", "scipy", "pandas")},
                "source_text_normalization": "CRLF to LF before SHA256",
                "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sources},
                "input_sha256": input_hashes,
                "artifacts_sha256": {name: digest(output/name) for name in sorted(names-{"reproducibility.json"})}}
    if args.plot:
        manifest["packages"]["matplotlib"] = version("matplotlib")
    save_json(output/"reproducibility.json", manifest)
    print(f"Wrote {len(result['trials']):,} trials, {len(result['summary'])} summaries, {len(result['paired'])} paired comparisons and 8 census checks to {output}.")


if __name__ == "__main__":
    main()
