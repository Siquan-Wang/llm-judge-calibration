"""Compare independent pilot tuning at a fixed total reference-label budget."""
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

from judgecal.pilot_study import independent_pilot_study


ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "human_normal": "Full audit normal", "human_eb": "Full audit EB",
    "same_audit_normal": "Same-audit tuned normal",
    "same_audit_grid_eb": "Same-audit grid EB",
    "pilot20_normal": "20% pilot normal", "pilot20_eb": "20% pilot EB",
    "pilot50_normal": "50% pilot normal", "pilot50_eb": "50% pilot EB",
}
PROFILES = ("balanced_strong", "boundary_strong", "uninformative", "perfect")
PROFILE_NAMES = {"balanced_strong": "Balanced strong", "boundary_strong": "Near-boundary strong",
                 "uninformative": "Balanced uninformative", "perfect": "Balanced perfect"}
COLORS = {"human_normal": "#59616d", "human_eb": "#59616d",
          "same_audit_normal": "#3575b5", "same_audit_grid_eb": "#9260b8",
          "pilot20_normal": "#008870", "pilot20_eb": "#008870",
          "pilot50_normal": "#bf653d", "pilot50_eb": "#bf653d"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode("utf-8"))


def save_csv(path, frame):
    # Explicit newlines and precision keep the historical artifact portable.
    raw = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")
    if str(path).endswith(".gz"):
        with Path(path).open("wb") as output:
            with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed:
                compressed.write(raw)
    else:
        Path(path).write_bytes(raw)


def make_plots(summary, output, alpha=.05):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "judgecal-pilot-study-v1"})

    def finish(fig, name):
        fig.savefig(output/f"{name}.png", dpi=180)
        fig.savefig(output/f"{name}.svg", metadata={"Date": None})
        target = output/f"{name}.svg"
        target.write_bytes(b"\n".join(line.rstrip() for line in target.read_bytes().splitlines())+b"\n")
        plt.close(fig)

    def format_axes(axes):
        for axis in axes.flat:
            axis.set_xscale("log")
            axis.set_xticks([20, 200, 1000])
            axis.xaxis.set_major_formatter(ScalarFormatter())
            axis.set_xlabel("Total paid labels (log scale)")
            axis.grid(alpha=.18)
            axis.spines[["top", "right"]].set_visible(False)

    point_methods = ("human_normal", "same_audit_normal", "same_audit_grid_eb", "pilot20_normal", "pilot50_normal")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for column, profile in enumerate(PROFILES):
        for method in point_methods:
            values = summary.loc[(summary.profile == profile) & (summary.method == method)].sort_values("total_budget")
            axes[0, column].plot(values.total_budget, values.rmse*100, marker="o", color=COLORS[method], label=NAMES[method])
            axes[1, column].errorbar(values.total_budget, values.bias*100,
                                    yerr=1.96*values.bias_mcse*100, marker="o", capsize=2,
                                    color=COLORS[method], label=NAMES[method])
        axes[0, column].set(title=PROFILE_NAMES[profile], ylabel="Population RMSE (pp)", ylim=(0, None))
        axes[1, column].set(ylabel="Bias with pointwise MC interval (pp)")
        axes[1, column].axhline(0, color="#30343b", linestyle="--", linewidth=.8)
    format_axes(axes)
    fig.suptitle("Independent tuning spends part of the same total label budget; compare within each law")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False)
    finish(fig, "pilot_point_tradeoff")

    normal_methods = ("human_normal", "same_audit_normal", "pilot20_normal", "pilot50_normal")
    eb_methods = ("human_eb", "same_audit_grid_eb", "pilot20_eb", "pilot50_eb")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for column, profile in enumerate(PROFILES):
        for method in normal_methods:
            values = summary.loc[(summary.profile == profile) & (summary.method == method)].sort_values("total_budget")
            error = np.maximum(0, np.array([values.coverage-values.coverage_mc_low,
                                            values.coverage_mc_high-values.coverage]))
            axes[0, column].errorbar(values.total_budget, values.coverage*100, yerr=error*100,
                                    marker="o", capsize=2, color=COLORS[method], label=NAMES[method])
        for method in eb_methods:
            values = summary.loc[(summary.profile == profile) & (summary.method == method)].sort_values("total_budget")
            axes[1, column].plot(values.total_budget, values.mean_width, marker="o", color=COLORS[method], label=NAMES[method])
        axes[0, column].set(title=PROFILE_NAMES[profile], ylabel="Normal coverage (%)", ylim=(0, 103))
        axes[0, column].axhline(100*(1-alpha), color="#30343b", linestyle="--", linewidth=.8)
        axes[1, column].set(ylabel="EB mean untruncated width", ylim=(0, None))
        axes[1, column].axhline(1, color="#30343b", linestyle=":", linewidth=.8)
    format_axes(axes)
    fig.suptitle("Normal coverage and conservative EB width are separate properties")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    eb_handles, eb_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles+eb_handles, labels+eb_labels, loc="outside lower center", ncol=4, frameon=False)
    finish(fig, "pilot_interval_tradeoff")


def make_report(summary, paired, configuration, protocol_link, plots):
    repetitions = int(summary.repetitions.iloc[0])
    boundary = summary.loc[(summary.profile == "boundary_strong") & (summary.total_budget == 20)].set_index("method")
    control = summary.loc[(summary.profile == "perfect") & (summary.total_budget == 20)].set_index("method")
    eb_summary = summary.loc[summary.interval_family == "empirical_bernstein"]
    small_gap = paired.loc[(paired.profile == "boundary_strong") & (paired.total_budget == 200)
                          & (paired.method == "pilot20_normal") & (paired.baseline == "human_normal")
                          & (paired.metric == "mse")].iloc[0]
    narrower = {}
    for method in ("pilot20_eb", "pilot50_eb"):
        for baseline in ("human_eb", "same_audit_grid_eb"):
            contrasts = paired.loc[(paired.method == method) & (paired.baseline == baseline) & (paired.metric == "width")]
            narrower[method, baseline] = int((contrasts.mean_difference < 0).sum())
    lines = ["# What does independent tuning cost at a fixed label budget?", "",
             f"[Protocol, equations and primary sources]({protocol_link})", "",
             "## Question and scope", "",
             "This retrospective simulation compares established same-audit and independent-pilot "
             "control-variate procedures at the same total reference-label budget. It introduces no new "
             "estimator or theorem. Conditional on an independent pilot, its fitted coefficient is fixed, "
             "the corrected mean is unbiased, and the existing scalar empirical-Bernstein (EB) bound applies "
             "under the declared iid assumptions. These facts do not guarantee smaller error, narrower "
             "intervals or finite-sample validity of a normal interval.", "",
             f"The complete matrix has 12 laws/budgets, eight procedures and {repetitions:,} independent "
             f"replications per cell ({len(summary)*repetitions:,} retained method trials). "
             f"Master seed: {configuration['seed']}. All configured cells and degenerate pilots are retained. "
             "No LLM calls, human annotations, benchmark observations or dollar savings were measured.", "",
             "## Observed findings", "",
             "At p=.95 and B=20, same-audit normal tuning has RMSE "
             f"{boundary.loc['same_audit_normal','rmse']:.5f} and coverage "
             f"{100*boundary.loc['same_audit_normal','coverage']:.2f}%. The 20% pilot gives "
             f"{boundary.loc['pilot20_normal','rmse']:.5f} and {100*boundary.loc['pilot20_normal','coverage']:.2f}%; "
             f"the 50% pilot gives {boundary.loc['pilot50_normal','rmse']:.5f} and "
             f"{100*boundary.loc['pilot50_normal','coverage']:.2f}%. Its independent-pilot unbiasedness "
             "argument does not establish an accurate small-sample normal approximation. Bias and its "
             "Monte Carlo error are retained below, alongside all other cells.", "",
             f"Observed EB coverage ranges from {100*eb_summary.coverage.min():.2f}% to "
             f"{100*eb_summary.coverage.max():.2f}%. The 20% pilot EB interval has smaller mean width than "
             f"full-audit EB in {narrower['pilot20_eb','human_eb']}/12 cells and the grid EB rule in "
             f"{narrower['pilot20_eb','same_audit_grid_eb']}/12; the corresponding 50% pilot counts are "
             f"{narrower['pilot50_eb','human_eb']}/12 and {narrower['pilot50_eb','same_audit_grid_eb']}/12. "
             "Avoiding a grid-selection penalty does not remove the cost of fewer residual labels.", "",
             "Even the perfect-proxy control retains small-pilot degeneracy. At B=20, "
             f"{100*control.loc['pilot20_normal','constant_pilot_fraction']:.2f}% of four-label pilots "
             f"have constant predictions, versus {100*control.loc['pilot50_normal','constant_pilot_fraction']:.2f}% "
             "of ten-label pilots. The two pilot rules have RMSE "
             f"{control.loc['pilot20_normal','rmse']:.5f} and {control.loc['pilot50_normal','rmse']:.5f}, "
             "respectively. This selected structural example does not choose a preferred allocation "
             "for other laws; the complete matrix stays visible.", "",
             "## Design and cost accounting", "",
             "Total paid labels are B=20,200,1000; the independent prediction pool always has N=10B rows. "
             "Profiles are balanced strong (p=.5, sensitivity=.95, specificity=.90), near-boundary strong "
             "(p=.95, same proxy), balanced uninformative (p=.5, sensitivity=.60, specificity=.40), and "
             "balanced perfect (p=.5, sensitivity=specificity=1). The perfect law is a structural control, "
             "not information supplied to a fitted rule. The target is the known population outcome mean p.", "",
             "A pilot uses P=floor(.20B) or floor(.50B) labels and leaves m=B-P for residual estimation. "
             "Pilot rows are not reused in the residual mean or prediction pool. The coefficient uses "
             "only paired pilot moments and the known m,N counts; exact constant pilot predictions "
             "select zero. A zero pilot coefficient therefore uses only m final labels, despite paying "
             "for B. All full-audit rules use B labels. The experiment materializes B+N proxy scores "
             "per draw; the full-audit reference-only rules need no proxy scores for their mathematical "
             "estimates. These logical counts are not measured operational costs.", "",
             "```text", "lambda_P = clip((N/(m+N)) * Cov_P(Y,F)/Var_P(F), 0, 1)",
             "estimate = mean(Y_R) + lambda_P * [mean(F_U) - mean(F_R)]", "```", "",
             "Same-audit normal tuning retains the existing per-pool variance formula. The pilot uses "
             "one pilot marginal variance for both pools, so this compares complete budget-matched "
             "procedures; it does not isolate independence from every other formula choice. The grid EB "
             "rule selects among the prespecified powers (0,.25,.5,.75,1) with simultaneous bounds. Each "
             "pilot EB rule instead conditions on its independently fitted scalar. Choosing whichever "
             "reported fraction or interval looks best afterwards is not a validated ninth procedure.", "",
             "## Paired point-error comparisons", "",
             "The signs below are observed per-cell MSE differences from shared draws, not probabilities "
             "of improvement on new tasks. Tabulated errors and widths are in outcome fractions; MSE "
             "contrasts use squared fractions. Plot RMSE and bias use percentage points (pp). "
             "No loss is pooled across laws or budgets. Standard errors "
             "use per-replication paired differences. Normal/EB variants of a pilot have identical points.", "",
             "For scale, at p=.95 and B=200, the 20% pilot-minus-audit MSE contrast is "
             f"{small_gap.mean_difference:.6g} with MCSE {small_gap.mcse:.6g}. The sign-count table "
             "records observed signs even when their magnitude is small relative to Monte Carlo uncertainty; "
             "it does not declare every counted comparison statistically resolved.", "",
             "| Pilot | Baseline | Lower MSE cells | Higher MSE cells | Exact ties |", "|---|---|---:|---:|---:|"]
    for method in ("pilot20_normal", "pilot50_normal"):
        for baseline in ("human_normal", "same_audit_normal"):
            values = paired.loc[(paired.method == method) & (paired.baseline == baseline) & (paired.metric == "mse")]
            if len(values) != 12:
                raise ValueError("paired MSE report requires all twelve declared cells")
            differences = values.mean_difference
            lines.append(f"| {NAMES[method]} | {NAMES[baseline]} | {int((differences<0).sum())}/12 | "
                         f"{int((differences>0).sum())}/12 | {int((differences==0).sum())}/12 |")
    lines += ["", "| Profile | Total B | Pilot | Baseline | Paired MSE difference | MCSE |",
              "|---|---:|---|---|---:|---:|"]
    for row in paired.loc[paired.method.isin(["pilot20_normal", "pilot50_normal"]) & (paired.metric == "mse")].itertuples():
        lines.append(f"| {PROFILE_NAMES[row.profile]} | {row.total_budget} | {NAMES[row.method]} | "
                     f"{NAMES[row.baseline]} | {row.mean_difference:.6g} | {row.mcse:.6g} |")
    if plots:
        lines += ["", "![Point-error and bias tradeoffs](pilot_point_tradeoff.svg)", "",
                  "Point errors target population p. Bias bars are pointwise approximate 95% Monte Carlo "
                  "intervals (estimate +/-1.96 MCSE), not the procedure's inference intervals. Panel y scales "
                  "may differ; compare rules within each law. Duplicate pilot EB point curves are omitted."]

    lines += ["", "## Complete error and uncertainty results", "",
              "Coverage means that the reported interval contains the known population p in this simulation. "
              "The protocol's eight-epsilon numerical membership margin affects scoring only, not estimates "
              "or interval endpoints. "
              "Coverage MC bounds are pointwise exact 95% binomial intervals. Monte Carlo precision does not "
              "establish universal validity or support a real-corpus coverage claim. All widths are untruncated. "
              "The full [0,1] containment fraction shows when a conservative interval covers every possible mean.", "",
              "| Profile | B | Procedure | Pilot P | Residual m | RMSE | Bias (MCSE) | Coverage (MC bounds) | Mean width | Full domain |",
              "|---|---:|---|---:|---:|---:|---|---|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {PROFILE_NAMES[row.profile]} | {row.total_budget} | {NAMES[row.method]} | "
                     f"{row.n_pilot} | {row.n_labeled} | {row.rmse:.5f} | {row.bias:.5f} ({row.bias_mcse:.5f}) | "
                     f"{row.coverage:.3f} [{row.coverage_mc_low:.3f},{row.coverage_mc_high:.3f}] | "
                     f"{row.mean_width:.4f} | {row.full_domain_fraction:.3f} |")
    if plots:
        lines += ["", "![Normal coverage and finite-bound width](pilot_interval_tradeoff.svg)", "",
                  "Top: normal interval coverage with exact pointwise Monte Carlo error bars. Bottom: EB widths; "
                  "the dotted line is the full parameter-domain width. Full EB coverage and all widths remain "
                  "in the table. Same-audit normal tuning and same-audit finite-grid EB are different procedures."]
    lines += ["", "## Degenerate pilots and interpretation", "",
              "| Profile | B | Pilot | Constant pilot fraction | Zero power fraction | Zero-width normal fraction |",
              "|---|---:|---|---:|---:|---:|"]
    for row in summary.loc[summary.method.isin(["pilot20_normal", "pilot50_normal"])].itertuples():
        lines.append(f"| {PROFILE_NAMES[row.profile]} | {row.total_budget} | {NAMES[row.method]} | "
                     f"{row.constant_pilot_fraction:.3f} | {row.zero_power_fraction:.3f} | {row.zero_width_fraction:.3f} |")
    lines += ["", "The independent-pilot unbiasedness argument is mathematical and conditional on the design; "
              "a small Monte Carlo bias estimate does not prove it. Nor does an unbiased sample variance "
              "estimate make a small-sample normal interval exact. A four-label pilot at B=20 is deliberately "
              "retained. When a proxy is uninformative, splitting spends labels without population covariance "
              "to recover their value; the protocol gives the exact fixed-coefficient variance decomposition. "
              "Neither fractions nor proxy profiles were selected by favorable outcomes.", "",
              "The study is restricted to iid bounded binary laws and fixed sampling sizes. It does not cover "
              "clustered corpora, sampling without replacement, distribution shift, predictor training on these "
              "labels, optional stopping or cross-fitting. Its mechanism families overlap earlier simulations, "
              "so the new results are a targeted procedural comparison, not external validation or twelve "
              "independent deployment domains.", "",
              "## Artifacts and reproduction", "",
              "- `trials.csv.gz`: every method/draw, pilot diagnostics, costs, errors and interval endpoints.",
              "- `summary.csv`: all method/cell summaries, exact coverage MC bounds and diagnostics.",
              "- `paired_differences.csv`: paired squared-error, coverage and width contrasts with MCSE.",
              "- `study_config.json`: full declared design, stream keys, method and sampling contracts.",
              "- `reproducibility.json`: environment, source and artifact fingerprints.", "",
              "```bash", f"python examples/independent_pilot_study.py --repetitions {repetitions} "
              f"--seed {configuration['seed']} --alpha {configuration['alpha']} --output reports/reproduced/pilot-study"+(" --plot" if plots else ""),
              "```", "", "The experiment uses addressable Philox streams. Extending the replication count preserves "
              "earlier keyed draws. Reproducing requires the recorded software environment; bitwise identity "
              "across different NumPy or rendering versions is not asserted. Smaller runs retain their actual "
              "repetition counts and are smoke checks, not the published full experiment."]
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reports/pilot-study")
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    names = {"trials.csv.gz", "summary.csv", "paired_differences.csv", "study_config.json", "REPORT.md", "reproducibility.json"}
    if args.plot:
        names |= {f"{stem}.{suffix}" for stem in ("pilot_point_tradeoff", "pilot_interval_tradeoff") for suffix in ("png", "svg")}
    if output.exists() and {item.name for item in output.iterdir()}-names:
        raise ValueError("output directory contains unrelated or stale artifacts")
    if (output/"reproducibility.json").exists():
        previous = json.loads((output/"reproducibility.json").read_text(encoding="utf-8"))
        if set(previous.get("artifacts_sha256", {})) != names-{"reproducibility.json"}:
            raise ValueError("output artifact set differs; use a fresh directory")
    trials, summary, paired = independent_pilot_study(args.repetitions, args.seed, args.alpha)
    config = dict(trials.attrs["configuration"])
    config.update({"repetitions": args.repetitions, "seed": args.seed, "alpha": args.alpha, "plots": args.plot})
    protocol = "https://github.com/Siquan-Wang/llm-judge-calibration/blob/main/docs/PILOT_STUDY_PROTOCOL.md"
    report = make_report(summary, paired, config, protocol, args.plot)
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output/"trials.csv.gz", trials)
    save_csv(output/"summary.csv", summary)
    save_csv(output/"paired_differences.csv", paired)
    write_json(output/"study_config.json", config)
    (output/"REPORT.md").write_bytes(report.encode("utf-8"))
    if args.plot:
        make_plots(summary, output, args.alpha)
    sources = [*sorted((ROOT/"src/judgecal").glob("*.py")), Path(__file__),
               ROOT/"docs/PILOT_STUDY_PROTOCOL.md", ROOT/"pyproject.toml"]
    source_hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in sources}
    manifest = {"kind": "independent-pilot fixed-total-budget simulation", "python": platform.python_version(),
                "platform": platform.system(), "packages": {name: version(name) for name in ("judgecal", "numpy", "scipy", "pandas")},
                "source_text_normalization": "CRLF to LF before SHA-256", "source_sha256": source_hashes,
                "artifacts_sha256": {name: digest(output/name) for name in sorted(names-{"reproducibility.json"})}}
    if args.plot:
        manifest["packages"]["matplotlib"] = version("matplotlib")
    write_json(output/"reproducibility.json", manifest)
    print(f"Wrote {len(trials):,} trials, {len(summary)} summaries and {len(paired)} paired contrasts to {output}.")


if __name__ == "__main__":
    main()
