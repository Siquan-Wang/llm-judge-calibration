"""Study signed proxy correction without changing the positive-power default."""
from __future__ import annotations

import argparse
import gzip
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from judgecal.judge_audit import judge_accuracy_audit
from judgecal.signed_study import signed_power_study
from llmbar_audit import read_verified_labels, paired_method_differences, save_plot, COHORTS, JUDGES
from research_study import ROOT, sha256, write_json


NAMES = {"human_only": "Gold audit only", "ppi": "PPI (power +1)",
         "ppi_tuned": "Tuned [0,1]", "ppi_signed": "Tuned [-1,1]", "raw_proxy": "Raw agreement"}
COLORS = {"human_only": "#525a66", "ppi": "#3673b5", "ppi_tuned": "#00876c", "ppi_signed": "#ad368c"}
PROFILES = ("positive", "inverse", "uninformative")


def verify_historical_alignment(trials, manifests, seeds):
    """The extension must use the old realized targets, methods and group splits."""
    historical = ROOT/"reports/llmbar-audit"
    expected = json.loads((historical/"split_manifest.json").read_text(encoding="utf-8"))
    key = lambda row: (row["cohort"], row["seed"], row["labeled_fraction"])
    reference = {key(row): row for row in expected if row["seed"] < seeds}
    observed = {key(row): row for row in manifests if key(row) in reference}
    if observed != reference:
        raise ValueError("signed extension changed a historical instruction split")
    old = pd.read_csv(historical/"trials.csv")
    old = old.loc[old.seed < seeds]
    new = trials.loc[(trials.seed < 30) & (trials.method != "ppi_signed"), old.columns].copy()
    new.attrs = {}
    keys = ["cohort", "judge", "seed", "labeled_fraction", "method"]
    pd.testing.assert_frame_equal(old.sort_values(keys).reset_index(drop=True),
                                  new.sort_values(keys).reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-14)
    return {"historical_splits_verified": len(reference), "historical_method_trials_verified": len(old),
            "split_manifest_sha256": sha256(historical/"split_manifest.json"),
            "trials_sha256": sha256(historical/"trials.csv")}


def make_report(simulation, empirical, config, protocol_link):
    signed = simulation.loc[simulation.method == "ppi_signed"]
    lines = ["# Can an inverse proxy still help an audit?", "",
             f"[Retrospective protocol, equations and sources]({protocol_link})", "",
             "## Question and method", "",
             "The preceding LLMBar study found that order agreement can be negatively associated with "
             "reference correctness. The default `[0,1]` coefficient then falls back to zero. This exploratory "
             "follow-up asks whether allowing an explicitly selected `[-1,1]` range can use that inverse signal. "
             "It applies the established PPI++ mean-estimation principle; it is not a new estimator or a "
             "preregistered validation of a newly discovered proxy.", "",
             "The estimate remains `mean(Y_L) + lambda * (mean(F_U) - mean(F_L))`. "
             "The signed coefficient minimizes the same per-pool estimated variance as before over `[-1,1]`. "
             "Sign and magnitude use audit correctness and both proxy pools, never evaluation correctness. "
             "The numeric API requires `power_bounds=(-1,1)` to opt in; existing defaults, categorical "
             "win-rate behavior and finite-sample bounds are unchanged.", "",
             "Using a negative coefficient on `F` is algebraically equivalent to using a positive coefficient "
             "on `1-F`. For LLMBar this complements the entire agreement score, including invalid-score zeros; "
             "it is not a relabeling of canonical choices or literal valid disagreement. "
             "Normal intervals remain asymptotic. Optimizing estimated variance does not guarantee lower "
             "realized MSE or nominal finite-sample coverage.", "",
             "## Known-truth experiment", "",
             f"All 18 settings use {config['repetitions']:,} independent replications, master seed {config['seed']}. "
             "Prevalence is 0.5 or 0.95; proxy sensitivity/specificity is 0.95/0.90 (positive), "
             "0.05/0.10 (inverse), or 0.60/0.40 (uninformative). Audit sizes are 20, 200 and 1,000, "
             "with prediction pools ten times larger. Four methods share every draw and target the known "
             "population prevalence. The oracle coefficient is a diagnostic, never an estimator input.", "",
             "| Profile | Prevalence | Audit n | Method | RMSE | Bias | Coverage | Exact MC 95% interval | Width | Mean power |",
             "|---|---:|---:|---|---:|---:|---:|---|---:|---:|"]
    for row in simulation.itertuples():
        lines.append(f"| {row.profile} | {row.truth:.2f} | {row.n_labeled} | {NAMES[row.method]} | "
                     f"{row.rmse:.4f} | {row.bias:.4f} | {row.coverage:.3f} | "
                     f"[{row.coverage_mc_low:.3f}, {row.coverage_mc_high:.3f}] | {row.mean_width:.4f} | {row.mean_selected_power:.3f} |")
    gaps = simulation.pivot(index="scenario", columns="method", values="rmse")
    delta = gaps.ppi_signed-gaps.ppi_tuned
    tied = np.isclose(delta, 0, rtol=0, atol=1e-12)
    balanced_inverse = simulation.loc[(simulation.profile == "inverse") & (simulation.truth == .5)
                                     & (simulation.n_labeled == 20)].set_index("method")
    balanced_noise = simulation.loc[(simulation.profile == "uninformative") & (simulation.truth == .5)
                                   & (simulation.n_labeled == 20)].set_index("method")
    rare_inverse = signed.loc[(signed.profile == "inverse") & (signed.truth == .95)
                              & (signed.n_labeled == 20)].iloc[0]
    lines += ["", f"Signed tuning has lower observed RMSE than positive-range tuning in "
              f"{int(((delta < 0) & ~tied).sum())}/{len(delta)} settings, ties in {int(tied.sum())}, "
              f"and has higher RMSE in {int(((delta > 0) & ~tied).sum())}. These are simulation results with "
              "sampling uncertainty. The paired squared-error contrasts below use the same draws; "
              "a lower point estimate alone does not establish a resolved improvement.", "",
              f"At prevalence 0.5 with 20 audit labels, inverse-proxy RMSE changes from "
              f"{balanced_inverse.loc['ppi_tuned', 'rmse']:.4f} under positive tuning to "
              f"{balanced_inverse.loc['ppi_signed', 'rmse']:.4f} under signed tuning. For an uninformative "
              f"proxy at the same prevalence and budget, it instead changes from "
              f"{balanced_noise.loc['ppi_tuned', 'rmse']:.4f} to {balanced_noise.loc['ppi_signed', 'rmse']:.4f}. "
              "The larger coefficient family can exploit inverse signal, but also fit noise.", "",
              f"For the inverse proxy at prevalence 0.95 and 20 audit labels, signed interval coverage is "
              f"{100*rare_inverse.coverage:.1f}% (exact Monte Carlo interval "
              f"{100*rare_inverse.coverage_mc_low:.1f}%–{100*rare_inverse.coverage_mc_high:.1f}%). "
              "Point-estimation improvement does not resolve the small-audit normal-interval failure.", "",
              "| Profile | Prevalence | Audit n | Signed minus positive MSE | Paired MCSE | Signed minus audit MSE | Paired MCSE |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in signed.itertuples():
        lines.append(f"| {row.profile} | {row.truth:.2f} | {row.n_labeled} | "
                     f"{row.mse_difference_vs_positive:.6f} | {row.mse_difference_vs_positive_mcse:.6f} | "
                     f"{row.mse_difference_vs_human:.6f} | {row.mse_difference_vs_human_mcse:.6f} |")
    if config["plots"]:
        lines += ["", "The figure focuses on audit-only and the two tuned methods; the table retains all four methods.",
                  "", "![Signed correction error and coverage](signed_simulation_tradeoff.svg)"]
    lines += ["", "## LLMBar: identical fixed targets and budgets", "",
              f"The same certified 419-comparison, three-judge snapshot is analyzed with {config['split_seeds']} "
              "shared instruction-group seeds, fixed 25% evaluation groups and nested 20/40/60% audits. "
              "The preceding four methods remain in every export; signed tuning adds a fifth method. "
              "The runner verifies every overlapping historical split and all preceding method outputs "
              "before writing a report. All comparisons use original-order accuracy relative to curated "
              "LLMBar instruction-following references.", "",
              "| Cohort | Judge | Audit fraction | Method | MAE (pp) | RMSE (pp) | Mean power |",
              "|---|---|---:|---|---:|---:|---:|"]
    for row in empirical.itertuples():
        power = "--" if pd.isna(row.mean_selected_power) else f"{row.mean_selected_power:.3f}"
        lines.append(f"| {row.cohort} | {row.judge} | {row.labeled_fraction:.0%} | {NAMES[row.method]} | "
                     f"{100*row.mean_absolute_error:.3f} | {100*row.root_mean_squared_error:.3f} | {power} |")
    pivot = empirical.pivot(index=["cohort", "judge", "labeled_fraction"], columns="method", values="mean_absolute_error")
    difference = pivot.ppi_signed-pivot.ppi_tuned
    tied = np.isclose(difference, 0, rtol=0, atol=1e-12)
    lines += ["", f"Compared with positive-range tuning, signed tuning lowers MAE in "
              f"{int(((difference < 0) & ~tied).sum())}/{len(difference)} cells, ties in {int(tied.sum())}, "
              f"and increases MAE in {int(((difference > 0) & ~tied).sum())}. "
              "These dependent-split averages are descriptive. The correction range was motivated by earlier "
              "results on this corpus; this is not independent external validation."]
    if config["plots"]:
        lines += ["", "The figure focuses on audit-only and the two tuned methods; the table and exports retain all five methods.",
                  "", "![Signed LLMBar correction](signed_llmbar_changes.svg)"]
    lines += ["", "## Interpretation and limits", "",
              "- Restricting the coefficient to `[-1,1]` contains the iid population oracle for a binary proxy "
              "and bounded outcome. For continuous proxies or general grouped data it is a declared stabilizing "
              "constraint, not a universal optimality result.",
              "- Both tuned methods use the same audit for correction and coefficient estimation. They may have "
              "finite-sample bias, zero empirical variance or poor normal coverage. No finite-sample bounds are "
              "transferred to adaptive sign selection.",
              "- LLMBar normal interval widths are diagnostics, not coverage estimates for realized heldout accuracy. "
              "No independent-replication MCSE or significance test is attached to overlapping corpus splits.",
              "- Gold audit labels are shared across judges. Signed tuning consumes exactly the same audit labels "
              "and cached two-order judgments as positive tuning; scoring labels are separate. No new API calls "
              "or human labels were purchased, and no deployment cost saving is inferred.",
              "- Historical GPT-4, ChatGPT and LLaMA2 versions, curated references, and adversarial filtering "
              "remain the limits documented in the preceding LLMBar protocol. A negative coefficient here "
              "does not establish a universal inverse relationship.", "",
              "## Reproduce", "", "```bash", 'pip install -e ".[dev,report]"',
              "python examples/signed_power_study.py --output reports/reproduced/signed-power --plot", "```", "",
              "All simulation trials are retained in deterministic `simulation_trials.csv.gz`; empirical "
              "trials, paired errors and exact split IDs are also retained. Configuration records the complete "
              "generator and historical alignment checks. Source/artifact hashes cover the actual run. "
              "Use `--repetitions 5 --seeds 2` for an explicitly recorded smoke run.", ""]
    return "\n".join(lines)


def make_plots(simulation, empirical, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"svg.hashsalt": "judgecal-signed-v1", "font.size": 9})
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    markers = {0.5: "o", 0.95: "s"}
    for r, profile in enumerate(PROFILES):
        for prevalence in (.5, .95):
            for method in ("human_only", "ppi_tuned", "ppi_signed"):
                rows = simulation.loc[(simulation.profile == profile) & (simulation.truth == prevalence)
                                      & (simulation.method == method)].sort_values("n_labeled")
                label = f"{NAMES[method]}, p={prevalence}"
                style = "-" if prevalence == .5 else "--"
                axes[r, 0].plot(rows.n_labeled, rows.rmse, marker=markers[prevalence], linestyle=style,
                                color=COLORS[method], label=label)
                axes[r, 1].errorbar(rows.n_labeled, rows.coverage,
                                    yerr=[rows.coverage-rows.coverage_mc_low, rows.coverage_mc_high-rows.coverage],
                                    marker=markers[prevalence], linestyle=style, color=COLORS[method], capsize=3)
        for c in range(2):
            ax = axes[r, c]
            ax.set_xscale("log")
            ax.set_xticks([20, 200, 1000], labels=["20", "200", "1000"])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=.2)
            if r == 2:
                ax.set_xlabel("Audit observations (log scale)")
        axes[r, 0].set_ylabel(f"{profile.capitalize()} proxy\nRMSE")
        axes[r, 1].set_ylabel("Population coverage")
        axes[r, 1].axhline(.95, color="black", linewidth=.8, linestyle=":")
        axes[r, 1].set_ylim(0, 1.025)
    axes[0, 0].set_title("Point-estimation error")
    axes[0, 1].set_title("Normal 95% intervals; bars show exact MC intervals")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Signed coefficients: gains from inverse signal and small-audit limitations")
    save_plot(fig, output, "signed_simulation_tradeoff")
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), constrained_layout=True, sharex=True, sharey=True)
    for r, cohort in enumerate(COHORTS):
        for c, judge in enumerate(JUDGES):
            ax = axes[r, c]
            for method in ("human_only", "ppi_tuned", "ppi_signed"):
                rows = empirical.loc[(empirical.cohort == cohort) & (empirical.judge == judge)
                                     & (empirical.method == method)].sort_values("labeled_fraction")
                ax.plot(rows.labeled_fraction*100, rows.mean_absolute_error*100, marker="o",
                        color=COLORS[method], label=NAMES[method])
            ax.set_xticks([20, 40, 60])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
            if r == 0:
                ax.set_title(judge)
            if r == 2:
                ax.set_xlabel("Audit instructions (% of cohort)")
            if c == 0:
                ax.set_ylabel(f"{cohort}\nMean absolute error (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("LLMBar: signed and positive correction on identical fixed evaluation pools")
    save_plot(fig, output, "signed_llmbar_changes")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reports/signed-power")
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if args.repetitions < 2 or args.seed < 0 or args.seeds < 1:
        parser.error("repetitions>=2, seed>=0 and seeds>=1 are required")
    output = args.output
    generated = {"simulation_trials.csv.gz", "simulation_summary.csv", "llmbar_trials.csv", "llmbar_summary.csv",
                 "llmbar_paired_differences.csv", "split_manifest.json", "config.json", "REPORT.md"}
    if args.plot:
        generated |= {f"{stem}.{suffix}" for stem in ("signed_simulation_tradeoff", "signed_llmbar_changes")
                      for suffix in ("png", "svg")}
    if output.exists() and {p.name for p in output.iterdir()} - generated - {"reproducibility.json"}:
        raise ValueError("output contains stale or unrelated artifacts; use a fresh directory")
    simulation_trials, simulation_summary = signed_power_study(args.repetitions, args.seed)
    simulation_config = simulation_trials.attrs.pop("configuration")
    simulation_trials.attrs = {}
    labels_path = ROOT/"reports/llmbar/labels.csv"
    labels, source = read_verified_labels(labels_path)
    empirical_trials, empirical_summary = judge_accuracy_audit(labels, seeds=tuple(range(args.seeds)), include_signed=True)
    manifests = empirical_trials.attrs.pop("split_manifest")
    empirical_protocol = empirical_trials.attrs.pop("audit_protocol")
    empirical_trials.attrs = {}
    alignment = verify_historical_alignment(empirical_trials, manifests, args.seeds)
    paired = paired_method_differences(empirical_trials)
    positive = empirical_trials.loc[empirical_trials.method == "ppi_tuned",
                                    ["cohort", "judge", "seed", "labeled_fraction", "absolute_error", "squared_error"]]
    keys = ["cohort", "judge", "seed", "labeled_fraction"]
    differences = empirical_trials.merge(positive, on=keys, suffixes=("", "_positive"), validate="many_to_one")
    differences["absolute_error_difference_vs_positive"] = differences.absolute_error-differences.absolute_error_positive
    differences["squared_error_difference_vs_positive"] = differences.squared_error-differences.squared_error_positive
    paired = paired.merge(differences[keys+["method", "absolute_error_difference_vs_positive",
                                           "squared_error_difference_vs_positive"]],
                          on=keys+["method"], validate="one_to_one")
    config = {"protocol": "signed-proxy-correction-v1", "repetitions": args.repetitions, "seed": args.seed,
              "split_seeds": args.seeds, "plots": args.plot, "simulation": simulation_config,
              "empirical": empirical_protocol, "historical_alignment": alignment,
              "labels_sha256": source["labels_sha256"], "source_revision": source["source_revision"]}
    output.mkdir(parents=True, exist_ok=True)
    with (output/"simulation_trials.csv.gz").open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, filename="", mode="wb", mtime=0) as compressed:
            compressed.write(simulation_trials.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    for name, frame in (("simulation_summary", simulation_summary), ("llmbar_trials", empirical_trials),
                        ("llmbar_summary", empirical_summary), ("llmbar_paired_differences", paired)):
        frame.to_csv(output/f"{name}.csv", index=False, lineterminator="\n")
    write_json(output/"split_manifest.json", manifests, compact_records=True)
    write_json(output/"config.json", config)
    link = os.path.relpath(ROOT/"docs/SIGNED_POWER_PROTOCOL.md", output).replace("\\", "/")
    (output/"REPORT.md").write_bytes(make_report(simulation_summary, empirical_summary, config, link).encode("utf-8"))
    if args.plot:
        make_plots(simulation_summary, empirical_summary, output)
    sources = sorted((ROOT/"src/judgecal").glob("*.py")) + [Path(__file__), ROOT/"examples/research_study.py",
                ROOT/"examples/llmbar_audit.py", ROOT/"docs/SIGNED_POWER_PROTOCOL.md", ROOT/"docs/METHODS.md", ROOT/"pyproject.toml"]
    manifest = {"python": platform.python_version(), "platform": platform.system(),
                "packages": {name: version(name) for name in ("judgecal", "numpy", "scipy", "pandas")},
                "input_labels_sha256": sha256(labels_path), "input_provenance_sha256": sha256(labels_path.with_name("dataset.json")),
                "historical_alignment": alignment,
                "source_text_normalization": "CRLF to LF before SHA-256 (matches Git text attributes)",
                "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                  for p in sources},
                "artifacts_sha256": {name: sha256(output/name) for name in sorted(generated)}}
    if args.plot:
        manifest["packages"]["matplotlib"] = version("matplotlib")
    write_json(output/"reproducibility.json", manifest)
    print(f"Wrote {len(simulation_trials)} synthetic and {len(empirical_trials)} empirical trials; "
          f"verified {alignment['historical_splits_verified']} historical splits.")


if __name__ == "__main__":
    main()
