"""Compare population inference and prediction of a random held-out mean."""
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

from judgecal.estimand_study import run_estimand_study
from judgecal.judge_audit import judge_accuracy_audit
from llmbar_audit import COHORTS, JUDGES, read_verified_labels, save_plot
from research_study import ROOT, sha256, write_json


PROFILES = ("positive", "inverse", "uninformative")
KEYS = ["cohort", "judge", "seed", "labeled_fraction"]


def historical_alignment(trials, manifests, seeds):
    previous = ROOT/"reports/signed-power"
    key = lambda row: (row["cohort"], row["seed"], row["labeled_fraction"])
    reference = {key(row): row for row in json.loads((previous/"split_manifest.json").read_text()) if row["seed"] < seeds}
    observed = {key(row): row for row in manifests if key(row) in reference}
    if observed != reference:
        raise ValueError("target sensitivity changed a historical instruction split")
    old = pd.read_csv(previous/"llmbar_trials.csv")
    old = old.loc[old.seed < seeds]
    new = trials.loc[(trials.seed < 30) & (trials.method != "audit_residual"), old.columns].copy()
    new.attrs = {}
    sort = KEYS+["method"]
    pd.testing.assert_frame_equal(old.sort_values(sort).reset_index(drop=True), new.sort_values(sort).reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-14)
    return {"historical_splits_verified": len(reference), "historical_method_trials_verified": len(old),
            "split_manifest_sha256": sha256(previous/"split_manifest.json"),
            "trials_sha256": sha256(previous/"llmbar_trials.csv")}


def paired_empirical(trials):
    result = trials[KEYS+["method", "split_sha256", "absolute_error", "squared_error"]].copy()
    for baseline, method in (("human", "human_only"), ("population", "ppi_signed")):
        reference = trials.loc[trials.method == method, KEYS+["absolute_error", "squared_error"]]
        reference = reference.rename(columns={name: f"{name}_{baseline}" for name in ("absolute_error", "squared_error")})
        result = result.merge(reference, on=KEYS, validate="many_to_one")
        for metric in ("absolute_error", "squared_error"):
            result[f"{metric}_difference_vs_{baseline}"] = result[metric]-result[f"{metric}_{baseline}"]
    return result


def paired_simulation(trials):
    keys = ["scenario", "repetition"]
    pool = trials.loc[trials.method == "pool_tuned"]
    population = trials.loc[trials.method == "population_tuned", keys+["population_error", "pool_error"]]
    paired = pool.merge(population, on=keys, suffixes=("", "_population_rule"), validate="one_to_one")
    rows = []
    for scenario, group in paired.groupby("scenario", sort=False):
        row = {"scenario": scenario, "profile": group.profile.iloc[0], "prevalence": group.prevalence.iloc[0],
               "n_labeled": int(group.n_labeled.iloc[0]), "n_unlabeled": int(group.n_unlabeled.iloc[0]),
               "prediction_audit_ratio": float(group.n_unlabeled.iloc[0]/group.n_labeled.iloc[0]),
               "repetitions": len(group)}
        for target in ("population", "pool"):
            delta = group[f"{target}_error"]**2-group[f"{target}_error_population_rule"]**2
            row[f"{target}_mse_difference"] = float(delta.mean())
            row[f"{target}_mse_difference_mcse"] = float(delta.std(ddof=1)/np.sqrt(len(delta)))
        rows.append(row)
    return pd.DataFrame(rows)


def make_report(summary, pairs, empirical, config, protocol_link):
    lines = ["# The target changes the useful correction", "",
             f"[Derivation, primary sources and retrospective protocol]({protocol_link})", "",
             "## Two different questions", "",
             "A population mean `E[Y]` and the actual mean `mean(Y_U)` of a held-out pool are different targets. "
             "This study compares the same corrected point estimates against both, using explicitly matched "
             "uncertainty calculations. The existing population PPI API retains its stated meaning; this is "
             "target sensitivity, not a claim that its coefficient formula was wrong.", "",
             "For the estimate `mean(Y_L) + lambda*(mean(F_U)-mean(F_L))`, let `R=Y-lambda*F`. "
             "Under independent, identically distributed pools, population error variance is "
             "`Var(R)/n + lambda² Var(F)/N`, while random-pool prediction error variance is "
             "`(1/n+1/N) Var(R)`. With `b=Cov(Y,F)/Var(F)`, their unrestricted oracle coefficients "
             "are `N/(n+N)*b` and `b`. The estimate and held-out target share proxy information; "
             "their covariance cannot be ignored. A perfect proxy illustrates the distinction: coefficient "
             "one recovers the pool mean exactly, whereas population shrinkage combines both samples.", "",
             "Both fitted coefficient ranges are fixed at `[-1,1]`. Population tuning uses the existing "
             "per-pool criterion. Pool tuning minimizes audit residual variance; it never reads held-out "
             "outcomes. The new IID prediction interval estimates marginal error over draws of both pools. "
             "It does not promise coverage conditional on a particular frozen pool or proxy composition.", "",
             "## Known-truth target comparison", "",
             f"The run uses {config['repetitions']:,} independent replications per setting and master seed "
             f"{config['seed']}. All 36 settings cross prevalence 0.5/0.95, positive/inverse/uninformative "
             "binary proxy laws, audit sizes 20/200, and prediction/audit ratios 0.5/1/10. Four methods "
             "share each draw: audit only, fixed coefficient one, signed population tuning and signed "
             "audit-residual tuning. The simulation retains held-out outcomes strictly for scoring.", "",
             "Every method is scored against both named targets. Each also retains both target-matched "
             "normal uncertainty calculations. Same-audit fitted coefficients remain plug-in asymptotic "
             "methods: exact Monte Carlo intervals quantify simulation coverage uncertainty, not exact "
             "validity of the underlying normal intervals. Coverage scoring allows a numerical margin "
             "of `8 * float64_epsilon * max(1, |lower|, |upper|, |truth|)` at either endpoint. "
             "This avoids roundoff-only misses for algebraically exact predictions; it changes no "
             "point, interval endpoint, interval width or variance.", "",
             "The table subtracts population-tuned squared error from pool-tuned squared error **within "
             "the same target and draw**. Negative differences favor pool tuning. Compare each contrast "
             "with its paired Monte Carlo standard error; values across different targets do not establish "
             "a superiority claim.", "",
             "| Proxy | Prevalence | Audit n | N/n | Population-target MSE difference | Paired MCSE | Pool-target MSE difference | Paired MCSE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in pairs.itertuples():
        lines.append(f"| {row.profile} | {row.prevalence:.2f} | {row.n_labeled} | {row.prediction_audit_ratio:g} | "
                     f"{row.population_mse_difference:.6f} | {row.population_mse_difference_mcse:.6f} | "
                     f"{row.pool_mse_difference:.6f} | {row.pool_mse_difference_mcse:.6f} |")
    if config["plots"]:
        lines += ["", "![Within-target paired MSE comparison](estimand_tradeoff.svg)", "",
                  "Bars show one paired Monte Carlo standard error; they are pointwise simulation diagnostics."]
    selected = summary.set_index(["scenario", "method"])
    illustration = summary.loc[(summary.profile == "positive") & (summary.prevalence == .5)
                               & (summary.n_labeled == 200) & (summary.n_unlabeled == 100)].set_index("method")
    small = summary.loc[(summary.profile == "positive") & (summary.prevalence == .95)
                        & (summary.n_labeled == 20) & (summary.n_unlabeled == 10)].set_index("method")
    lines += ["", "For the positive proxy with prevalence 0.5, 200 audit rows and 100 prediction rows, "
              f"switching from population tuning to pool tuning changes population RMSE from "
              f"{illustration.loc['population_tuned', 'population_rmse']:.4f} to {illustration.loc['pool_tuned', 'population_rmse']:.4f}, "
              f"while pool-target RMSE changes from {illustration.loc['population_tuned', 'pool_rmse']:.4f} "
              f"to {illustration.loc['pool_tuned', 'pool_rmse']:.4f}. The two within-target comparisons have opposite directions.", "",
              "Matching a target's oracle criterion does not guarantee finite-sample gains after fitting. "
              "For the positive proxy at prevalence 0.95 with 20 audit and 10 prediction rows, "
              f"pool-target RMSE changes from {small.loc['population_tuned', 'pool_rmse']:.4f} "
              f"to {small.loc['pool_tuned', 'pool_rmse']:.4f}; the pool-tuned marginal prediction interval "
              f"covers in {100*small.loc['pool_tuned', 'pool_coverage']:.1f}% of replications. "
              "The full grid and its Monte Carlo uncertainty are retained below."]
    lines += ["", "### Matched interval coverage", "",
              "Population-tuned estimates below use population confidence intervals; pool-tuned estimates "
              "use marginal prediction intervals for the random pool mean. Brackets show exact pointwise "
              "95% Monte Carlo intervals for the observed coverage rate. The nominal target is 95% for both.", "",
              "| Proxy | Prevalence | Audit n | N/n | Population CI coverage [MC interval] | Random-pool PI coverage [MC interval] |",
              "|---|---:|---:|---:|---|---|"]
    for row in pairs.itertuples():
        pop = selected.loc[(row.scenario, "population_tuned")]
        pool = selected.loc[(row.scenario, "pool_tuned")]
        lines.append(f"| {row.profile} | {row.prevalence:.2f} | {row.n_labeled} | {row.prediction_audit_ratio:g} | "
                     f"{pop.population_coverage:.3f} [{pop.population_coverage_mc_low:.3f}, {pop.population_coverage_mc_high:.3f}] | "
                     f"{pool.pool_coverage:.3f} [{pool.pool_coverage_mc_low:.3f}, {pool.pool_coverage_mc_high:.3f}] |")
    lines += ["", "All four methods' RMSE, bias, interval width, matched coverage and exact coverage "
              "Monte Carlo intervals are retained in `simulation_summary.csv`; every draw is in "
              "`simulation_trials.csv.gz`. No estimate, interval or failed small-audit case is clipped or dropped.", "",
              "## LLMBar point-error sensitivity", "",
              "The complete original comparison panel, invalid-output policy, instruction grouping, "
              "audit costs and fixed heldout targets are unchanged. A sixth method minimizes **audit "
              "cluster residual variance**. This is a point-only empirical candidate; unequal instruction "
              "groups do not inherit the IID pool prediction interval. Its interval and standard-error "
              "fields are left empty for this method in the shared CSV. The preceding population interval fields retain their old scope.", "",
              f"This run uses {config['split_seeds']} seeds and verifies every overlapping historical split "
              "and all five preceding methods before export. The comparison is retrospective on the same "
              "curated, historical-model benchmark. Its errors concern reference correctness, not latent truth.", "",
              "| Cohort | Judge | Audit fraction | Gold audit MAE (pp) | Signed population MAE (pp) | Audit-residual MAE (pp) | Residual minus population (pp) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    pivot = empirical.pivot(index=["cohort", "judge", "labeled_fraction"], columns="method", values="mean_absolute_error")
    for (cohort, judge, budget), row in pivot.iterrows():
        lines.append(f"| {cohort} | {judge} | {budget:.0%} | {100*row.human_only:.3f} | "
                     f"{100*row.ppi_signed:.3f} | {100*row.audit_residual:.3f} | {100*(row.audit_residual-row.ppi_signed):+.3f} |")
    difference = pivot.audit_residual-pivot.ppi_signed
    tied = np.isclose(difference, 0, rtol=0, atol=1e-12)
    lines += ["", f"Audit-residual tuning has lower MAE in {int(((difference < 0) & ~tied).sum())}/{len(difference)} "
              f"cells, ties in {int(tied.sum())}, and has higher MAE in {int(((difference > 0) & ~tied).sum())}. "
              "The six-method exports retain all outcomes. These are descriptive paired comparisons, "
              "with no empirical coverage or independent-deployment significance claim."]
    if config["plots"]:
        lines += ["", "![LLMBar audit-residual tuning](pool_target_llmbar.svg)"]
    lines += ["", "## Sampling and interpretation limits", "",
              "- A marginal prediction interval for a random heldout mean is different from a population "
              "confidence interval and from a guarantee conditional on every realized pool. Outcome or "
              "residual shift invalidates the simple transport of audit residual moments.",
              "- Uniform disjoint row partitions of a fixed finite population have a related fixed-coefficient "
              "variance identity: finite-population corrections cancel with negative cross-pool covariance. "
              "An exhaustive small-population test verifies that identity. It is joint randomization over "
              "both pools, not a conditional guarantee after freezing U, and does not make adaptive tuning exact.",
              "- Instruction sampling with unequal group sizes creates ratio estimators. The empirical "
              "audit-residual criterion is not advertised as the exact finite-corpus conditional optimum. "
              "No IID pool interval is applied to those grouped rows.",
              "- Original PPI already studies finite-population calibration. This project contributes an "
              "explicit, reproducible comparison of targets and assumptions, not new finite-sample theory.",
              "- Existing gold labels are shared across judges, and both tuned empirical methods consume "
              "the same cached inputs. There are no new model calls, bought labels or inferred dollar savings.", "",
              "## Reproduce", "", "```bash", 'pip install -e ".[dev,report]"',
              "python examples/estimand_study.py --output reports/reproduced/estimand --plot", "```", "",
              "Use `--repetitions 5 --seeds 2` for a recorded smoke run. Configuration includes generators, "
              "target definitions and historical alignment; the manifest pins inputs, source and every artifact.", ""]
    return "\n".join(lines)


def make_plots(pairs, empirical, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"svg.hashsalt": "judgecal-estimand-v1", "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True, sharex=True)
    styles = ((.5, 20, "#277da1", "o", "-"), (.5, 200, "#277da1", "s", "--"),
              (.95, 20, "#a64c83", "o", "-"), (.95, 200, "#a64c83", "s", "--"))
    for r, target in enumerate(("population", "pool")):
        for c, profile in enumerate(PROFILES):
            ax = axes[r, c]
            for prevalence, n, color, marker, line in styles:
                rows = pairs.loc[(pairs.profile == profile) & (pairs.prevalence == prevalence)
                                 & (pairs.n_labeled == n)].sort_values("prediction_audit_ratio")
                ax.errorbar(rows.prediction_audit_ratio, rows[f"{target}_mse_difference"]*10000,
                            yerr=rows[f"{target}_mse_difference_mcse"]*10000, marker=marker, linestyle=line,
                            color=color, capsize=3, label=f"p={prevalence}, n={n}")
            ax.axhline(0, color="black", linewidth=.8)
            ax.set_xscale("log")
            ax.set_xticks([.5, 1, 10], labels=["0.5", "1", "10"])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=.2)
            if r == 0:
                ax.set_title(f"{profile.capitalize()} proxy")
            if r == 1:
                ax.set_xlabel("Prediction / audit size (log scale)")
            if c == 0:
                ax.set_ylabel(f"{target.capitalize()} target\nMSE difference (pp²)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Pool-tuned minus population-tuned error: each target has its own comparison")
    save_plot(fig, output, "estimand_tradeoff")
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), constrained_layout=True, sharex=True, sharey=True)
    methods = (("human_only", "Gold audit only", "#525a66"), ("ppi_signed", "Signed population", "#00876c"),
               ("audit_residual", "Audit-residual point", "#ad368c"))
    for r, cohort in enumerate(COHORTS):
        for c, judge in enumerate(JUDGES):
            ax = axes[r, c]
            for method, label, color in methods:
                rows = empirical.loc[(empirical.cohort == cohort) & (empirical.judge == judge)
                                     & (empirical.method == method)].sort_values("labeled_fraction")
                ax.plot(rows.labeled_fraction*100, rows.mean_absolute_error*100, marker="o", color=color, label=label)
            ax.set_xticks([20, 40, 60]); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.2)
            if r == 0:
                ax.set_title(judge)
            if r == 2:
                ax.set_xlabel("Audit instructions (% of cohort)")
            if c == 0:
                ax.set_ylabel(f"{cohort}\nHeldout-accuracy MAE (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("LLMBar point estimates: unchanged target, grouping and audit costs")
    save_plot(fig, output, "pool_target_llmbar")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reports/estimand")
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if args.repetitions < 2 or args.seed < 0 or args.seeds < 1:
        parser.error("repetitions>=2, seed>=0 and seeds>=1 are required")
    output = args.output
    generated = {"simulation_trials.csv.gz", "simulation_summary.csv", "simulation_paired_differences.csv",
                 "llmbar_trials.csv", "llmbar_summary.csv", "llmbar_paired_differences.csv",
                 "split_manifest.json", "config.json", "REPORT.md"}
    if args.plot:
        generated |= {f"{stem}.{ext}" for stem in ("estimand_tradeoff", "pool_target_llmbar") for ext in ("png", "svg")}
    if output.exists() and {p.name for p in output.iterdir()} - generated - {"reproducibility.json"}:
        raise ValueError("output contains stale or unrelated artifacts; use a fresh directory")
    synthetic, summary = run_estimand_study(args.repetitions, args.seed)
    simulation_config = synthetic.attrs.pop("configuration"); synthetic.attrs = {}
    pairs = paired_simulation(synthetic)
    labels_path = ROOT/"reports/llmbar/labels.csv"
    labels, source = read_verified_labels(labels_path)
    empirical, empirical_summary = judge_accuracy_audit(labels, seeds=tuple(range(args.seeds)),
                                                       include_signed=True, include_pool_tuned=True)
    manifests = empirical.attrs.pop("split_manifest"); protocol = empirical.attrs.pop("audit_protocol"); empirical.attrs = {}
    alignment = historical_alignment(empirical, manifests, args.seeds)
    config = {"protocol": "population-versus-heldout-target-v1", "repetitions": args.repetitions, "seed": args.seed,
              "split_seeds": args.seeds, "plots": args.plot, "simulation": simulation_config, "empirical": protocol,
              "historical_alignment": alignment, "labels_sha256": source["labels_sha256"]}
    output.mkdir(parents=True, exist_ok=True)
    with (output/"simulation_trials.csv.gz").open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, filename="", mode="wb", mtime=0) as compressed:
            compressed.write(synthetic.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    for name, table in (("simulation_summary", summary), ("simulation_paired_differences", pairs),
                        ("llmbar_trials", empirical), ("llmbar_summary", empirical_summary),
                        ("llmbar_paired_differences", paired_empirical(empirical))):
        table.to_csv(output/f"{name}.csv", index=False, lineterminator="\n")
    write_json(output/"split_manifest.json", manifests, compact_records=True); write_json(output/"config.json", config)
    link = os.path.relpath(ROOT/"docs/ESTIMAND_PROTOCOL.md", output).replace("\\", "/")
    (output/"REPORT.md").write_bytes(make_report(summary, pairs, empirical_summary, config, link).encode("utf-8"))
    if args.plot:
        make_plots(pairs, empirical_summary, output)
    sources = sorted((ROOT/"src/judgecal").glob("*.py")) + [Path(__file__), ROOT/"examples/research_study.py",
                ROOT/"examples/llmbar_audit.py", ROOT/"docs/ESTIMAND_PROTOCOL.md", ROOT/"docs/METHODS.md", ROOT/"pyproject.toml"]
    manifest = {"python": platform.python_version(), "platform": platform.system(),
                "packages": {name: version(name) for name in ("judgecal", "numpy", "scipy", "pandas")},
                "input_labels_sha256": sha256(labels_path), "input_provenance_sha256": sha256(labels_path.with_name("dataset.json")),
                "historical_alignment": alignment, "source_text_normalization": "CRLF to LF before SHA-256",
                "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                  for p in sources},
                "artifacts_sha256": {name: sha256(output/name) for name in sorted(generated)}}
    if args.plot:
        manifest["packages"]["matplotlib"] = version("matplotlib")
    write_json(output/"reproducibility.json", manifest)
    print(f"Wrote {len(synthetic)} synthetic and {len(empirical)} empirical trials, retaining both targets.")


if __name__ == "__main__":
    main()
