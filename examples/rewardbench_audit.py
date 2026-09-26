"""Reproduce the pinned RewardBench cross-judge agreement audit offline."""
from __future__ import annotations

import argparse
import gzip
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from judgecal.cross_judge_audit import cross_judge_accuracy_audit
from judgecal.rewardbench import REWARDBENCH_JUDGES as JUDGES
from research_study import ROOT, sha256, write_json


LABELS_SHA256 = "7dc7d8aad832f470a2831df5011fd7e8e1c670996a225b5d91c81f760350120f"
PROVENANCE_SHA256 = "3ff9bd62f5a939c645b1a18e75cacf83bbe4e37a3d6e26747980894f993062f2"
COHORTS = ("NonLLMBar", "All", "LLMBar")
METHODS = ("raw_proxy", "human_only", "ppi", "ppi_tuned", "ppi_signed", "audit_residual")
NAMES = dict(zip(METHODS, ("Raw agreement", "Reference audit only", "PPI (power 1)",
                         "Positive population tuning", "Signed population tuning", "Audit-residual tuning")))
COLORS = dict(zip(METHODS, ("#b36b35", "#5e6572", "#3575bb", "#009575", "#825bc2", "#ce4660")))
SHORT_JUDGES = {"openai/gpt-4o-2024-08-06": "GPT-4o (2024-08-06)",
                "openai/gpt-4o-mini-2024-07-18": "GPT-4o mini (2024-07-18)"}
KEYS = ["cohort", "target_judge", "auxiliary_judge", "seed", "labeled_fraction"]


def read_verified_labels(path):
    path = Path(path)
    metadata = json.loads(path.with_name("dataset.json").read_text(encoding="utf-8"))
    if sha256(path) != LABELS_SHA256 or metadata.get("labels_sha256") != LABELS_SHA256:
        raise ValueError("input differs from the certified complete RewardBench labels")
    if sha256(path.with_name("dataset.json")) != PROVENANCE_SHA256:
        raise ValueError("provenance differs from the certified RewardBench manifest")
    return pd.read_csv(path, dtype={"sample_id": str, "instruction_id": str}), metadata


def cohort_frames(frame):
    is_llmbar = frame.subset.str.startswith("llmbar-")
    yield "NonLLMBar", frame.loc[~is_llmbar]
    yield "All", frame
    yield "LLMBar", frame.loc[is_llmbar]


def corpus_diagnostics(labels):
    records = []
    for cohort, rows in list(cohort_frames(labels)) + list(labels.groupby("subset", sort=True)):
        for judge, group in rows.groupby("judge", sort=True):
            y = group.reference_correct.to_numpy(dtype=float)
            f = group.agreement_proxy.to_numpy(dtype=float)
            agree = f == 1
            covariance = float(np.mean((y-y.mean())*(f-f.mean())))
            records.append({"cohort": cohort, "target_judge": judge,
                            "auxiliary_judge": group.auxiliary_judge.iloc[0],
                            "comparisons": len(group), "prompt_groups": group.instruction_id.nunique(),
                            "reference_accuracy": float(y.mean()), "cross_judge_agreement": float(f.mean()),
                            "proxy_minus_accuracy": float(f.mean()-y.mean()),
                            "agree_count": int(agree.sum()), "agree_but_wrong": int((agree & (y == 0)).sum()),
                            "accuracy_given_agreement": float(y[agree].mean()) if agree.any() else np.nan,
                            "accuracy_given_disagreement": float(y[~agree].mean()) if (~agree).any() else np.nan,
                            "outcome_proxy_covariance": covariance})
    return pd.DataFrame(records)


def paired_differences(trials):
    result = trials[KEYS + ["method", "split_sha256", "absolute_error", "squared_error"]].copy()
    for name, method in (("reference_audit", "human_only"), ("signed_population", "ppi_signed")):
        baseline = trials.loc[trials.method == method, KEYS + ["absolute_error", "squared_error"]]
        result = result.merge(baseline, on=KEYS, suffixes=("", "_baseline"), validate="many_to_one")
        for metric in ("absolute_error", "squared_error"):
            result[f"{metric}_difference_vs_{name}"] = result[metric]-result.pop(f"{metric}_baseline")
    return result.drop(columns=["absolute_error", "squared_error"])


def shared_costs(trials):
    """Union of logical inputs across both directions, not a sum of repeated reads."""
    records = []
    for key, rows in trials.groupby(["cohort", "seed", "labeled_fraction", "method"], sort=True):
        if len(rows) != 2 or set(rows.target_judge) != set(JUDGES):
            raise ValueError("cost union requires both designated judge directions")
        for field in ("n_labeled", "n_evaluation", "reference_labels_used", "split_sha256"):
            if rows[field].nunique() != 1:
                raise ValueError("judge directions must share identical pools and reference costs")
        n, target = int(rows.n_labeled.iloc[0]), int(rows.n_evaluation.iloc[0])
        method = key[-1]
        reference = int(rows.reference_labels_used.iloc[0])
        cached = 2*target if method == "raw_proxy" else 2*n if method == "human_only" else 2*(n+target)
        records.append(dict(zip(("cohort", "seed", "labeled_fraction", "method"), key),
                            split_sha256=rows.split_sha256.iloc[0], n_audit_comparisons=n,
                            n_evaluation_comparisons=target, inference_reference_labels_union=reference,
                            inference_cached_judgments_union=cached,
                            scoring_reference_labels_union=target, scoring_cached_judgments_union=2*target,
                            inference_and_scoring_reference_union=reference+target,
                            inference_and_scoring_cached_union=2*target if method == "raw_proxy" else 2*(n+target)))
    return pd.DataFrame(records)


def encode_split_manifest(manifests):
    """Store each content identity once and losslessly index every exact pool."""
    pools = ("labeled", "evaluation", "unused")
    catalogs = {kind: sorted({identity for m in manifests for pool in pools for identity in m[f"{pool}_{kind}"]})
                for kind in ("sample_ids", "instruction_ids")}
    indexes = {kind: {identity: i for i, identity in enumerate(catalog)} for kind, catalog in catalogs.items()}
    indexed = []
    for manifest in manifests:
        item = dict(manifest)
        for kind in catalogs:
            for pool in pools:
                field = f"{pool}_{kind}"
                item[field] = [indexes[kind][identity] for identity in item[field]]
        indexed.append(item)
    return {"schema": "indexed-content-identities-v1", **catalogs, "manifests": indexed}


def decode_split_manifest(encoded):
    """Recover the original string IDs and their order from the public artifact."""
    if encoded.get("schema") != "indexed-content-identities-v1":
        raise ValueError("unsupported split manifest encoding")
    manifests = []
    for source in encoded["manifests"]:
        item = dict(source)
        for kind in ("sample_ids", "instruction_ids"):
            for pool in ("labeled", "evaluation", "unused"):
                field = f"{pool}_{kind}"
                indexes = item[field]
                if any(type(i) is not int or not 0 <= i < len(encoded[kind]) for i in indexes):
                    raise ValueError("invalid split identity index")
                item[field] = [encoded[kind][i] for i in indexes]
        manifests.append(item)
    return manifests


def save_plot(fig, output, name):
    fig.savefig(output/f"{name}.png", dpi=180)
    fig.savefig(output/f"{name}.svg", metadata={"Date": None})
    path = output/f"{name}.svg"
    path.write_bytes(b"\n".join(line.rstrip() for line in path.read_bytes().splitlines())+b"\n")


def make_plots(summary, diagnostics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"svg.hashsalt": "judgecal-rewardbench-v1", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True, constrained_layout=True)
    for ax, cohort in zip(axes, COHORTS):
        rows = diagnostics.loc[diagnostics.cohort == cohort].set_index("target_judge").loc[list(JUDGES)]
        x = np.arange(2)
        ax.bar(x-.18, rows.reference_accuracy, .36, color="#3575bb", label="Reference accuracy")
        ax.bar(x+.18, rows.cross_judge_agreement, .36, color="#b36b35", label="Shared agreement proxy")
        ax.set(title=cohort + (" (primary)" if cohort == "NonLLMBar" else " (sensitivity)"),
               xticks=x, xticklabels=["GPT-4o", "GPT-4o mini"], ylim=(0, 1.03))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Fraction of cached comparisons")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("One shared agreement signal, two reference-accuracy targets", fontsize=13)
    save_plot(fig, output, "agreement_vs_accuracy"); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for r, cohort in enumerate(COHORTS):
        for c, judge in enumerate(JUDGES):
            ax = axes[r, c]
            for method in METHODS:
                data = summary.loc[(summary.cohort == cohort) & (summary.target_judge == judge)
                                   & (summary.method == method)].sort_values("labeled_fraction")
                ax.plot(100*data.labeled_fraction, 100*data.mean_absolute_error,
                        marker="o", color=COLORS[method], label=NAMES[method], linewidth=1.8)
            ax.set(title=f"{cohort}: {SHORT_JUDGES[judge]}", xticks=[20, 40, 60], ylim=(0, None))
            ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.2)
            if c == 0:
                ax.set_ylabel("Mean absolute error (percentage points)")
            if r == 2:
                ax.set_xlabel("Audited prompt groups (% of total cohort)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Fixed held-out reference accuracy; shared splits and reference budgets", fontsize=13)
    save_plot(fig, output, "cross_judge_audit_budget"); plt.close(fig)


def make_report(summary, diagnostics, config, links):
    lines = ["# Cross-judge agreement as an accuracy-audit proxy", "",
             f"[Retrospective protocol]({links['protocol']}) | [Source notices]({links['notice']})", "",
             "## Question and design", "",
             "Does the same cross-judge agreement signal help estimate either judge's benchmark-reference "
             "accuracy? This study fixes the GPT-4o 2024-08-06 and GPT-4o mini 2024-07-18 caches, both "
             "target/auxiliary directions, and all 2,985 comparisons before fitting. It makes no new model calls.", "",
             "The binary proxy is equality of the two canonical choices. It can be reconstructed as equality "
             "of two binary correctness bits because each judge selects one of the same two answers. Reversing "
             "the common reference flips both bits and leaves agreement unchanged. A single auxiliary "
             "correctness bit would leak gold and is never used as the proxy. Both judges can agree on a wrong answer.", "",
             "NonLLMBar is the primary component: 2,566 comparisons and 2,315 exact-prompt groups. All is "
             "mixture sensitivity (2,985/2,733), and LLMBar is overlap sensitivity (419/418). The LLMBar "
             "component revisits source material represented in the preceding benchmark; it is not independent "
             "new evidence. Comparisons have equal weight; these row means are not the official weighted leaderboard score.", "",
             "## Full-corpus diagnostics", "",
             "These reference-based descriptions never enter proxy construction or coefficient fitting. "
             "The complete 23-subset breakdown is retained in `corpus_diagnostics.csv`.", "",
             "| Cohort | Target judge | Comparisons | Accuracy | Shared agreement | Agree but wrong | Accuracy if agree / disagree |",
             "|---|---|---:|---:|---:|---:|---|"]
    for cohort in COHORTS:
        for row in diagnostics.loc[diagnostics.cohort == cohort].itertuples():
            lines.append(f"| {cohort} | {SHORT_JUDGES[row.target_judge]} | {row.comparisons} | "
                         f"{row.reference_accuracy:.3f} | {row.cross_judge_agreement:.3f} | {row.agree_but_wrong} | "
                         f"{row.accuracy_given_agreement:.3f} / {row.accuracy_given_disagreement:.3f} |")
    if config["plots"]:
        lines += ["", "![Reference accuracy and shared agreement](agreement_vs_accuracy.svg)"]
    primary = diagnostics.loc[diagnostics.cohort == "NonLLMBar"].set_index("target_judge")
    overlap = diagnostics.loc[diagnostics.cohort == "LLMBar"].set_index("target_judge")
    full_model, small_model = JUDGES
    lines += ["", f"In the primary component, the shared agreement rate is {100*primary.loc[full_model, 'cross_judge_agreement']:.2f}%, "
              f"while the two target accuracies are {100*primary.loc[full_model, 'reference_accuracy']:.2f}% and "
              f"{100*primary.loc[small_model, 'reference_accuracy']:.2f}%. "
              f"Both models agree on the reference-incorrect answer in {int(primary.loc[full_model, 'agree_but_wrong'])} comparisons. "
              "Equal proxy values therefore do not imply equal accuracy or equal usefulness for the two targets.", "",
              "The relationship also changes across components: for GPT-4o in LLMBar, accuracy given "
              f"agreement is {100*overlap.loc[full_model, 'accuracy_given_agreement']:.1f}%, versus "
              f"{100*overlap.loc[full_model, 'accuracy_given_disagreement']:.1f}% given disagreement. "
              "These are descriptive conditional rates; the experiment never assumes that agreement always predicts correctness."]
    lines += ["", "## Fixed-target audit results", "",
              f"For each cohort and {config['split_seeds']} deterministic seeds, floor(25% of all prompt groups) "
              "form the fixed evaluation pool. Audit prefixes contain floor(20%, 40%, 60% of total cohort "
              "groups), nested within the remaining bank. Both judge directions and all six methods use "
              "the same whole-group splits. Exact repeated prompts across subsets stay together. Actual "
              "comparison-label costs can differ from group counts and are recorded per trial.", "",
              "Only audit reference labels fit corrections. Errors target realized held-out reference accuracy. "
              "The four corrected methods use fixed coefficient one, population tuning over [0,1], population "
              "tuning over [-1,1], or audit-residual tuning over [-1,1]. The last is a point-only grouped "
              "candidate, without a prediction interval. Existing population interval fields have their "
              "original target and assumptions; they are not scored for fixed-pool coverage.", "",
              "| Cohort | Target judge | Audit groups | Method | MAE (pp) | RMSE (pp) | Mean power | Mean reference labels |",
              "|---|---|---:|---|---:|---:|---:|---:|"]
    for cohort in COHORTS:
        for row in summary.loc[summary.cohort == cohort].itertuples():
            power = "--" if pd.isna(row.mean_selected_power) else f"{row.mean_selected_power:.3f}"
            lines.append(f"| {cohort} | {SHORT_JUDGES[row.target_judge]} | {row.labeled_fraction:.0%} | "
                         f"{NAMES[row.method]} | {100*row.mean_absolute_error:.3f} | "
                         f"{100*np.sqrt(row.mean_squared_error):.3f} | {power} | {row.mean_reference_labels_used:.1f} |")
    table = summary.pivot(index=["cohort", "target_judge", "labeled_fraction"], columns="method", values="mean_absolute_error")
    lines += ["", "### Primary paired direction of change", "",
              "Each comparison uses the same target, audit budget and splits. The following counts are "
              "descriptive cell comparisons, not significance tests or probabilities of deployment improvement.", ""]
    for method in ("ppi", "ppi_tuned", "ppi_signed", "audit_residual"):
        difference = table.loc["NonLLMBar", method]-table.loc["NonLLMBar", "human_only"]
        ties = np.isclose(difference, 0, rtol=0, atol=1e-12)
        lines.append(f"- {NAMES[method]} versus reference-audit-only: {int(((difference < 0) & ~ties).sum())} "
                     f"lower-MAE, {int(ties.sum())} tied, {int(((difference > 0) & ~ties).sum())} higher-MAE cells out of six.")
    low_full = table.loc[("NonLLMBar", full_model, .2)]
    low_small = table.loc[("NonLLMBar", small_model, .2)]
    mid_small = table.loc[("NonLLMBar", small_model, .4)]
    raw_difference = low_full.ppi_signed-low_full.raw_proxy
    raw_direction = "tied" if abs(raw_difference) <= 1e-12 else "lower" if raw_difference < 0 else "higher"
    residual_difference = mid_small.audit_residual-mid_small.human_only
    residual_direction = "tied" if abs(residual_difference) <= 1e-12 else "lower" if residual_difference < 0 else "higher"
    lines += ["", "At the primary 20% group budget, raw agreement has MAE "
              f"{100*low_full.raw_proxy:.3f} pp for GPT-4o, compared with {100*low_full.human_only:.3f} pp "
              f"for the reference audit and {100*low_full.ppi_signed:.3f} pp for signed population tuning. "
              f"For this configuration, signed population tuning has {raw_direction} MAE relative to raw agreement "
              "for that target and budget. For GPT-4o mini on the "
              f"same pools, raw MAE is {100*low_small.raw_proxy:.3f} pp, reference-audit MAE "
              f"{100*low_small.human_only:.3f} pp, and signed-population MAE {100*low_small.ppi_signed:.3f} pp. "
              "At the mini model's 40% budget, audit-residual tuning has "
              f"MAE {100*mid_small.audit_residual:.3f} pp versus {100*mid_small.human_only:.3f} pp "
              f"for reference audit ({residual_direction} MAE). All directions of change remain in the complete tables."]
    lines += ["", "Ties use absolute tolerance 1e-12. All primary and sensitivity cells, including losses, "
              "remain in the tables. `paired_method_differences.csv` retains same-split absolute/squared "
              "error contrasts against reference audit and signed population tuning. `paired_summary.csv` "
              "averages those contrasts descriptively; overlapping splits receive no independent-replication standard errors."]
    if config["plots"]:
        lines += ["", "![Audit errors across shared budgets](cross_judge_audit_budget.svg)", "",
                  "Panels use different vertical scales; compare methods within the same target and component."]
    lines += ["", "## Costs, provenance and limits", "",
              "`reference_labels_used` counts inference reference labels; a single revealed reference scores "
              "both judges. `shared_costs.csv` deduplicates inputs across both directions: raw agreement uses "
              "2N cached judgments and no audit reference, reference-audit-only uses n references and 2n "
              "judgments across the two targets, and corrected methods use n references and 2(n+N) judgments. "
              "Held-out scoring uses N additional references and 2N target judgments, whose overlap with "
              "inference reads is explicitly deduplicated. These are logical cached records, not fresh "
              "API calls, money, token counts or guaranteed annotation savings.", "",
              "- The 23 subsets mix curated preference, instruction following, safety conventions and "
              "code/math constructions. Their supplied references are the target, not infallible latent truth.",
              "- Both selected judges belong to one model family. A third inspected Gemini cache has two "
              "nonbinary fallback scores and is excluded as a whole under the binary reconstruction contract; "
              "no comparisons are removed from the selected complete panel.",
              "- Exact prompt grouping preserves known repeats, including cross-subset repeats, but does "
              "not prove independence of semantically related tasks. The unequal-group empirical study "
              "does not inherit IID prediction or finite-sample interval guarantees.",
              "- Cache pins reproduce historical 2024 results. Exact executed code, shuffle states, raw "
              "verdicts and complete invocation are absent. The inspected implementation documents scoring "
              "semantics, not a recovered execution trace or current-model performance.",
              "- Source notices distinguish benchmark, source-content, code and cached-result terms. The "
              "results card releases research results without a named license; the package MIT license "
              "does not relicense those source records. Only derived numeric data and hashes are included.", "",
              "## Reproduce", "", "```bash", 'pip install -e ".[dev,report]"',
              "python examples/rewardbench_audit.py --output reports/reproduced/rewardbench-audit --plot", "```", "",
              "This uses the certified committed numeric panel, with no network or model. To independently "
              "rebuild the panel from pinned public sources, install the `data` extra and run:", "", "```bash",
              "python examples/prepare_rewardbench.py --download --output reports/reproduced/rewardbench", "```", "",
              "The preparation verifies all source bytes, full-content joins, reference orientation and "
              "46 source subset aggregates. Stable content keys preserve the two distinct rows with raw "
              "ID 3692. All exact splits are retained in deterministic `split_manifest.json.gz`, using "
              "shared sorted identity catalogs and zero-based index lists. `decode_split_manifest` in this "
              "runner losslessly restores every original ID list and its order. "
              "`reproducibility.json` fingerprints source code, data, configuration and every report artifact.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"reports/rewardbench/labels.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"reports/rewardbench-audit")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    generated = {"trials.csv", "summary.csv", "corpus_diagnostics.csv", "paired_method_differences.csv",
                 "paired_summary.csv", "shared_costs.csv", "split_manifest.json.gz", "config.json", "REPORT.md"}
    if args.plot:
        generated |= {f"{name}.{ext}" for name in ("agreement_vs_accuracy", "cross_judge_audit_budget") for ext in ("png", "svg")}
    args.output.mkdir(parents=True, exist_ok=True)
    if {p.name for p in args.output.iterdir()} - generated - {"reproducibility.json"}:
        raise ValueError("output contains stale or unrelated artifacts")
    labels, _ = read_verified_labels(args.input)
    trials, summary = cross_judge_accuracy_audit(labels, seeds=range(args.seeds))
    manifests = trials.attrs.pop("split_manifest"); protocol = trials.attrs.pop("audit_protocol")
    cohort_metadata = trials.attrs.pop("cohort_metadata"); trials.attrs = {}
    diagnostics = corpus_diagnostics(labels)
    pairs = paired_differences(trials)
    contrast_columns = [name for name in pairs if "difference_vs_" in name]
    paired_summary = pairs.groupby(["cohort", "target_judge", "auxiliary_judge", "labeled_fraction", "method"], sort=True)[contrast_columns].mean().reset_index()
    costs = shared_costs(trials)
    config = {"study": "rewardbench-cross-judge-audit-v1", "retrospective": True,
              "primary_cohort": "NonLLMBar", "sensitivity_cohorts": ["All", "LLMBar"],
              "split_seeds": args.seeds, "plots": args.plot, "audit_protocol": protocol, "cohort_metadata": cohort_metadata,
              "trial_rows": len(trials), "summary_rows": len(summary), "split_manifests": len(manifests),
              "split_manifest_encoding": "indexed-content-identities-v1; zero-based catalog indexes preserve exact ID order",
              "paired_contrasts": "same target, cohort, seed and budget; descriptive means without iid split MCSE",
              "shared_costs": "union across both target directions; inference and held-out scoring separated"}
    for name, frame in (("trials", trials), ("summary", summary), ("corpus_diagnostics", diagnostics),
                        ("paired_method_differences", pairs), ("paired_summary", paired_summary), ("shared_costs", costs)):
        (args.output/f"{name}.csv").write_bytes(frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    stream = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
        compressed.write((json.dumps(encode_split_manifest(manifests), sort_keys=True, separators=(",", ":"))+"\n").encode("utf-8"))
    (args.output/"split_manifest.json.gz").write_bytes(stream.getvalue())
    write_json(args.output/"config.json", config)
    links = {"protocol": Path(os.path.relpath(ROOT/"docs/REWARDBENCH_PROTOCOL.md", args.output)).as_posix(),
             "notice": Path(os.path.relpath(args.input.with_name("NOTICE.md"), args.output)).as_posix()}
    (args.output/"REPORT.md").write_bytes(make_report(summary, diagnostics, config, links).encode("utf-8"))
    if args.plot:
        make_plots(summary, diagnostics, args.output)
    sources = sorted((ROOT/"src/judgecal").glob("*.py")) + [Path(__file__), ROOT/"examples/prepare_rewardbench.py",
                ROOT/"examples/research_study.py", ROOT/"docs/REWARDBENCH_PROTOCOL.md", ROOT/"pyproject.toml"]
    reproducibility = {"python": platform.python_version(), "platform": platform.system(),
                       "packages": {name: version(name) for name in ("judgecal", "numpy", "scipy", "pandas")},
                       "input_labels_sha256": sha256(args.input), "input_provenance_sha256": sha256(args.input.with_name("dataset.json")),
                       "input_notice_sha256": sha256(args.input.with_name("NOTICE.md")),
                       "source_text_normalization": "CRLF to LF before SHA-256",
                       "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sources},
                       "artifacts_sha256": {name: sha256(args.output/name) for name in sorted(generated)}}
    if args.plot:
        reproducibility["packages"]["matplotlib"] = version("matplotlib")
    write_json(args.output/"reproducibility.json", reproducibility)
    print(f"Wrote {len(trials)} trials, {len(summary)} summaries and {len(manifests)} exact shared splits.")


if __name__ == "__main__":
    main()
