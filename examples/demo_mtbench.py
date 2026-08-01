"""End-to-end demo on MT-Bench human judgments (Zheng et al., NeurIPS 2023).

Downloads `lmsys/mt_bench_human_judgments` from the Hugging Face Hub, aligns
GPT-4 pairwise votes with human majority votes, then answers three questions:

1. How well does GPT-4-as-judge agree with humans, with honest uncertainty?
2. What is the judge's error structure (sensitivity/specificity), and how
   much does a naive judge-based win-rate move after correcting for it?
3. How does judge-human agreement compare with human-human agreement
   (the practical ceiling for any judge)?

Run:  pip install -e ".[data]"  &&  python examples/demo_mtbench.py
"""
from __future__ import annotations

import itertools

import pandas as pd

from judgecal import (
    agreement_with_ci,
    cohens_kappa,
    compare_models,
    judge_confusion,
    paired_agreement_gap,  # noqa: F401  (exposed for further exploration)
)
from judgecal.data import load_mt_bench, load_mt_bench_annotator_votes


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    df = load_mt_bench()
    print(f"Aligned comparisons (GPT-4 vote + human majority vote): {len(df)}")

    section("1. GPT-4-as-judge vs. human majority: agreement with uncertainty")
    ci = agreement_with_ci(df["gpt4_winner"], df["human_winner"])
    kappa = cohens_kappa(df["gpt4_winner"], df["human_winner"])
    print(f"Agreement (ties excluded): {ci.point:.3f}  "
          f"95% CrI [{ci.low:.3f}, {ci.high:.3f}]  (Beta-Binomial)")
    print(f"Cohen's kappa (all labels): {kappa:.3f}")

    section("2. Judge error structure and measurement-error-corrected win-rates")
    conf = judge_confusion(df["gpt4_winner"], df["human_winner"])
    print(f"Sensitivity P(judge=A | human=A): {conf.sensitivity:.3f}")
    print(f"Specificity P(judge=B | human=B): {conf.specificity:.3f}")
    print(f"Youden's J (judge informativeness): {conf.youden_j:.3f}")

    print("\nPer model pair: raw GPT-4 win-rate vs. Rogan-Gladen corrected")
    rows = []
    for (ma, mb), g in df.groupby(["model_a", "model_b"]):
        if len(g) < 30:
            continue
        res = compare_models(g["gpt4_winner"], confusion=conf, random_state=0)
        rows.append({
            "pair": f"{ma} vs {mb}",
            "n": res["n"],
            "raw_win_rate": round(res["win_rate"], 3),
            "corrected": round(res["corrected_win_rate"], 3),
            "ci_95": f"[{res['ci']['low']:.3f}, {res['ci']['high']:.3f}]",
            "significant": res["significant"],
        })
    print(pd.DataFrame(rows).to_string(index=False))

    section("3. Human-human agreement: the ceiling for any judge")
    votes = load_mt_bench_annotator_votes()
    keys = ["question_id", "turn", "model_a", "model_b"]
    agree_a, agree_b = [], []
    for _, g in votes.groupby(keys):
        if g["judge"].nunique() < 2:
            continue
        per_annotator = g.groupby("judge")["label"].first()
        for x, y in itertools.combinations(per_annotator.tolist(), 2):
            agree_a.append(x)
            agree_b.append(y)
    hh = agreement_with_ci(agree_a, agree_b)
    print(f"Human-human agreement (ties excluded): {hh.point:.3f}  "
          f"95% CrI [{hh.low:.3f}, {hh.high:.3f}]  "
          f"on {len(agree_a)} annotator pairs")
    print(f"Judge-human agreement from step 1:     {ci.point:.3f}")
    print("\nInterpretation: if the judge-human interval overlaps the "
          "human-human interval,\nthe judge is statistically at the "
          "annotator-noise ceiling for this task.")


if __name__ == "__main__":
    main()
