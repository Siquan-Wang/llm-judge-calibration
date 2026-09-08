"""Public-data demo with question-disjoint human-audit and evaluation sets.

Run: pip install -e ".[data]" && python examples/demo_mtbench.py
No model API calls are made. A fixed seed assigns 40 of the 80 questions to
the labeled audit before examining outcomes. All turns and model pairs of a
question stay together. Human labels in the remaining questions are hidden
from the estimator and revealed only for a retrospective descriptive check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from judgecal.data import load_mt_bench
from judgecal.ppi import prediction_powered_win_rate


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def split_by_question(df: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed 50/50 split by prompt, shared across every model-pair analysis."""
    questions = np.sort(df["question_id"].unique())
    if len(questions) < 4:
        raise ValueError("The demo requires at least four distinct questions")
    questions = np.random.default_rng(seed).permutation(questions)
    audit_ids = questions[:len(questions) // 2]
    audit = df.loc[df["question_id"].isin(audit_ids)].copy()
    evaluation = df.loc[~df["question_id"].isin(audit_ids)].copy()
    return audit, evaluation


def _human_rate(labels: pd.Series) -> float:
    return float(labels.map({"A": 1.0, "B": 0.0, "tie": 0.5}).mean())


def main() -> None:
    df = load_mt_bench()
    print(f"Dataset: {df.attrs['dataset_id']} @ {df.attrs['dataset_revision']}")
    print(f"Attribution: {df.attrs['citation']} ({df.attrs['license']})")
    print(f"Aligned comparisons: {len(df)}; prompt groups: {df.question_id.nunique()}")
    print(f"Source counts: {df.attrs['source_counts']}")
    print("Human reference: plurality of distinct annotators; tied vote counts become tie.")
    audit, evaluation = split_by_question(df, seed=0)
    print(f"Audit: {len(audit)} rows / {audit.question_id.nunique()} questions; "
          f"evaluation: {len(evaluation)} rows / {evaluation.question_id.nunique()} questions.")

    section("1. Estimate each model pair's win rate using only audit human labels")
    print("A is the first named model. All ties score 0.5 in every rate.")
    print("PPI = evaluation judge mean + audit mean(human - judge).")
    rows = []
    for (ma, mb), pool in evaluation.groupby(["model_a", "model_b"], sort=True):
        labeled = audit.loc[(audit.model_a == ma) & (audit.model_b == mb)]
        if labeled.question_id.nunique() < 2 or pool.question_id.nunique() < 2:
            raise ValueError(f"Insufficient independent questions for {ma} vs {mb}")
        result = prediction_powered_win_rate(
            labeled.gpt4_winner, labeled.human_winner, pool.gpt4_winner,
            target="A", labeled_groups=labeled.question_id,
            unlabeled_groups=pool.question_id,
        )
        # Evaluation human labels are used only AFTER producing the estimate.
        held_out_reference = _human_rate(pool.human_winner)
        rows.append({
            "A vs B": f"{ma} vs {mb}",
            "audit_q": result.n_labeled_groups,
            "eval_q": result.n_unlabeled_groups,
            "raw": round(result.raw_rate, 3),
            "audit_human": round(result.human_only_rate, 3),
            "PPI": round(result.point, 3),
            "PPI_95%": f"[{result.interval.low:.3f}, {result.interval.high:.3f}]",
            "heldout_human": round(held_out_reference, 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("Intervals are asymptotic with question-cluster variance, not finite-sample guarantees.")
    print("They target a population mean under sampling/transport assumptions, not coverage "
          "of this fixed held-out human reference. Estimates and intervals are not clipped.")

    section("2. Reveal evaluation labels for a descriptive agreement check")
    all_agreement = float((evaluation.gpt4_winner == evaluation.human_winner).mean())
    decisive = evaluation.loc[(evaluation.gpt4_winner != "tie") & (evaluation.human_winner != "tie")]
    decisive_agreement = float((decisive.gpt4_winner == decisive.human_winner).mean())
    inconsistent = int(evaluation.gpt4_inconsistent_order.sum())
    print(f"Agreement including ties: {all_agreement:.3f} on {len(evaluation)} comparisons.")
    print(f"Agreement conditional on neither side tying: {decisive_agreement:.3f} "
          f"on {len(decisive)} comparisons.")
    print(f"GPT-4 inconsistent-order ties: {inconsistent}/{len(evaluation)}; "
          "retained as ties and separately flagged.")
    print("The source does not include the two separate order-specific votes. "
          "These diagnostics cannot identify a causal position-bias effect.")
    print("This fixed demo split is illustrative. It does not establish human-level "
          "equivalence, a human noise ceiling, or improvement on every model pair.")


if __name__ == "__main__":
    main()
