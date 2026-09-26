"""Sampling identity, outcome information boundary, costs and paired summaries."""
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from judgecal.finite_corpus_study import (
    METHODS, catalog_sha256, decode_sample, draw_permutation, evaluate_audit,
    finite_corpus_study, paired_differences, prepare_frame, summarize_trials,
)


@pytest.fixture(scope="module")
def labels():
    data = pd.read_csv(Path(__file__).resolve().parents[1]/"reports/rewardbench/labels.csv")
    panel = data.loc[(~data.subset.str.startswith("llmbar-")) & (data.judge == data.judge.iloc[0])]
    # Unequal sizes and an exceptional size-five group make costs nontrivial.
    sizes = panel.groupby("instruction_id").size()
    ids = sizes.sort_values(ascending=False).index[:7].tolist()+sizes[sizes == 1].index[:8].tolist()
    return data.loc[data.instruction_id.isin(ids)].copy()


def test_frame_and_stream_identity_survive_row_reordering_and_longer_run(labels):
    a = finite_corpus_study(labels, repetitions=2, seed=19)
    b = finite_corpus_study(labels.sample(frac=1, random_state=12), repetitions=3, seed=19)
    assert a["catalog"] == b["catalog"]
    assert a["permutations"]["draws"] == b["permutations"]["draws"][:2]
    pd.testing.assert_frame_equal(a["trials"], b["trials"].loc[b["trials"].repetition < 2].reset_index(drop=True))
    assert a["configuration"]["frame_sha256"] == catalog_sha256(a["catalog"])
    assert a["census"].exact.all() and len(a["census"]) == 8
    assert len(a["trials"]) == 48 and len(a["summary"]) == 24 and len(a["paired"]) == 36


def test_manifest_recovers_nested_membership_shared_cost_and_hashes(labels):
    result = finite_corpus_study(labels, repetitions=2, seed=17)
    for draw in result["permutations"]["draws"]:
        previous = set()
        splits = result["splits"].loc[result["splits"].repetition == draw["repetition"]]
        for row in splits.itertuples():
            groups, rows = decode_sample(result["catalog"], draw["permutation"], row.n_audited_groups)
            assert previous <= set(groups)
            previous = set(groups)
            assert row.audited_rows == len(rows) == len(set(rows))
            assert row.selected_group_sha256 == hashlib.sha256(("\n".join(groups)+"\n").encode()).hexdigest()
            assert row.selected_row_sha256 == hashlib.sha256(("\n".join(rows)+"\n").encode()).hexdigest()
            fits = result["trials"].loc[(result["trials"].repetition == row.repetition)
                                        & (result["trials"].group_fraction == row.group_fraction)]
            assert len(fits) == 8
            assert fits.audited_rows.eq(len(rows)).all()
            assert fits.selected_group_sha256.eq(row.selected_group_sha256).all()
    with pytest.raises(ValueError, match="exactly once"):
        decode_sample(result["catalog"], [0]*len(result["catalog"]["groups"]), 1)


def test_unsampled_reference_mutation_cannot_change_fit_or_frame(labels):
    catalog, sizes, proxy, gold = prepare_frame(labels)
    selected = draw_permutation(len(sizes), 31, 0)[:3]
    judge = next(iter(gold))
    before = evaluate_audit(sizes, proxy, selected, gold[judge][selected])
    sampled_ids = {catalog["groups"][i]["instruction_id"] for i in selected}
    mutated = labels.copy()
    mask = ~mutated.instruction_id.isin(sampled_ids)
    mutated.loc[mask, "reference_choice"] = 3-mutated.loc[mask, "reference_choice"]
    mutated["reference_correct"] = mutated.canonical_choice.eq(mutated.reference_choice).astype(int)
    other_catalog, other_sizes, other_proxy, other_gold = prepare_frame(mutated)
    assert catalog == other_catalog
    assert np.array_equal(proxy, other_proxy)
    assert not np.array_equal(gold[judge], other_gold[judge])
    assert before == evaluate_audit(other_sizes, other_proxy, selected, other_gold[judge][selected])
    changed = gold[judge][selected].copy()
    changed[0] = 0 if changed[0] > 0 else sizes[selected[0]]
    assert before != evaluate_audit(sizes, proxy, selected, changed)


def test_size_control_needs_no_agreement_signal_and_cost_expectation(labels):
    _, sizes, proxy, gold = prepare_frame(labels)
    selected = np.arange(3)
    y = next(iter(gold.values()))[selected]
    a = {row["method"]: row for row in evaluate_audit(sizes, proxy, selected, y)}
    b = {row["method"]: row for row in evaluate_audit(sizes, sizes, selected, y)}
    assert a["size_grid_ebs"] == b["size_grid_ebs"]
    for key in ("point", "selected_power", "candidate_points_json", "candidate_radii_json", "width"):
        assert b["agreement_grid_ebs"][key] == b["size_grid_ebs"][key]
    expected = 3*sizes.sum()/len(sizes)
    enumerated = np.mean([sizes[list(indices)].sum() for indices in combinations(range(len(sizes)), 3)])
    assert enumerated == pytest.approx(expected)
    assert a["ht_hs"]["expected_audited_rows"] == pytest.approx(expected)


def test_paired_mse_uncertainty_uses_within_draw_differences(labels):
    result = finite_corpus_study(labels, repetitions=4, seed=47)
    trials = result["trials"].copy()
    judge = trials.target_judge.iloc[0]
    fraction = trials.group_fraction.iloc[0]
    mask = (trials.target_judge == judge) & (trials.group_fraction == fraction)
    reference = [0.01, 0.1, 0.05, 0.07]
    left = [0.02, 0.08, 0.055, 0.06]
    trials.loc[mask & trials.method.eq("size_grid_ebs"), "squared_error"] = reference
    trials.loc[mask & trials.method.eq("agreement_grid_ebs"), "squared_error"] = left
    pairs = paired_differences(trials)
    row = pairs.loc[(pairs.target_judge == judge) & (pairs.group_fraction == fraction)
                    & pairs.method.eq("agreement_grid_ebs") & pairs.baseline.eq("size_grid_ebs")].iloc[0]
    delta = np.array(left)-reference
    assert row.mse_difference == pytest.approx(delta.mean())
    assert row.mse_mcse == pytest.approx(delta.std(ddof=1)/2)
    summary = summarize_trials(trials)
    row = summary.loc[(summary.target_judge == judge) & (summary.group_fraction == fraction)
                      & summary.method.eq("agreement_grid_ebs")].iloc[0]
    assert row.mse == pytest.approx(np.mean(left))
    assert row.mse_mcse == pytest.approx(np.std(left, ddof=1)/2)
    assert len(pairs) == 6*len(METHODS)*(len(METHODS)-1)//2


@pytest.mark.parametrize("name,value", [("repetitions", 1), ("repetitions", True), ("seed", -1), ("seed", 1.1), ("alpha", 0)])
def test_invalid_study_options_fail_before_drawing(labels, name, value):
    with pytest.raises(ValueError):
        finite_corpus_study(labels, **{name: value})
