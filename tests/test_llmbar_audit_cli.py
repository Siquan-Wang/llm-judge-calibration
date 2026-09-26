"""Offline corpus integrity, retained evidence, and reproducible judge audits."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cli(monkeypatch):
    path = Path(__file__).resolve().parents[1]/"examples/llmbar_audit.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("llmbar_audit_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostics_retain_invalids_and_consistent_wrong_choices(cli):
    frame = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"], "instruction_id": ["a", "b", "c", "d"],
        "subset": ["Natural"]*4, "judge": ["judge"]*4,
        "reference_label": [1, 1, 2, 2], "forward_prediction": [1, 2, 0, 2],
        "reverse_prediction": [1, 2, 0, 0],
    })
    row = cli.corpus_diagnostics(frame).set_index("cohort").loc["All"]
    assert row.comparisons == 4
    assert row.forward_accuracy == .5 and row.reverse_accuracy == .25
    assert row.order_agreement == .5  # Two invalids are not valid agreement.
    assert row.agree_but_wrong == row.agree_and_forward_correct == 1
    assert row.accuracy_given_agreement == .5
    assert row.forward_invalid == 1 and row.reverse_invalid == 2


def test_committed_snapshot_and_known_source_counts(cli):
    frame, source = cli.read_verified_labels(cli.ROOT/"reports/llmbar/labels.csv")
    assert len(frame) == 419*3
    assert frame.sample_id.nunique() == 419
    assert frame.instruction_id.nunique() == 418
    assert set(frame.judge) == set(cli.JUDGES)
    assert source["source_revision"] == cli.REVISION
    diagnostic = cli.corpus_diagnostics(frame).set_index(["cohort", "judge"])
    # Independently counted from the five pinned cache files for each judge.
    correct = {"Natural": (95, 80, 79), "Neighbor": (104, 30, 34),
               "GPTInst": (78, 25, 28), "GPTOut": (35, 17, 27), "Manual": (35, 18, 17)}
    for subset, values in correct.items():
        for judge, count in zip(cli.JUDGES, values):
            row = diagnostic.loc[(subset, judge)]
            assert row.forward_accuracy*row.comparisons == pytest.approx(count)
    all_rows = diagnostic.loc["All"]
    assert all_rows.forward_invalid.sum() == 1 and all_rows.reverse_invalid.sum() == 1
    assert all_rows.loc["LLaMA2", "forward_invalid"] == 1
    assert all_rows.loc["LLaMA2", "reverse_invalid"] == 1


def test_lossless_common_splits_costs_and_deterministic_export(cli, tmp_path):
    args = ["--output", str(tmp_path), "--seeds", "2"]
    cli.main(args)
    trials = pd.read_csv(tmp_path/"trials.csv")
    summary = pd.read_csv(tmp_path/"summary.csv")
    assert len(trials) == 3*3*3*2*4 and len(summary) == 3*3*3*4
    assert summary.trials.eq(2).all()
    assert not any("coverage" in name or "mcse" in name for name in summary)
    observed = trials.groupby(["cohort", "judge", "labeled_fraction", "method"]).agg(
        mae=("absolute_error", "mean"), mse=("squared_error", "mean"))
    saved = summary.set_index(["cohort", "judge", "labeled_fraction", "method"]).sort_index()
    np.testing.assert_allclose(observed.mae, saved.mean_absolute_error, atol=1e-15)
    np.testing.assert_allclose(observed.mse, saved.mean_squared_error, atol=1e-15)
    paired = pd.read_csv(tmp_path/"paired_method_differences.csv")
    assert paired.loc[paired.method == "human_only", "absolute_error_difference_vs_human"].eq(0).all()
    keys = ["cohort", "judge", "labeled_fraction", "method"]
    differences = paired.groupby(keys).absolute_error_difference_vs_human.mean()
    human = saved.xs("human_only", level="method").mean_absolute_error
    for index, difference in differences.items():
        assert difference == pytest.approx(saved.loc[index, "mean_absolute_error"]-human.loc[index[:-1]], abs=1e-15)
    splits = json.loads((tmp_path/"split_manifest.json").read_text())
    assert len(splits) == 3*3*2
    for split in splits:
        rows = trials.loc[(trials.cohort == split["cohort"]) & (trials.seed == split["seed"])
                          & (trials.labeled_fraction == split["labeled_fraction"])]
        assert len(rows) == 3*4 and rows.split_sha256.nunique() == 1
        assert rows.loc[rows.method != "raw_proxy", "human_labels_used"].eq(split["n_labeled"]).all()
        assert rows.loc[rows.method == "raw_proxy", "human_labels_used"].eq(0).all()
        assert rows.heldout_reference_labels_scored.eq(split["n_evaluation"]).all()
    manifest = json.loads((tmp_path/"reproducibility.json").read_text())
    before = {name: (tmp_path/name).read_bytes() for name in manifest["artifacts_sha256"]}
    for name, digest in manifest["artifacts_sha256"].items():
        assert hashlib.sha256(before[name]).hexdigest() == digest
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((cli.ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest
    cli.main(args)
    assert before == {name: (tmp_path/name).read_bytes() for name in before}
    assert manifest == json.loads((tmp_path/"reproducibility.json").read_text())


def test_tampered_input_cannot_enter_an_audit(cli, tmp_path):
    source = cli.ROOT/"reports/llmbar"
    (tmp_path/"dataset.json").write_bytes((source/"dataset.json").read_bytes())
    (tmp_path/"labels.csv").write_bytes((source/"labels.csv").read_bytes()+b"\n")
    with pytest.raises(ValueError, match="checksum"):
        cli.read_verified_labels(tmp_path/"labels.csv")


def test_self_declared_hash_cannot_certify_a_replacement_corpus(cli, tmp_path):
    original = cli.ROOT/"reports/llmbar"
    frame = pd.read_csv(original/"labels.csv")
    frame.loc[0, "forward_prediction"] = 3-frame.loc[0, "forward_prediction"]
    frame.to_csv(tmp_path/"labels.csv", index=False, lineterminator="\n")
    manifest = json.loads((original/"dataset.json").read_text())
    manifest["labels_sha256"] = hashlib.sha256((tmp_path/"labels.csv").read_bytes()).hexdigest()
    (tmp_path/"dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned complete LLMBar label snapshot"):
        cli.read_verified_labels(tmp_path/"labels.csv")


def test_source_metadata_cannot_silently_change(cli, tmp_path):
    original = cli.ROOT/"reports/llmbar"
    (tmp_path/"labels.csv").write_bytes((original/"labels.csv").read_bytes())
    manifest = json.loads((original/"dataset.json").read_text())
    manifest["comparisons"] = 1
    (tmp_path/"dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source manifest"):
        cli.read_verified_labels(tmp_path/"labels.csv")


def test_input_snapshot_and_stale_artifacts_are_preserved(cli, tmp_path):
    input_path = cli.ROOT/"reports/llmbar/labels.csv"
    with pytest.raises(ValueError, match="separate output"):
        cli.main(["--output", str(input_path.parent), "--seeds", "1"])
    (tmp_path/"accuracy_vs_consistency.svg").write_bytes(b"older figure")
    with pytest.raises(ValueError, match="stale or unrelated"):
        cli.main(["--output", str(tmp_path), "--seeds", "1"])
    assert (tmp_path/"accuracy_vs_consistency.svg").read_bytes() == b"older figure"
