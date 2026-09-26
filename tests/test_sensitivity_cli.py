"""Sensitivity evidence must match the published plurality experiment exactly."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cli(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/reference_sensitivity.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("sensitivity_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outputs_preserve_published_plurality_and_paired_costs(cli, tmp_path):
    args = ["--output", str(tmp_path), "--seeds", "1"]
    cli.main(args)
    trials = pd.read_csv(tmp_path/"trials.csv")
    original = pd.read_csv(cli.ROOT/"reports/research/trials.csv")
    original = original.loc[original.seed == 0].reset_index(drop=True)
    plurality = trials.loc[trials.reference_definition == "plurality", original.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(plurality, original, check_exact=False, rtol=1e-12, atol=1e-14)
    assert len(trials) == 15*3*4*2
    paired = pd.read_csv(tmp_path/"paired_reference_differences.csv")
    assert len(paired)*2 == len(trials)
    np.testing.assert_allclose(paired.estimate_change, paired.estimate_mean_vote-paired.estimate_plurality, atol=1e-15)
    assert (paired.loc[paired.method == "raw_judge", "estimate_change"] == 0).all()
    assert (paired.loc[paired.method == "raw_judge", "human_votes_used"] == 0).all()
    manifest = json.loads((tmp_path/"reproducibility.json").read_text())
    before = {name: (tmp_path/name).read_bytes() for name in manifest["artifacts_sha256"]}
    for name, checksum in manifest["artifacts_sha256"].items():
        assert hashlib.sha256(before[name]).hexdigest() == checksum
        assert b"\r\n" not in before[name]
    for name, checksum in manifest["source_sha256"].items():
        assert hashlib.sha256((cli.ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == checksum
    config = json.loads((tmp_path/"config.json").read_text())
    assert config["split_seeds"] == 1
    diagnostics = json.loads((tmp_path/"reference_diagnostics.json").read_text())
    assert diagnostics["comparisons"] == 1814
    assert diagnostics["human_votes"] == 3354
    assert diagnostics["single_vote_comparisons"] == 854
    assert diagnostics["changed_scores"] == 305
    changes = pd.read_csv(tmp_path/"reference_changes.csv")
    assert len(changes) == 1814
    np.testing.assert_allclose(changes.mean_vote_score,
                               (changes.n_human_a + .5*changes.n_human_ties)/changes.n_human_votes)
    assert changes.score_change.abs().mean() == pytest.approx(diagnostics["mean_absolute_score_change"])
    cli.main(args)
    assert before == {name: (tmp_path/name).read_bytes() for name in before}
    assert json.loads((tmp_path/"reproducibility.json").read_text()) == manifest


def test_input_and_stale_outputs_are_preserved(cli, tmp_path):
    with pytest.raises(ValueError, match="preserve the input study"):
        cli.main(["--output", str(cli.ROOT/"reports/mtbench"), "--seeds", "1"])
    (tmp_path/"reference_sensitivity.svg").write_bytes(b"older plot")
    with pytest.raises(ValueError, match="stale or unrelated"):
        cli.main(["--output", str(tmp_path), "--seeds", "1"])
    assert (tmp_path/"reference_sensitivity.svg").read_bytes() == b"older plot"


@pytest.mark.parametrize("column,value", [("human_votes_used", 99), ("split_sha256", "different")])
def test_unpaired_evidence_cannot_be_summarized(cli, column, value):
    frame, _ = cli.read_verified_labels(cli.ROOT/"reports/mtbench/labels.csv")
    trials, _ = cli.reference_definition_sensitivity(frame, seeds=[0], fractions=[.2])
    index = trials.index[trials.reference_definition == "mean_vote"][0]
    trials.loc[index, column] = value
    with pytest.raises(ValueError, match=column):
        cli.paired_reference_differences(trials)
