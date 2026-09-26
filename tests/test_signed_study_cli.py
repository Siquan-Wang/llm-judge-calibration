"""Retained signed-study evidence and historical common-split invariants."""
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cli(monkeypatch):
    path = Path(__file__).resolve().parents[1]/"examples/signed_power_study.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("signed_power_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_reproducible_evidence_and_paired_contrasts(cli, tmp_path):
    args = ["--output", str(tmp_path), "--repetitions", "3", "--seeds", "2"]
    cli.main(args)
    compressed = (tmp_path/"simulation_trials.csv.gz").read_bytes()
    assert compressed[4:8] == b"\x00\x00\x00\x00"
    simulation = pd.read_csv(io.BytesIO(gzip.decompress(compressed)))
    summary = pd.read_csv(tmp_path/"simulation_summary.csv")
    assert len(simulation) == 18*4*3 and len(summary) == 18*4
    assert set(simulation.method) == {"human_only", "ppi", "ppi_tuned", "ppi_signed"}
    saved = summary.set_index(["scenario", "method"]).sort_index()
    observed = simulation.groupby(["scenario", "method"]).agg(
        bias=("error", "mean"), coverage=("covered", "mean"), width=("interval_width", "mean"))
    for actual, expected in ((observed.bias, saved.bias), (observed.coverage, saved.coverage),
                             (observed.width, saved.mean_width)):
        np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert (summary.coverage_mc_low <= summary.coverage).all()
    assert (summary.coverage_mc_high >= summary.coverage).all()
    for (scenario, method), rows in simulation.groupby(["scenario", "method"]):
        for baseline in ("human", "positive"):
            values = rows[f"squared_error_difference_vs_{baseline}"].to_numpy()
            assert saved.loc[(scenario, method), f"mse_difference_vs_{baseline}"] == pytest.approx(values.mean())
            assert saved.loc[(scenario, method), f"mse_difference_vs_{baseline}_mcse"] == pytest.approx(values.std(ddof=1)/np.sqrt(3))
    empirical = pd.read_csv(tmp_path/"llmbar_trials.csv")
    assert len(empirical) == 3*3*3*2*5
    assert len(pd.read_csv(tmp_path/"llmbar_summary.csv")) == 3*3*3*5
    paired = pd.read_csv(tmp_path/"llmbar_paired_differences.csv")
    keys = ["cohort", "judge", "seed", "labeled_fraction", "method"]
    indexed = empirical.set_index(keys)
    for row in paired.itertuples(index=False):
        key = (row.cohort, row.judge, row.seed, row.labeled_fraction, row.method)
        for baseline, method in (("human", "human_only"), ("positive", "ppi_tuned")):
            comparison = indexed.loc[key[:-1]+(method,)]
            for metric in ("absolute_error", "squared_error"):
                assert getattr(row, f"{metric}_difference_vs_{baseline}") == pytest.approx(indexed.loc[key, metric]-comparison[metric], abs=1e-15)
    config = json.loads((tmp_path/"config.json").read_text())
    assert config["historical_alignment"]["historical_splits_verified"] == 18
    assert config["historical_alignment"]["historical_method_trials_verified"] == 216
    assert config["repetitions"] == 3 and config["split_seeds"] == 2
    manifest = json.loads((tmp_path/"reproducibility.json").read_text())
    before = {name: (tmp_path/name).read_bytes() for name in manifest["artifacts_sha256"]}
    for name, digest in manifest["artifacts_sha256"].items():
        assert hashlib.sha256(before[name]).hexdigest() == digest
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((cli.ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest
    cli.main(args)
    assert before == {name: (tmp_path/name).read_bytes() for name in before}
    assert manifest == json.loads((tmp_path/"reproducibility.json").read_text())


def test_historical_split_or_target_cannot_silently_change(cli):
    from judgecal.judge_audit import judge_accuracy_audit
    labels, _ = cli.read_verified_labels(cli.ROOT/"reports/llmbar/labels.csv")
    trials, _ = judge_accuracy_audit(labels, seeds=[0], include_signed=True)
    manifests = trials.attrs["split_manifest"]
    trials.attrs = {}
    assert cli.verify_historical_alignment(trials, manifests, 1)["historical_splits_verified"] == 9
    changed = trials.copy()
    changed.loc[changed.method == "human_only", "heldout_accuracy_reference"] += .1
    with pytest.raises(AssertionError):
        cli.verify_historical_alignment(changed, manifests, 1)
    manifests[0]["split_sha256"] = "changed"
    with pytest.raises(ValueError, match="historical instruction split"):
        cli.verify_historical_alignment(trials, manifests, 1)


def test_stale_figure_is_preserved(cli, tmp_path):
    (tmp_path/"signed_simulation_tradeoff.svg").write_bytes(b"older figure")
    with pytest.raises(ValueError, match="stale or unrelated"):
        cli.main(["--output", str(tmp_path), "--repetitions", "2", "--seeds", "1"])
    assert (tmp_path/"signed_simulation_tradeoff.svg").read_bytes() == b"older figure"
