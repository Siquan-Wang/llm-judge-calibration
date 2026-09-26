"""Complete target-specific evidence, aligned contrasts and preserved history."""
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
    path = Path(__file__).resolve().parents[1]/"examples/estimand_study.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("estimand_study_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_targets_common_history_and_deterministic_artifacts(cli, tmp_path):
    args = ["--output", str(tmp_path), "--repetitions", "3", "--seeds", "2"]
    cli.main(args)
    compressed = (tmp_path/"simulation_trials.csv.gz").read_bytes()
    assert compressed[4:8] == b"\x00\x00\x00\x00"
    trials = pd.read_csv(io.BytesIO(gzip.decompress(compressed)))
    summary = pd.read_csv(tmp_path/"simulation_summary.csv").set_index(["scenario", "method"])
    assert len(trials) == 36*4*3 and len(summary) == 36*4
    for key, rows in trials.groupby(["scenario", "method"]):
        for target in ("population", "pool"):
            assert summary.loc[key, f"{target}_rmse"] == pytest.approx(np.sqrt(np.mean(rows[f"{target}_error"]**2)))
            assert summary.loc[key, f"{target}_coverage"] == pytest.approx(rows[f"{target}_covered"].mean())
            assert summary.loc[key, f"{target}_coverage_mc_low"] <= summary.loc[key, f"{target}_coverage"]
            assert summary.loc[key, f"{target}_coverage_mc_high"] >= summary.loc[key, f"{target}_coverage"]
    pairs = pd.read_csv(tmp_path/"simulation_paired_differences.csv").set_index("scenario")
    for scenario, rows in trials.groupby("scenario"):
        for target in ("population", "pool"):
            errors = rows.pivot(index="repetition", columns="method", values=f"{target}_error")
            differences = errors.pool_tuned**2-errors.population_tuned**2
            assert pairs.loc[scenario, f"{target}_mse_difference"] == pytest.approx(differences.mean())
            assert pairs.loc[scenario, f"{target}_mse_difference_mcse"] == pytest.approx(differences.std(ddof=1)/np.sqrt(3))
    empirical = pd.read_csv(tmp_path/"llmbar_trials.csv")
    assert len(empirical) == 3*3*3*2*6
    assert len(pd.read_csv(tmp_path/"llmbar_summary.csv")) == 3*3*3*6
    new = empirical.loc[empirical.method == "audit_residual"]
    assert new[["interval_low", "interval_high", "interval_width", "standard_error", "estimated_variance_ratio"]].isna().all().all()
    indexed = empirical.set_index(cli.KEYS+["method"])
    for row in pd.read_csv(tmp_path/"llmbar_paired_differences.csv").itertuples(index=False):
        key = (row.cohort, row.judge, row.seed, row.labeled_fraction, row.method)
        for baseline, method in (("human", "human_only"), ("population", "ppi_signed")):
            for metric in ("absolute_error", "squared_error"):
                assert getattr(row, f"{metric}_difference_vs_{baseline}") == pytest.approx(
                    indexed.loc[key, metric]-indexed.loc[key[:-1]+(method,), metric], abs=1e-15)
    config = json.loads((tmp_path/"config.json").read_text())
    assert config["historical_alignment"]["historical_splits_verified"] == 18
    assert config["historical_alignment"]["historical_method_trials_verified"] == 270
    manifest = json.loads((tmp_path/"reproducibility.json").read_text())
    before = {name: (tmp_path/name).read_bytes() for name in manifest["artifacts_sha256"]}
    for name, digest in manifest["artifacts_sha256"].items():
        assert hashlib.sha256(before[name]).hexdigest() == digest
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((cli.ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest
    cli.main(args)
    assert before == {name: (tmp_path/name).read_bytes() for name in before}
    assert manifest == json.loads((tmp_path/"reproducibility.json").read_text())


def test_preserved_point_fields_are_verified_not_only_splits(cli):
    from judgecal.judge_audit import judge_accuracy_audit
    labels, _ = cli.read_verified_labels(cli.ROOT/"reports/llmbar/labels.csv")
    trials, _ = judge_accuracy_audit(labels, seeds=[0], include_signed=True, include_pool_tuned=True)
    manifests = trials.attrs["split_manifest"]; trials.attrs = {}
    assert cli.historical_alignment(trials, manifests, 1)["historical_method_trials_verified"] == 135
    trials.loc[trials.method == "ppi_signed", "estimate"] += .1
    with pytest.raises(AssertionError):
        cli.historical_alignment(trials, manifests, 1)


def test_stale_plot_is_not_adopted(cli, tmp_path):
    (tmp_path/"estimand_tradeoff.svg").write_bytes(b"old plot")
    with pytest.raises(ValueError, match="stale or unrelated"):
        cli.main(["--output", str(tmp_path), "--repetitions", "2", "--seeds", "1"])
    assert (tmp_path/"estimand_tradeoff.svg").read_bytes() == b"old plot"
