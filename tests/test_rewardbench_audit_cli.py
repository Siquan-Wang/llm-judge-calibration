"""Reproducible, paired, fully retained cross-judge evidence and shared costs."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cli(monkeypatch):
    path = Path(__file__).resolve().parents[1]/"examples/rewardbench_audit.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("rewardbench_audit_cli", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_full_shared_evidence_and_deterministic_export(cli, tmp_path):
    args = ["--seeds", "2", "--output", str(tmp_path)]
    cli.main(args)
    report = (tmp_path/"REPORT.md").read_text(encoding="utf-8")
    # The two-seed direction differs from the published 30-seed example.
    assert "signed population tuning has lower MAE relative to raw agreement" in report
    trials = pd.read_csv(tmp_path/"trials.csv", float_precision="round_trip")
    summary = pd.read_csv(tmp_path/"summary.csv").set_index(["cohort", "target_judge", "labeled_fraction", "method"])
    assert len(trials) == 216 and len(summary) == 108
    for key, rows in trials.groupby(["cohort", "target_judge", "labeled_fraction", "method"]):
        assert summary.loc[key, "mean_absolute_error"] == pytest.approx(rows.absolute_error.mean())
        assert summary.loc[key, "mean_squared_error"] == pytest.approx(rows.squared_error.mean())
    pairs = pd.read_csv(tmp_path/"paired_method_differences.csv")
    indexed = trials.set_index(cli.KEYS+["method"])
    for row in pairs.itertuples():
        key = tuple(getattr(row, k) for k in cli.KEYS)
        for baseline, method in (("reference_audit", "human_only"), ("signed_population", "ppi_signed")):
            for metric in ("absolute_error", "squared_error"):
                assert getattr(row, f"{metric}_difference_vs_{baseline}") == pytest.approx(
                    indexed.loc[key+(row.method,), metric]-indexed.loc[key+(method,), metric], abs=1e-15)
    compressed = (tmp_path/"split_manifest.json.gz").read_bytes()
    assert compressed[4:8] == b"\x00\x00\x00\x00"
    encoded = json.loads(gzip.decompress(compressed))
    manifests = cli.decode_split_manifest(encoded)
    assert len(manifests) == 18
    assert len(encoded["sample_ids"]) == 2985 and len(encoded["instruction_ids"]) == 2733
    assert cli.encode_split_manifest(manifests) == encoded
    by_key = {(m["cohort"], m["seed"], m["labeled_fraction"]): m for m in manifests}
    for row in pd.read_csv(tmp_path/"shared_costs.csv").itertuples():
        manifest = by_key[row.cohort, row.seed, row.labeled_fraction]
        labeled, evaluation = set(manifest["labeled_sample_ids"]), set(manifest["evaluation_sample_ids"])
        assert not labeled.intersection(evaluation)
        references = set() if row.method == "raw_proxy" else labeled
        cached_ids = evaluation if row.method == "raw_proxy" else labeled if row.method == "human_only" else labeled | evaluation
        cached = {(sample, judge) for sample in cached_ids for judge in cli.JUDGES}
        scoring = {(sample, judge) for sample in evaluation for judge in cli.JUDGES}
        assert row.inference_reference_labels_union == len(references)
        assert row.inference_cached_judgments_union == len(cached)
        assert row.inference_and_scoring_reference_union == len(references | evaluation)
        assert row.inference_and_scoring_cached_union == len(cached | scoring)
    new = trials.loc[trials.method == "audit_residual"]
    assert new[["interval_low", "interval_high", "interval_width", "standard_error", "estimated_variance_ratio"]].isna().all().all()
    config = json.loads((tmp_path/"config.json").read_text())
    assert config["primary_cohort"] == "NonLLMBar" and len(config["cohort_metadata"]) == 3
    fingerprints = json.loads((tmp_path/"reproducibility.json").read_text())
    before = {name: (tmp_path/name).read_bytes() for name in fingerprints["artifacts_sha256"]}
    for name, digest in fingerprints["artifacts_sha256"].items():
        assert hashlib.sha256(before[name]).hexdigest() == digest
    for name, digest in fingerprints["source_sha256"].items():
        assert hashlib.sha256((cli.ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest
    cli.main(args)
    assert before == {name: (tmp_path/name).read_bytes() for name in before}
    assert fingerprints == json.loads((tmp_path/"reproducibility.json").read_text())


def test_rejects_modified_labels_or_provenance(cli, tmp_path):
    labels = cli.ROOT/"reports/rewardbench/labels.csv"
    (tmp_path/"labels.csv").write_bytes(labels.read_bytes()+b"\n")
    (tmp_path/"dataset.json").write_bytes(labels.with_name("dataset.json").read_bytes())
    with pytest.raises(ValueError, match="certified complete"):
        cli.read_verified_labels(tmp_path/"labels.csv")
    (tmp_path/"labels.csv").write_bytes(labels.read_bytes())
    metadata = json.loads(labels.with_name("dataset.json").read_text())
    metadata["historical_invocation_verified"] = True
    (tmp_path/"dataset.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="provenance differs"):
        cli.read_verified_labels(tmp_path/"labels.csv")


def test_stale_plot_not_silently_kept(cli, tmp_path):
    (tmp_path/"agreement_vs_accuracy.svg").write_bytes(b"unrelated old figure")
    with pytest.raises(ValueError, match="stale or unrelated"):
        cli.main(["--seeds", "2", "--output", str(tmp_path)])
    assert (tmp_path/"agreement_vs_accuracy.svg").read_bytes() == b"unrelated old figure"
