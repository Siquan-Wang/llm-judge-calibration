"""Check full-grid evidence retention and deterministic finite-study exports."""
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
    path = Path(__file__).resolve().parents[1]/"examples/finite_sample_study.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("finite_study_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lossless_deterministic_complete_experiment(cli, tmp_path):
    args = ["--output", str(tmp_path), "--repetitions", "2"]
    cli.main(args)
    compressed = (tmp_path/"trials.csv.gz").read_bytes()
    assert compressed[4:8] == b"\x00\x00\x00\x00"
    trials = pd.read_csv(io.BytesIO(gzip.decompress(compressed)))
    summary = pd.read_csv(tmp_path/"summary.csv")
    assert len(trials) == 14*8*2
    assert len(summary) == 14*8
    assert set(trials.method) == set(cli.NAMES)
    observed = trials.groupby(["scenario", "method"]).agg(
        coverage=("covered", "mean"), mean_width=("interval_width", "mean"))
    saved = summary.set_index(["scenario", "method"]).sort_index()
    np.testing.assert_allclose(observed.coverage, saved.coverage)
    np.testing.assert_allclose(observed.mean_width, saved.mean_width)
    assert (saved.coverage_mc_low <= saved.coverage).all()
    assert (saved.coverage_mc_high >= saved.coverage).all()
    assert (saved.loc[saved.coverage == 1, "coverage_mc_low"] < 1).all()
    assert not summary.loc[summary.method.str.contains("normal"), "theorem_applicable"].any()
    assert not summary.loc[summary.validity_scope != "iid_same_population", "theorem_applicable"].any()
    manifest = json.loads((tmp_path/"reproducibility.json").read_text())
    before = {name: (tmp_path/name).read_bytes() for name in manifest["artifacts_sha256"]}
    for name, digest in manifest["artifacts_sha256"].items():
        assert hashlib.sha256(before[name]).hexdigest() == digest
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((cli.ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest
    config = json.loads((tmp_path/"config.json").read_text())
    assert config["repetitions"] == 2 and config["seed"] == 2028
    plot_data = cli.iid_plot_data(summary, config)
    assert plot_data.scenario.nunique() == 12
    # The dependent-row stress case shares a profile with an iid case; it
    # must never be mixed into that case's main coverage/width curve.
    assert plot_data.scenario_family.eq("iid").all()
    assert not plot_data.duplicated(["profile", "method", "n_labeled"]).any()
    cli.main(args)
    assert before == {name: (tmp_path/name).read_bytes() for name in before}
    assert json.loads((tmp_path/"reproducibility.json").read_text()) == manifest


def test_stale_artifacts_are_never_silently_rehashed(cli, tmp_path):
    (tmp_path/"finite_sample_tradeoff.svg").write_bytes(b"older figure")
    with pytest.raises(ValueError, match="stale or unrelated"):
        cli.main(["--output", str(tmp_path), "--repetitions", "2"])
    assert (tmp_path/"finite_sample_tradeoff.svg").read_bytes() == b"older figure"
