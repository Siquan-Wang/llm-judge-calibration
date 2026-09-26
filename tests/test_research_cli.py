"""Research artifacts must remain attributable and internally reproducible."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "examples/research_study.py"
    spec = importlib.util.spec_from_file_location("research_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_input_integrity_checked_before_analysis(cli, tmp_path):
    source = cli.ROOT / "reports/mtbench/labels.csv"
    (tmp_path/"labels.csv").write_bytes(source.read_bytes() + b"\n")
    (tmp_path/"dataset.json").write_bytes(source.with_name("dataset.json").read_bytes())
    with pytest.raises(ValueError, match="checksum"):
        cli.read_verified_labels(tmp_path/"labels.csv")


def test_runner_preserves_input_study(cli):
    with pytest.raises(ValueError, match="preserve the input study"):
        cli.main(["--output", str(cli.ROOT/"reports/mtbench"), "--repetitions", "2", "--seeds", "1"])


def test_stale_plot_cannot_be_misattributed_to_new_run(cli, tmp_path):
    (tmp_path/"simulation.png").write_bytes(b"an older figure")
    with pytest.raises(ValueError, match="stale or unrelated"):
        cli.main(["--output", str(tmp_path), "--repetitions", "2", "--seeds", "1"])
    assert (tmp_path/"simulation.png").read_bytes() == b"an older figure"


def test_research_outputs_are_deterministic_and_hash_verified(cli, tmp_path):
    args = ["--output", str(tmp_path), "--repetitions", "2", "--seeds", "1"]
    cli.main(args)
    manifest = json.loads((tmp_path/"reproducibility.json").read_text())
    before = {name: (tmp_path/name).read_bytes() for name in manifest["artifacts_sha256"]}
    for name, checksum in manifest["artifacts_sha256"].items():
        assert hashlib.sha256(before[name]).hexdigest() == checksum
        if name.endswith((".csv", ".json", ".md", ".svg")):
            assert b"\r\n" not in before[name]
    for name, checksum in manifest["source_sha256"].items():
        content = (cli.ROOT/name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(content).hexdigest() == checksum
    config = json.loads((tmp_path/"config.json").read_text())
    assert config["simulation_repetitions"] == 2
    assert config["split_seeds"] == 1
    assert len(pd.read_csv(tmp_path/"trials.csv")) == 15*3*4
    assert len(pd.read_csv(tmp_path/"simulation_trials.csv")) == 7*2*4
    cli.main(args)
    assert before == {name: (tmp_path/name).read_bytes() for name in before}
    assert json.loads((tmp_path/"reproducibility.json").read_text()) == manifest
