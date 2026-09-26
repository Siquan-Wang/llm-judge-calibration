import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "examples/stress_grid.py"
    spec = importlib.util.spec_from_file_location("stress_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_grid_smoke_is_lossless_and_reproducible(cli, tmp_path):
    args = ["--output", str(tmp_path), "--repetitions", "2"]
    cli.main(args)
    compressed = (tmp_path/"trials.csv.gz").read_bytes()
    assert compressed[4:8] == b"\x00\x00\x00\x00"
    trials = pd.read_csv(io.BytesIO(gzip.decompress(compressed)))
    assert len(trials) == 42*2*4
    assert trials.scenario.nunique() == 42
    assert set(trials.method) == {"raw_judge", "human_only", "ppi", "ppi_tuned"}
    manifest = json.loads((tmp_path/"reproducibility.json").read_text())
    for name, digest in manifest["artifacts_sha256"].items():
        assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() == digest
    cli.main(args)
    assert (tmp_path/"trials.csv.gz").read_bytes() == compressed
    assert json.loads((tmp_path/"reproducibility.json").read_text()) == manifest


def test_stress_runner_preserves_unrelated_files(cli, tmp_path):
    (tmp_path/"notes.txt").write_text("keep this")
    with pytest.raises(ValueError, match="stale or unrelated"):
        cli.main(["--output", str(tmp_path), "--repetitions", "2"])
    assert (tmp_path/"notes.txt").read_text() == "keep this"
