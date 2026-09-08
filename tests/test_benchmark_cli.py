import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "examples" / "benchmark_mtbench.py"
    spec = importlib.util.spec_from_file_location("benchmark_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unattributed_csv_cannot_be_reported_as_mtbench(cli, tmp_path, monkeypatch):
    source = tmp_path / "labels.csv"
    pd.DataFrame({"question_id": [1]}).to_csv(source, index=False)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--input", str(source)])
    with pytest.raises(ValueError, match="provenance sidecar"):
        cli.main()


def test_modified_csv_cannot_keep_verified_provenance(cli, tmp_path, monkeypatch):
    source = tmp_path / "labels.csv"
    pd.DataFrame({"question_id": [1]}).to_csv(source, index=False)
    (tmp_path / "dataset.json").write_text(json.dumps({
        "dataset_id": "lmsys/mt_bench_human_judgments",
        "dataset_revision": "f7d2896d2cc5d80f8b55c2bbc722613555233c25",
        "labels_sha256": "tampered",
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--input", str(source)])
    with pytest.raises(ValueError, match="checksum"):
        cli.main()
