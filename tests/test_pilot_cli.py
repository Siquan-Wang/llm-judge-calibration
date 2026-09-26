"""Published-study export integrity, non-default confidence and safe reruns."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def pilot_cli():
    path = Path(__file__).resolve().parents[1]/"examples/independent_pilot_study.py"
    spec = importlib.util.spec_from_file_location("pilot_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_is_complete_reproducible_and_fingerprinted(pilot_cli, tmp_path):
    output = tmp_path/"report"
    args = ["--repetitions", "2", "--seed", "17", "--alpha", ".10", "--output", str(output)]
    pilot_cli.main(args)
    contents = {path.name: path.read_bytes() for path in output.iterdir()}
    manifest = json.loads(contents["reproducibility.json"])
    for name, expected in manifest["artifacts_sha256"].items():
        assert hashlib.sha256(contents[name]).hexdigest() == expected
    for name, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((pilot_cli.ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected
    trials = pd.read_csv(output/"trials.csv.gz")
    summary = pd.read_csv(output/"summary.csv")
    paired = pd.read_csv(output/"paired_differences.csv")
    assert len(trials) == 192 and len(summary) == 96 and len(paired) == 576
    assert trials.groupby(["scenario", "method"]).size().eq(2).all()
    assert set(trials.alpha) == {.1}
    assert trials.total_labels_used.equals(trials.total_budget)
    config = json.loads(contents["study_config.json"])
    assert config["alpha"] == .1 and config["repetitions"] == 2
    report = contents["REPORT.md"].decode("utf-8")
    assert "192 retained method trials" in report
    assert "--repetitions 2 --seed 17 --alpha 0.1" in report
    assert "squared fractions" in report
    assert "no new estimator or theorem" in report
    assert len(gzip.decompress(contents["trials.csv.gz"]).splitlines()) == 193
    pilot_cli.main(args)
    assert contents == {path.name: path.read_bytes() for path in output.iterdir()}


def test_unrelated_files_are_not_overwritten(pilot_cli, tmp_path):
    sentinel = tmp_path/"do-not-touch.txt"
    sentinel.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="unrelated or stale"):
        pilot_cli.main(["--output", str(tmp_path), "--repetitions", "2"])
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == "original"


def test_plot_nominal_reference_tracks_requested_alpha(pilot_cli, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes
    _, summary, _ = pilot_cli.independent_pilot_study(repetitions=2, seed=17, alpha=.10)
    horizontal_references = []
    original = Axes.axhline

    def capture(self, y=0, *args, **kwargs):
        horizontal_references.append(y)
        return original(self, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "axhline", capture)
    pilot_cli.make_plots(summary, tmp_path, alpha=.10)
    assert horizontal_references.count(90) == 4
    assert 95 not in horizontal_references
    assert {path.name for path in tmp_path.iterdir()} == {
        "pilot_point_tradeoff.png", "pilot_point_tradeoff.svg",
        "pilot_interval_tradeoff.png", "pilot_interval_tradeoff.svg",
    }
