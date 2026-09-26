"""Offline CLI exports, replay identities, provenance and output safety."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def corpus_cli():
    path = Path(__file__).resolve().parents[1] / "examples/finite_corpus_study.py"
    spec = importlib.util.spec_from_file_location("finite_corpus_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exported(corpus_cli, tmp_path_factory):
    output = tmp_path_factory.mktemp("finite-corpus-export") / "report"
    args = ["--repetitions", "2", "--seed", "17", "--alpha", ".10", "--output", str(output)]
    corpus_cli.main(args)
    return output, args


def _contents(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def test_complete_nondefault_alpha_export_and_fingerprints(corpus_cli, exported):
    output, _ = exported
    contents = _contents(output)
    assert set(contents) == {
        "trials.csv.gz", "summary.csv", "paired_differences.csv", "splits.csv",
        "census.csv", "frame_catalog.json", "permutations.json.gz",
        "study_config.json", "REPORT.md", "reproducibility.json",
    }
    tables = {name: pd.read_csv(output / name, float_precision="round_trip") for name in
              ("trials.csv.gz", "summary.csv", "paired_differences.csv", "census.csv", "splits.csv")}
    trials, summary, paired, census, splits = tables.values()
    assert [len(table) for table in tables.values()] == [48, 24, 36, 8, 6]
    assert trials.groupby(["target_judge", "group_fraction", "method"]).size().eq(2).all()
    assert trials.groupby(["repetition", "group_fraction"]).size().eq(8).all()
    assert set(trials.alpha) == set(summary.alpha) == {.1}
    assert set(trials.n_audited_groups) == {463, 1389, 2083}
    assert set(trials.n_groups) == {2315} and set(trials.n_rows) == {2566}
    assert set(trials.method) == set(corpus_cli.METHODS)
    assert census.exact.all() and census.width.eq(0).all()
    assert census.point.equals(census.truth)
    assert census.n_audited_groups.eq(2315).all() and census.audited_rows.eq(2566).all()
    assert paired.groupby(["target_judge", "group_fraction"]).size().eq(6).all()
    primary = paired.loc[(paired.method == "agreement_grid_ebs") & (paired.baseline == "size_grid_ebs")]
    assert len(primary) == 6
    for row in primary.itertuples():
        cell = trials.loc[(trials.target_judge == row.target_judge) &
                          (trials.group_fraction == row.group_fraction)]
        errors = cell.pivot(index="repetition", columns="method", values="squared_error")
        differences = errors["agreement_grid_ebs"] - errors["size_grid_ebs"]
        assert row.mse_difference == pytest.approx(differences.mean(), abs=1e-18)
        assert row.mse_mcse == pytest.approx(differences.std(ddof=1) / np.sqrt(2), abs=1e-18)

    config = json.loads(contents["study_config.json"])
    assert config["alpha"] == .1 and config["seed"] == 17 and config["repetitions"] == 2
    assert config["new_model_calls"] == 0 and config["plots"] is False
    assert config["audit_group_counts"] == [463, 1389, 2083]
    manifest = json.loads(contents["reproducibility.json"])
    assert set(manifest["artifacts_sha256"]) == set(contents) - {"reproducibility.json"}
    for name, expected in manifest["artifacts_sha256"].items():
        assert _sha(contents[name]) == expected
    for name, expected in manifest["source_sha256"].items():
        assert not Path(name).is_absolute()
        assert _sha((corpus_cli.ROOT / name).read_bytes().replace(b"\r\n", b"\n")) == expected
    for name, expected in manifest["input_sha256"].items():
        assert _sha((corpus_cli.ROOT / "reports/rewardbench" / name).read_bytes()) == expected
    assert manifest["input_sha256"]["labels.csv"] == corpus_cli.LABELS_SHA256
    assert str(output) not in json.dumps(manifest)
    assert str(corpus_cli.ROOT) not in json.dumps(manifest)
    report = contents["REPORT.md"].decode("utf-8")
    assert "48 retained method-trial rows" in report
    assert "--repetitions 2 --seed 17 --alpha 0.1" in report
    assert "no new estimator or theorem" in report
    assert "same-audit finite-grid selector" in report
    assert "squared" in report  # MSE units must differ from plotted RMSE units.


def test_saved_catalog_permutations_and_split_hashes_replay_every_sample(exported):
    output, _ = exported
    catalog = json.loads((output / "frame_catalog.json").read_text(encoding="utf-8"))
    config = json.loads((output / "study_config.json").read_text(encoding="utf-8"))
    permutations = json.loads(gzip.decompress((output / "permutations.json.gz").read_bytes()))
    canonical = json.dumps(catalog, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    assert _sha(canonical) == config["frame_sha256"] == permutations["frame_sha256"]
    assert catalog["row_ids"] == sorted(set(catalog["row_ids"]))
    groups = catalog["groups"]
    assert [group["instruction_id"] for group in groups] == sorted(group["instruction_id"] for group in groups)
    assert sorted(index for group in groups for index in group["row_indices"]) == list(range(2566))
    splits = pd.read_csv(output / "splits.csv")
    trials = pd.read_csv(output / "trials.csv.gz")
    assert [draw["repetition"] for draw in permutations["draws"]] == [0, 1]
    for draw in permutations["draws"]:
        repetition = draw["repetition"]
        permutation = draw["permutation"]
        expected = np.random.Generator(np.random.Philox(np.random.SeedSequence([17, repetition]))).permutation(2315)
        assert permutation == expected.tolist()
        assert sorted(permutation) == list(range(2315))
        previous = set()
        for split in splits.loc[splits.repetition == repetition].sort_values("group_fraction").itertuples():
            selected = permutation[:split.n_audited_groups]
            assert previous <= set(selected)
            previous = set(selected)
            selected_groups = [groups[index] for index in selected]
            group_ids = sorted(group["instruction_id"] for group in selected_groups)
            row_ids = sorted(catalog["row_ids"][index] for group in selected_groups for index in group["row_indices"])
            assert _sha(("\n".join(group_ids) + "\n").encode()) == split.selected_group_sha256
            assert _sha(("\n".join(row_ids) + "\n").encode()) == split.selected_row_sha256
            assert len(row_ids) == split.audited_rows
            shared = trials.loc[(trials.repetition == repetition) & (trials.group_fraction == split.group_fraction)]
            assert shared.selected_group_sha256.eq(split.selected_group_sha256).all()
            assert shared.selected_row_sha256.eq(split.selected_row_sha256).all()
            assert shared.audited_rows.eq(split.audited_rows).all()


def test_rerun_and_new_destination_are_byte_identical(corpus_cli, exported, tmp_path):
    output, args = exported
    before = _contents(output)
    for name in ("trials.csv.gz", "permutations.json.gz"):
        raw = before[name]
        assert raw[:2] == b"\x1f\x8b" and raw[4:8] == b"\0\0\0\0"
        assert not raw[3] & 8  # No destination filename in the gzip header.
    assert len(gzip.decompress(before["trials.csv.gz"]).splitlines()) == 49
    corpus_cli.main(args)
    assert _contents(output) == before
    elsewhere = tmp_path / "elsewhere"
    corpus_cli.main(args[:-1] + [str(elsewhere)])
    assert _contents(elsewhere) == before


def _copy_input(corpus_cli, directory, labels_name="labels.csv"):
    directory.mkdir()
    source = corpus_cli.ROOT / "reports/rewardbench"
    for name in ("labels.csv", "dataset.json", "NOTICE.md"):
        destination = labels_name if name == "labels.csv" else name
        (directory / destination).write_bytes((source / name).read_bytes())
    return directory / labels_name


@pytest.mark.parametrize("changed", ["labels.csv", "dataset.json"])
def test_input_pins_reject_changes_before_any_export(corpus_cli, tmp_path, monkeypatch, changed):
    path = _copy_input(corpus_cli, tmp_path / "source")
    altered = path.with_name(changed)
    altered.write_bytes(altered.read_bytes() + b"\n")
    output = tmp_path / "report"

    def forbid_study(*args, **kwargs):
        pytest.fail("study must not run after an input pin mismatch")

    monkeypatch.setattr(corpus_cli, "finite_corpus_study", forbid_study)
    with pytest.raises(ValueError, match="certified"):
        corpus_cli.main(["--input", str(path), "--output", str(output), "--repetitions", "2"])
    assert not output.exists()


def test_actual_input_path_is_recorded_under_logical_labels_name(corpus_cli, tmp_path):
    path = _copy_input(corpus_cli, tmp_path / "source", "certified-renamed.csv")
    output = tmp_path / "report"
    corpus_cli.main(["--input", str(path), "--output", str(output), "--repetitions", "2"])
    manifest = json.loads((output / "reproducibility.json").read_text(encoding="utf-8"))
    assert manifest["input_sha256"]["labels.csv"] == _sha(path.read_bytes())


def test_missing_notice_is_detected_before_any_export(corpus_cli, tmp_path, monkeypatch):
    path = _copy_input(corpus_cli, tmp_path / "source")
    path.with_name("NOTICE.md").unlink()
    output = tmp_path / "report"

    def forbid_study(*args, **kwargs):
        pytest.fail("required provenance must be checked before running the study")

    monkeypatch.setattr(corpus_cli, "finite_corpus_study", forbid_study)
    with pytest.raises((ValueError, FileNotFoundError)):
        corpus_cli.main(["--input", str(path), "--output", str(output), "--repetitions", "2"])
    assert not output.exists()


def test_unrelated_files_are_preserved(corpus_cli, tmp_path):
    sentinel = tmp_path / "do-not-touch.txt"
    sentinel.write_bytes(b"original user content")
    with pytest.raises(ValueError, match="unrelated or stale"):
        corpus_cli.main(["--output", str(tmp_path), "--repetitions", "2"])
    assert _contents(tmp_path) == {sentinel.name: b"original user content"}


def test_stale_plot_artifact_is_not_silently_removed(corpus_cli, exported):
    output, args = exported
    before = _contents(output)
    stale = output / "finite_corpus_tradeoff.svg"
    stale.write_bytes(b"previous plot")
    try:
        with pytest.raises(ValueError, match="unrelated or stale"):
            corpus_cli.main(args)
        assert _contents(output) == {**before, stale.name: b"previous plot"}
    finally:
        stale.unlink()


def test_plot_nominal_reference_uses_requested_alpha(corpus_cli, exported, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    output, _ = exported
    summary = pd.read_csv(output / "summary.csv")
    references = []
    original = Axes.axhline

    def capture(self, y=0, *args, **kwargs):
        references.append(y)
        return original(self, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "axhline", capture)
    corpus_cli.make_plot(summary, tmp_path, .1)
    assert references == [90, 90]
    assert {path.name for path in tmp_path.iterdir()} == {
        "finite_corpus_tradeoff.png", "finite_corpus_tradeoff.svg",
    }
