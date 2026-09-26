"""Offline source-fixture tests: join semantics, canonical orientation, failures."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from judgecal import llmbar as lb


def documents():
    result = {}
    for subset in lb.SUBSETS:
        gold = [dict(input="shared instruction" if subset in ("Natural", "Neighbor") else subset,
                     output_1=f"{subset} answer a {i}", output_2=f"{subset} answer b {i}", label=i+1)
                for i in range(2)]
        result[lb._gold_path(subset)] = gold
        for judge in lb.JUDGES:
            cached = copy.deepcopy(gold)
            for row in cached:
                row["results"] = [{
                    "swap = False": {"winner": "1", "completion": ["Output (a)", "stop"]},
                    # The raw completion chooses displayed b after swapping:
                    # upstream has already mapped that choice to original 1.
                    "swap = True": {"winner": "1", "completion": ["Output (b)", "stop"]},
                }]
            result[lb._cache_path(subset, judge)] = cached
    return result


def test_content_join_reordering_and_repeated_instruction_group():
    docs = documents()
    expected = lb.extract_llmbar_labels(docs)
    for rows in docs.values():
        rows.reverse()
    actual = lb.extract_llmbar_labels(docs)
    pd.testing.assert_frame_equal(actual, expected)
    assert tuple(actual.columns) == lb.LABEL_COLUMNS
    assert len(actual) == 30 and actual.sample_id.nunique() == 10
    assert actual.instruction_id.nunique() == 4
    assert (actual.forward_prediction == actual.reverse_prediction).all()
    assert (actual.reverse_prediction == 1).all()  # never flip an already canonical winner
    assert not set(lb._TEXT_KEYS).intersection(actual.columns)
    assert actual.groupby("sample_id").judge.nunique().eq(3).all()


def test_ids_exclude_gold_but_preserve_exact_text():
    docs = documents()
    original = lb.extract_llmbar_labels(docs)
    for rows in docs.values():
        for row in rows:
            row["label"] = 3-row["label"]
    flipped = lb.extract_llmbar_labels(docs)
    pd.testing.assert_frame_equal(original.drop(columns="reference_label"), flipped.drop(columns="reference_label"))
    assert (original.reference_label == 3-flipped.reference_label).all()
    row = docs[lb._gold_path("Natural")][0]
    expected = hashlib.sha256(json.dumps([row[k] for k in lb._TEXT_KEYS], ensure_ascii=False,
                                        separators=(",", ":")).encode()).hexdigest()
    assert expected in set(flipped.sample_id)
    assert hashlib.sha256(row["input"].encode()).hexdigest() in set(flipped.instruction_id)


def test_invalid_prediction_retained_with_zero_without_changing_gold():
    docs = documents()
    row = docs[lb._cache_path("GPTOut", "LLaMA2")][0]
    row["results"][0]["swap = True"]["winner"] = None
    frame = lb.extract_llmbar_labels(docs)
    assert len(frame) == 30
    bad = frame.loc[frame.reverse_prediction == 0]
    assert len(bad) == 1 and bad.iloc[0].reference_label == 1
    assert bad.iloc[0].forward_prediction == 1


@pytest.mark.parametrize("bad", [True, 1, 1.0, "0", "A", "", [], {}])
def test_unknown_cached_winner_rejected(bad):
    docs = documents()
    docs[lb._cache_path("Natural", "GPT-4")][0]["results"][0]["swap = True"]["winner"] = bad
    with pytest.raises(ValueError, match="winner"):
        lb.extract_llmbar_labels(docs)


@pytest.mark.parametrize("bad", [None, True, "1", 1.0, 0, 3])
def test_invalid_reference_rejected(bad):
    docs = documents()
    docs[lb._gold_path("Natural")][0]["label"] = bad
    with pytest.raises(ValueError, match="Reference label"):
        lb.extract_llmbar_labels(docs)


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.pop(), "every gold"),
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "every gold"),
    (lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])), "unique gold"),
    (lambda rows: rows[0].__setitem__("output_2", "wrong answer"), "does not match"),
    (lambda rows: rows[0].__setitem__("label", 2), "disagrees"),
    (lambda rows: rows[0].__setitem__("extra", 1), "schema"),
    (lambda rows: rows[0].__setitem__("input", 1), "strings"),
    (lambda rows: rows[0].__setitem__("results", []), "one run"),
    (lambda rows: rows[0]["results"].append(copy.deepcopy(rows[0]["results"][0])), "one run"),
    (lambda rows: rows[0]["results"][0].pop("swap = True"), "one run"),
    (lambda rows: rows[0]["results"][0]["swap = True"].__setitem__("completion", "Output (b)"), "completion schema"),
])
def test_misaligned_and_unknown_source_schema_rejected(mutation, match):
    docs = documents()
    mutation(docs[lb._cache_path("Natural", "GPT-4")])
    with pytest.raises(ValueError, match=match):
        lb.extract_llmbar_labels(docs)


def test_duplicate_gold_and_missing_documents_rejected():
    docs = documents()
    docs[lb._gold_path("Natural")][1] = docs[lb._gold_path("Natural")][0]
    with pytest.raises(ValueError, match="Duplicate"):
        lb.extract_llmbar_labels(docs)
    docs = documents()
    docs.pop(lb._cache_path("Manual", "LLaMA2"))
    with pytest.raises(ValueError, match="exactly"):
        lb.extract_llmbar_labels(docs)


def test_bounded_source_download_and_digest_validation(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return self.payload[:limit]
    response = Response()
    monkeypatch.setattr(lb, "urlopen", lambda *a, **k: response)
    response.payload = b"changed public data"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        lb._read_pinned_source("LICENSE", 1)
    response.payload = b"x" * 1_000_001
    with pytest.raises(ValueError, match="large"):
        lb._read_pinned_source("LICENSE", 1)


@pytest.mark.parametrize("timeout", [True, "30", 0, -1, float("nan"), float("inf"), 10**1000])
def test_invalid_timeout_before_network(timeout, monkeypatch):
    monkeypatch.setattr(lb, "urlopen", lambda *a, **k: pytest.fail("network called"))
    with pytest.raises(ValueError, match="timeout"):
        lb.fetch_llmbar_labels(timeout=timeout)


def test_cli_idempotent_lf_hash_and_no_overwrite(tmp_path):
    path = Path(__file__).parents[1] / "examples" / "prepare_llmbar.py"
    spec = importlib.util.spec_from_file_location("prepare_llmbar", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = lb.extract_llmbar_labels(documents())
    payload = lb.labels_csv_bytes(frame)
    assert b"\r\n" not in payload
    manifest = dict(dataset_id=lb.LLMBAR_DATASET, source_revision=lb.LLMBAR_REVISION,
                    labels_sha256=hashlib.sha256(payload).hexdigest(), source_url=lb.LLMBAR_URL,
                    citation="Fixture attribution", paper_url="https://arxiv.org/abs/2310.07641",
                    upstream_license_text="Fixture license")
    sentinel = tmp_path / "unrelated.txt"
    sentinel.write_text("preserve me")
    module.write_derived_files(frame, manifest, tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    module.write_derived_files(frame, manifest, tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert (tmp_path / "labels.csv").read_bytes() == payload
    assert sentinel.read_text() == "preserve me"
    (tmp_path / "dataset.json").write_text("unrelated existing content")
    with pytest.raises(FileExistsError, match="Refusing"):
        module.write_derived_files(frame, manifest, tmp_path)
    assert (tmp_path / "dataset.json").read_text() == "unrelated existing content"
    manifest["labels_sha256"] = "0"*64
    with pytest.raises(ValueError, match="do not match"):
        module.write_derived_files(frame, manifest, tmp_path)


def test_published_snapshot_integrity_and_all_parse_failures_retained():
    folder = Path(__file__).parents[1] / "reports" / "llmbar"
    payload = (folder / "labels.csv").read_bytes()
    manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(payload).hexdigest() == lb.LLMBAR_LABELS_SHA256 == manifest["labels_sha256"]
    assert b"\r\n" not in payload
    frame = pd.read_csv(folder / "labels.csv")
    assert len(frame) == 1257
    assert frame.sample_id.nunique() == 419 and frame.instruction_id.nunique() == 418
    assert set(frame.columns) == set(lb.LABEL_COLUMNS)
    assert frame.groupby("judge").size().to_dict() == dict.fromkeys(lb.JUDGES, 419)
    assert frame.drop_duplicates("sample_id").groupby("subset").size().to_dict() == lb._COUNTS
    assert not frame.duplicated(["sample_id", "judge"]).any()
    assert frame.groupby("sample_id").reference_label.nunique().eq(1).all()
    forward_invalid = frame.loc[frame.forward_prediction == 0]
    reverse_invalid = frame.loc[frame.reverse_prediction == 0]
    assert list(zip(forward_invalid.subset, forward_invalid.judge)) == [("GPTInst", "LLaMA2")]
    assert list(zip(reverse_invalid.subset, reverse_invalid.judge)) == [("GPTOut", "LLaMA2")]
    sources = {item["path"]: item for item in manifest["sources"]}
    assert {path: item["sha256"] for path, item in sources.items()} == lb.SOURCE_SHA256
    assert sum(item["bytes"] for item in sources.values()) == 2630581
    assert all(lb.LLMBAR_REVISION in item["url"] for item in sources.values())
    assert "Copyright (c) 2023 Princeton Natural Language Processing" in (folder / "DATA_LICENSE.md").read_text()
