"""Offline schema, content-join, orientation and provenance contracts."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from judgecal import rewardbench as rb


def fixture_sources():
    # A repeated source ID and an exact prompt shared across subsets are intended.
    records = [
        dict(prompt="same prompt", chosen="z response", rejected="a response",
             chosen_model="a", rejected_model="b", subset="donotanswer", id=3692),
        dict(prompt="same prompt", chosen="b response", rejected="c response",
             chosen_model="c", rejected_model="d", subset="hep-python", id=3692),
        dict(prompt="Unicode: é\n", chosen="你好", rejected="再见",
             chosen_model="e", rejected_model="f", subset="llmbar-natural", id=7),
    ]
    caches = {}
    for judge, scores in zip(rb.REWARDBENCH_JUDGES, ([1, 0, 1], [1, 1, 0])):
        caches[judge] = dict(
            model=judge, model_type="Generative RM", id=[r["id"] for r in records],
            subset=[r["subset"] for r in records], results=list(scores),
            text_chosen=[[dict(role="user", content=r["prompt"]),
                          dict(role="assistant", content=r["chosen"])] for r in records],
            text_rejected=[[dict(role="user", content=r["prompt"]),
                            dict(role="assistant", content=r["rejected"])] for r in records],
        )
    return records, caches


def extract(records=None, caches=None):
    if records is None:
        records, caches = fixture_sources()
    return rb.extract_rewardbench_labels(records, caches)


def test_exact_schema_choices_and_gold_free_agreement():
    records, caches = fixture_sources()
    labels = extract(records, caches)
    assert tuple(labels.columns) == rb.LABEL_COLUMNS
    assert len(labels) == 6
    assert labels.sample_id.nunique() == 3
    assert labels.instruction_id.nunique() == 2
    for row in labels.itertuples():
        record = next(r for r in records if r["subset"] == row.subset)
        score = caches[row.judge]["results"][records.index(record)]
        assert row.reference_choice == sorted([record["chosen"], record["rejected"]]).index(record["chosen"]) + 1
        assert row.canonical_choice == (row.reference_choice if score else 3 - row.reference_choice)
        assert row.reference_correct == int(row.canonical_choice == row.reference_choice)
        assert row.auxiliary_judge in rb.REWARDBENCH_JUDGES
        assert row.auxiliary_judge != row.judge
        pair = labels[labels.sample_id == row.sample_id]
        assert row.agreement_proxy == int(pair.canonical_choice.nunique() == 1)
        assert row.agreement_proxy == int(pair.reference_correct.nunique() == 1)
    assert not any(c in labels for c in ("prompt", "chosen", "rejected", "text_chosen", "text_rejected"))


def test_comparison_hash_is_canonical_utf8_json_and_prompt_group_is_global():
    records, caches = fixture_sources()
    labels = extract(records, caches)
    for record in records:
        row = labels[labels.subset == record["subset"]].iloc[0]
        canonical = json.dumps([record["prompt"], sorted([record["chosen"], record["rejected"]])],
                               ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        assert row.sample_id == hashlib.sha256(canonical).hexdigest()
        assert row.instruction_id == hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest()
    repeated = labels[labels.source_id == 3692]
    assert len(repeated) == 4
    assert repeated.sample_id.nunique() == 2
    assert repeated.instruction_id.nunique() == 1
    assert repeated.subset.nunique() == 2


def test_dataset_and_cache_row_order_are_independently_irrelevant():
    records, caches = fixture_sources()
    expected = extract(records, caches)
    records = records[::-1]
    for index, judge in enumerate(rb.REWARDBENCH_JUDGES):
        order = ([1, 2, 0], [2, 0, 1])[index]
        for key, value in caches[judge].items():
            if isinstance(value, list):
                caches[judge][key] = [value[i] for i in order]
    actual = extract(records, dict(reversed(list(caches.items()))))
    pd.testing.assert_frame_equal(actual, expected)
    assert rb.labels_csv_bytes(actual) == rb.labels_csv_bytes(expected)


def test_consistent_gold_reversal_preserves_ids_decisions_and_proxy():
    records, caches = fixture_sources()
    before = extract(records, caches)
    for record in records:
        record["chosen"], record["rejected"] = record["rejected"], record["chosen"]
        record["chosen_model"], record["rejected_model"] = record["rejected_model"], record["chosen_model"]
    for cache in caches.values():
        cache["text_chosen"], cache["text_rejected"] = cache["text_rejected"], cache["text_chosen"]
        cache["results"] = [1 - score for score in cache["results"]]
    after = extract(records, caches)
    unchanged = [c for c in rb.LABEL_COLUMNS if c not in ("reference_choice", "reference_correct")]
    pd.testing.assert_frame_equal(before[unchanged], after[unchanged])
    np.testing.assert_array_equal(after.reference_choice, 3 - before.reference_choice)
    np.testing.assert_array_equal(after.reference_correct, 1 - before.reference_correct)


def test_hidden_label_mutation_does_not_recompute_decisions_or_proxy():
    labels = extract()
    changed = labels.copy()
    changed["reference_choice"] = 3 - changed.reference_choice
    changed["reference_correct"] = (changed.canonical_choice == changed.reference_choice).astype(int)
    pd.testing.assert_frame_equal(labels[["sample_id", "instruction_id", "canonical_choice", "agreement_proxy"]],
                                  changed[["sample_id", "instruction_id", "canonical_choice", "agreement_proxy"]])
    assert (labels.reference_correct + changed.reference_correct == 1).all()


@pytest.mark.parametrize("score", [True, False, np.bool_(True), "0", "1", .5, -.1, 2,
                                      float("nan"), float("inf"), -float("inf"), 1 + 0j, None, [], {}])
def test_reject_nonbinary_or_coerced_scores(score):
    records, caches = fixture_sources()
    caches[rb.REWARDBENCH_JUDGES[0]]["results"][0] = score
    with pytest.raises(ValueError, match="numeric binary"):
        extract(records, caches)


@pytest.mark.parametrize("score", [0, 1, 0., 1., np.int64(0), np.float64(1)])
def test_accept_exact_real_binary_scores(score):
    records, caches = fixture_sources()
    caches[rb.REWARDBENCH_JUDGES[0]]["results"][0] = score
    assert len(extract(records, caches)) == 6


@pytest.mark.parametrize("source_id", [True, np.bool_(False), -1, 3.0, "3", None, float("nan")])
def test_reject_invalid_source_identifier(source_id):
    records, caches = fixture_sources()
    records[0]["id"] = source_id
    with pytest.raises(ValueError, match="nonnegative integer"):
        extract(records, caches)


@pytest.mark.parametrize("mutate", [
    lambda r, c: r[0].pop("chosen_model"),
    lambda r, c: r[0].update(extra="unexpected"),
    lambda r, c: r[0].update(prompt=None),
    lambda r, c: r[0].update(chosen_model=None),
    lambda r, c: r[0].update(subset="unknown"),
    lambda r, c: r[0].update(rejected=r[0]["chosen"]),
    lambda r, c: c.pop(rb.REWARDBENCH_JUDGES[1]),
    lambda r, c: c.update({"google/gemini-1.5-flash-001": deepcopy(c[rb.REWARDBENCH_JUDGES[0]])}),
    lambda r, c: c[rb.REWARDBENCH_JUDGES[0]].update(model="another-model"),
    lambda r, c: c[rb.REWARDBENCH_JUDGES[0]].update(model_type="Reward Model"),
    lambda r, c: c[rb.REWARDBENCH_JUDGES[0]].update(unexpected=[]),
    lambda r, c: c[rb.REWARDBENCH_JUDGES[0]]["results"].pop(),
    lambda r, c: c[rb.REWARDBENCH_JUDGES[0]].update(results=(1, 0, 1)),
    lambda r, c: c[rb.REWARDBENCH_JUDGES[0]]["id"].__setitem__(0, 99),
    lambda r, c: c[rb.REWARDBENCH_JUDGES[0]]["id"].__setitem__(0, True),
    lambda r, c: c[rb.REWARDBENCH_JUDGES[0]]["subset"].__setitem__(0, "hep-go"),
    lambda r, c: c[rb.REWARDBENCH_JUDGES[0]]["text_chosen"][0].pop(),
    lambda r, c: c[rb.REWARDBENCH_JUDGES[0]]["text_chosen"][0][0].update(role="system"),
    lambda r, c: c[rb.REWARDBENCH_JUDGES[0]]["text_chosen"][0][1].update(content=1),
    lambda r, c: c[rb.REWARDBENCH_JUDGES[0]]["text_chosen"][0][1].update(extra=1),
    lambda r, c: c[rb.REWARDBENCH_JUDGES[0]]["text_chosen"][0][0].update(content="different prompt"),
    lambda r, c: c[rb.REWARDBENCH_JUDGES[0]]["text_chosen"][0][1].update(content="different response"),
])
def test_strict_schema_and_full_content_metadata_join(mutate):
    records, caches = fixture_sources()
    mutate(records, caches)
    with pytest.raises(ValueError):
        extract(records, caches)


@pytest.mark.parametrize("records,caches", [([], {}), (None, {}), ("string", {}), ({}, {})])
def test_invalid_top_level_inputs(records, caches):
    with pytest.raises(ValueError):
        rb.extract_rewardbench_labels(records, caches)


@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_full_dataset_key_in_either_orientation_is_rejected(reverse):
    records, caches = fixture_sources()
    duplicate = deepcopy(records[0])
    duplicate["id"] = 999
    duplicate["subset"] = "hep-go"
    if reverse:
        duplicate["chosen"], duplicate["rejected"] = duplicate["rejected"], duplicate["chosen"]
    records.append(duplicate)
    with pytest.raises(ValueError, match="Duplicate full comparison"):
        extract(records, caches)


def test_duplicate_cache_key_with_equal_lengths_is_rejected():
    records, caches = fixture_sources()
    cache = caches[rb.REWARDBENCH_JUDGES[0]]
    for key, value in cache.items():
        if isinstance(value, list):
            cache[key][1] = deepcopy(cache[key][0])
    with pytest.raises(ValueError, match="unique full reference"):
        extract(records, caches)


def test_no_whitespace_or_unicode_normalization():
    records, caches = fixture_sources()
    original = extract(records, caches)
    records[2]["prompt"] += " "
    with pytest.raises(ValueError):
        extract(records, caches)
    for cache in caches.values():
        for key in ("text_chosen", "text_rejected"):
            cache[key][2][0]["content"] += " "
    changed = extract(records, caches)
    assert set(changed.sample_id) != set(original.sample_id)
    assert set(changed.instruction_id) != set(original.instruction_id)


def test_errors_never_echo_raw_prompt_or_response():
    records, caches = fixture_sources()
    secret = "UNIQUE_SOURCE_TEXT_THAT_MUST_NOT_APPEAR"
    caches[rb.REWARDBENCH_JUDGES[0]]["text_chosen"][0][1]["content"] = secret
    with pytest.raises(ValueError) as caught:
        extract(records, caches)
    assert secret not in str(caught.value)


def test_csv_exact_order_lf_and_no_text():
    labels = extract()
    data = rb.labels_csv_bytes(labels)
    assert data.splitlines()[0].decode() == ",".join(rb.LABEL_COLUMNS)
    assert data.count(b"\n") == 7
    assert b"\r" not in data
    assert b"same prompt" not in data
    assert rb.labels_csv_bytes(extract()) == data
    with pytest.raises(ValueError, match="exact RewardBench"):
        rb.labels_csv_bytes(labels[list(reversed(labels.columns))])


def test_pinned_manifest_contract():
    assert len(rb.SOURCE_FILES) == 11
    assert len(rb.REWARDBENCH_SUBSETS) == 23
    assert sum(source["bytes"] for source in rb.SOURCE_FILES.values()) < 17_000_000
    for name, source in rb.SOURCE_FILES.items():
        assert set(source) == {"url", "sha256", "bytes", "role"}
        assert len(source["sha256"]) == 64
        assert source["bytes"] > 0
        assert source["url"].startswith("https://")
        assert "gemini" not in name


def test_source_byte_verification_requires_exact_hash_length_and_names(monkeypatch):
    data = b"a small source fixture\n"
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(rb, "SOURCE_FILES", {"fixture": dict(url="https://example.invalid", sha256=digest,
                                                           bytes=len(data), role="test")})
    assert rb.verify_source_bytes({"fixture": data}) == {"fixture": digest}
    for bad in ({}, {"fixture": data, "extra": b""}, {"fixture": data[:-1]},
                {"fixture": data.replace(b"small", b"other")}, {"fixture": data.decode()}):
        with pytest.raises(ValueError):
            rb.verify_source_bytes(bad)


def test_metadata_requires_verified_sources_and_certified_population():
    with pytest.raises(ValueError, match="verified"):
        rb.build_rewardbench_metadata(extract(), verified_sources={})
    with pytest.raises(ValueError, match="certified public-panel"):
        rb.build_rewardbench_metadata(extract(), verified_sources={
            name: info["sha256"] for name, info in rb.SOURCE_FILES.items()})
