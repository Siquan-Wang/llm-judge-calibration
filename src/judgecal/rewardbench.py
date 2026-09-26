"""Strict offline extraction of two pinned RewardBench correctness caches.

The caches contain one randomly ordered judgment per comparison, not paired
presentation orders. The binary scores encode correctness against the benchmark
reference. Matching two reconstructed canonical choices supplies a gold-free
agreement proxy; an individual correctness score is not a gold-free prediction.
Only derived labels, identifiers and source metadata leave the parser. No model
calls, network access, optional parquet reader, or source text export occurs here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
import io
import json
from numbers import Integral, Real

import pandas as pd


REWARDBENCH_DATASET = "allenai/reward-bench"
REWARDBENCH_REVISION = "168d848cdbbea9764fae4a544dc9ca1e6cca4931"
REWARDBENCH_RESULTS_DATASET = "allenai/reward-bench-results"
REWARDBENCH_RESULTS_REVISION = "96302cc604e4f257dc526272c803c4eff27e3469"
REWARDBENCH_CODE_REVISION = "47512a010f748dcdacab422654b43c51c91fa63d"
REWARDBENCH_JUDGES = (
    "openai/gpt-4o-2024-08-06", "openai/gpt-4o-mini-2024-07-18",
)
REWARDBENCH_SUBSETS = frozenset((
    "alpacaeval-easy", "alpacaeval-hard", "alpacaeval-length",
    "mt-bench-easy", "mt-bench-med", "mt-bench-hard", "llmbar-natural",
    "llmbar-adver-neighbor", "llmbar-adver-GPTInst", "llmbar-adver-GPTOut",
    "llmbar-adver-manual", "refusals-dangerous", "refusals-offensive",
    "xstest-should-refuse", "xstest-should-respond", "donotanswer", "hep-cpp",
    "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust", "math-prm",
))
LABEL_COLUMNS = (
    "sample_id", "instruction_id", "source_id", "subset", "judge",
    "reference_choice", "canonical_choice", "reference_correct",
    "agreement_proxy", "auxiliary_judge",
)
_DATA = f"https://huggingface.co/datasets/{REWARDBENCH_DATASET}/resolve/{REWARDBENCH_REVISION}/"
_RESULTS = f"https://huggingface.co/datasets/{REWARDBENCH_RESULTS_DATASET}/resolve/{REWARDBENCH_RESULTS_REVISION}/"
_CODE = f"https://raw.githubusercontent.com/allenai/reward-bench/{REWARDBENCH_CODE_REVISION}/"


def _source(url: str, sha256: str, size: int, role: str) -> dict:
    return {"url": url, "sha256": sha256, "bytes": size, "role": role}


# Exact upstream bytes inspected independently of the derived labels. The code
# revision documents available semantics/configuration, not a proved run commit.
SOURCE_FILES = {
    "filtered.parquet": _source(_DATA + "data/filtered-00000-of-00001.parquet",
        "65473c20ed0627e02503557ae102c25c6b0e66b5ed69ee327e224bb95e70ca29", 2354751, "reference_dataset"),
    "gpt-4o-2024-08-06.json": _source(_RESULTS + "eval-set-scores/openai/gpt-4o-2024-08-06.json",
        "fee650e9066e2be5cb4bba1a605ea905720aa24a87eaf0113c09e4742761e069", 6871466, "binary_judge_cache"),
    "gpt-4o-mini-2024-07-18.json": _source(_RESULTS + "eval-set-scores/openai/gpt-4o-mini-2024-07-18.json",
        "5fcd13fad7b57b29d6120a9782ac762080a796860c66f87d9da8d61ea13bfc77", 6871471, "binary_judge_cache"),
    "gpt-4o-2024-08-06-aggregate.json": _source(_RESULTS + "eval-set/openai/gpt-4o-2024-08-06.json",
        "bc89223df6622c5a571bded58622a340f8ed6051d4c3820170054069e99ab2ad", 969, "published_subset_aggregates"),
    "gpt-4o-mini-2024-07-18-aggregate.json": _source(_RESULTS + "eval-set/openai/gpt-4o-mini-2024-07-18.json",
        "b3152175ae2ebbb2d634f677b41410d4a0ac03d6b727ca06e703da72504a9468", 991, "published_subset_aggregates"),
    "DATASET_README.md": _source(_DATA + "README.md",
        "819086ae803ffe3d3126eb2c188b8c67e2a0acb91b67096727f59f5aff423256", 21113, "dataset_card"),
    "DATA_LICENSE.md": _source(_DATA + "LICENSE.md",
        "d1659bcee5039d7de41a1a57bc003a9f8a55ffaf397753ae05b66bce6561ed60", 19792, "dataset_subset_license_notices"),
    "RESULTS_README.md": _source(_RESULTS + "README.md",
        "15b3ad0f0b498cae879a6524be4b2309c9177cf17526de97976644274f3cf012", 2409, "results_research_release_card"),
    "CODE_LICENSE": _source("https://raw.githubusercontent.com/allenai/reward-bench/bc72fb2a573fc31c614eef3405d354b398977b02/LICENSE",
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", 11357, "code_license_only"),
    "generative_47512a.py": _source(_CODE + "rewardbench/generative.py",
        "3f8526a7d5359a1aaf16537e107f245dad8e04e70819d87248d467f66ab17db7", 21806, "available_judge_helpers"),
    "run_generative_47512a.py": _source(_CODE + "scripts/run_generative.py",
        "8187db5c301d2ebd94356abdfefb22579511c8a5ca8ab9ada97b2dab86795b8e", 15249, "available_single_order_runner"),
}


def _identifier(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Source id must be a nonnegative integer, excluding booleans")
    return int(value)


def _subset(value: object) -> str:
    if not isinstance(value, str) or value not in REWARDBENCH_SUBSETS:
        raise ValueError("Unknown RewardBench subset")
    return value


def _binary(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Real) or value not in (0, 1):
        raise ValueError("Cached correctness must be numeric binary 0 or 1, excluding booleans")
    return int(value)


def _conversation(value: object) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Cached conversation must have exactly user and assistant messages")
    for message, role in zip(value, ("user", "assistant")):
        if (not isinstance(message, Mapping) or set(message) != {"role", "content"}
                or message["role"] != role or not isinstance(message["content"], str)):
            raise ValueError("Unexpected cached conversation message schema")
    return value[0]["content"], value[1]["content"]


def _canonical_key(prompt: str, chosen: str, rejected: str) -> str:
    return json.dumps([prompt, sorted((chosen, rejected))], ensure_ascii=False,
                      separators=(",", ":"))


def extract_rewardbench_labels(dataset_records: Sequence[Mapping],
                              caches_by_judge: Mapping[str, Mapping]) -> pd.DataFrame:
    """Join exact content, then certify metadata, for the two selected caches.

    Small fixtures are accepted. Source bytes/population size are certified only
    by ``verify_source_bytes`` and ``build_rewardbench_metadata``. IDs never join
    records: the public source has a duplicated numeric ID. Hashes exclude the
    reference orientation, and exact prompt hashes group across subset labels.
    Strings are neither stripped nor Unicode-normalized. Unsupported scores,
    including the source runner's 0.5 unknown fallback, are rejected, not coerced.
    """
    if (not isinstance(dataset_records, Sequence)
            or isinstance(dataset_records, (str, bytes)) or not dataset_records):
        raise ValueError("Expected a nonempty sequence of dataset records")
    if not isinstance(caches_by_judge, Mapping) or set(caches_by_judge) != set(REWARDBENCH_JUDGES):
        raise ValueError("Expected exactly the two pinned OpenAI judge caches")
    reference = {}
    canonical_seen = set()
    dataset_keys = {"prompt", "chosen", "chosen_model", "rejected", "rejected_model", "subset", "id"}
    for record in dataset_records:
        if not isinstance(record, Mapping) or set(record) != dataset_keys:
            raise ValueError("Unexpected reference dataset record schema")
        if any(not isinstance(record[k], str) for k in ("prompt", "chosen", "rejected", "chosen_model", "rejected_model")):
            raise ValueError("Reference text and model metadata must be strings")
        prompt, chosen, rejected = (record[k] for k in ("prompt", "chosen", "rejected"))
        if chosen == rejected:
            raise ValueError("Identical responses have no unique canonical orientation")
        canonical_key = _canonical_key(prompt, chosen, rejected)
        if canonical_key in canonical_seen:
            raise ValueError("Duplicate full comparison, including reversed orientation")
        canonical_seen.add(canonical_key)
        sample_id = hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()
        instruction_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        key = (prompt, chosen, rejected)
        reference[key] = (sample_id, instruction_id, _identifier(record["id"]),
                          _subset(record["subset"]), 1 if chosen < rejected else 2)
    scores = {}
    cache_keys = {"id", "model", "model_type", "results", "subset", "text_chosen", "text_rejected"}
    array_keys = ("id", "results", "subset", "text_chosen", "text_rejected")
    for judge in REWARDBENCH_JUDGES:
        cache = caches_by_judge[judge]
        if (not isinstance(cache, Mapping) or set(cache) != cache_keys
                or cache["model"] != judge or cache["model_type"] != "Generative RM"):
            raise ValueError("Unexpected cache schema or model identity")
        if any(not isinstance(cache[k], list) or len(cache[k]) != len(reference) for k in array_keys):
            raise ValueError("Cache columns must include every reference comparison exactly once")
        matched = {}
        for index in range(len(reference)):
            prompt, chosen = _conversation(cache["text_chosen"][index])
            other_prompt, rejected = _conversation(cache["text_rejected"][index])
            if prompt != other_prompt:
                raise ValueError("Chosen and rejected cache prompts disagree")
            key = (prompt, chosen, rejected)
            if key not in reference or key in matched:
                raise ValueError("Cache content does not match unique full reference comparisons")
            info = reference[key]
            if (_identifier(cache["id"][index]) != info[2]
                    or _subset(cache["subset"][index]) != info[3]):
                raise ValueError("Cache metadata disagrees with the full-content reference match")
            matched[key] = _binary(cache["results"][index])
        scores[judge] = matched
    output = []
    for key, info in reference.items():
        choices = {judge: info[4] if scores[judge][key] else 3 - info[4]
                   for judge in REWARDBENCH_JUDGES}
        agreement = int(choices[REWARDBENCH_JUDGES[0]] == choices[REWARDBENCH_JUDGES[1]])
        for judge, auxiliary in (REWARDBENCH_JUDGES, REWARDBENCH_JUDGES[::-1]):
            output.append((*info[:4], judge, info[4], choices[judge],
                           scores[judge][key], agreement, auxiliary))
    return pd.DataFrame(output, columns=LABEL_COLUMNS).sort_values(
        ["subset", "sample_id", "judge"], kind="mergesort"
    ).reset_index(drop=True)


def labels_csv_bytes(labels: pd.DataFrame) -> bytes:
    """Serialize the exact derived schema with deterministic UTF-8/LF bytes."""
    if not isinstance(labels, pd.DataFrame) or tuple(labels.columns) != LABEL_COLUMNS:
        raise ValueError("Expected the exact RewardBench derived-label schema")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(LABEL_COLUMNS)
    writer.writerows(labels.itertuples(index=False, name=None))
    return output.getvalue().encode("utf-8")


def verify_source_bytes(source_bytes: Mapping[str, bytes]) -> dict[str, str]:
    """Verify all eleven pinned source/provenance files, without reading paths."""
    if not isinstance(source_bytes, Mapping) or set(source_bytes) != set(SOURCE_FILES):
        raise ValueError("Expected exactly the eleven pinned source files")
    hashes = {}
    for name, expected in SOURCE_FILES.items():
        data = source_bytes[name]
        if not isinstance(data, bytes):
            raise ValueError("Pinned source input must be bytes")
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"Pinned source bytes do not match: {name}")
        hashes[name] = digest
    return hashes


def build_rewardbench_metadata(labels: pd.DataFrame, *,
                              verified_sources: Mapping[str, str]) -> dict:
    """Describe the certified public panel, retaining licensing limitations.

    Pass the digest mapping returned by ``verify_source_bytes`` after extracting
    the corresponding verified files. This routine cannot bind an arbitrary
    parsed Python object back to source bytes; that linkage belongs to the
    preparation runner. It does enforce the published derived-panel counts.
    """
    expected = {name: info["sha256"] for name, info in SOURCE_FILES.items()}
    if not isinstance(verified_sources, Mapping) or dict(verified_sources) != expected:
        raise ValueError("All pinned source hashes must have been verified")
    encoded = labels_csv_bytes(labels)
    unique = labels.drop_duplicates("sample_id")
    if (len(labels) != 5970 or len(unique) != 2985
            or unique["instruction_id"].nunique() != 2733
            or unique["source_id"].nunique() != 2984
            or set(labels["judge"]) != set(REWARDBENCH_JUDGES)
            or set(labels["subset"]) != REWARDBENCH_SUBSETS
            or not labels.groupby("sample_id")["judge"].agg(set).map(
                lambda x: x == set(REWARDBENCH_JUDGES)).all()):
        raise ValueError("Derived labels do not have the certified public-panel dimensions")
    return {
        "dataset": REWARDBENCH_DATASET, "dataset_revision": REWARDBENCH_REVISION,
        "results_dataset": REWARDBENCH_RESULTS_DATASET,
        "results_revision": REWARDBENCH_RESULTS_REVISION,
        "available_code_revision": REWARDBENCH_CODE_REVISION,
        "historical_invocation_verified": False,
        "judges": list(REWARDBENCH_JUDGES), "comparisons": len(unique),
        "rows": len(labels), "instruction_groups": 2733, "unique_source_ids": 2984,
        "source_id_is_unique": False, "source_id_join_used": False,
        "subsets": sorted(REWARDBENCH_SUBSETS),
        "subset_comparisons": {k: int(v) for k, v in unique.groupby("subset").size().items()},
        "labels_sha256": hashlib.sha256(encoded).hexdigest(), "labels_bytes": len(encoded),
        "label_columns": list(LABEL_COLUMNS), "sources": {
            name: dict(info) for name, info in SOURCE_FILES.items()},
        "source_text_exported": False,
        "sample_id_encoding": "sha256(UTF-8 compact ensure_ascii=False JSON [prompt, sorted([chosen,rejected])])",
        "instruction_id_encoding": "sha256(exact prompt UTF-8), no normalization or subset salt",
        "canonical_choices": "1/2 index in lexicographically sorted exact response strings",
        "outcome": "designated judge correctness against heterogeneous benchmark reference",
        "proxy": "equality of the two canonical choices; invariant to consistent gold reversal",
        "presentation": "one cached randomly swapped judgment per model; not both orders",
        "unknown_score_policy": "reject; only exact numeric 0/1 accepted, no 0.5 coercion",
        "label_cost": "one reference reveal gives correctness for both cached judges",
        "dataset_license": "ODC-BY dataset card plus source-subset license notices; see pinned DATA_LICENSE.md",
        "cache_license": "named license not specified at pinned results revision; README describes release for further research",
        "code_license_scope": "Apache-2.0 applies to inspected upstream code; not inferred for caches or source datasets",
        "derived_artifact_license": "No upstream data or cache relicensing is asserted by the package license",
        "limitations": [
            "No exact historical command, random order, completion text, seed, or request timestamp in selected score caches",
            "Available source configuration is evidence of semantics, not proof of historical invocation",
            "Benchmark correctness is not latent truth or a uniform human-preference estimand",
            "Exact prompt grouping does not establish independence of semantically related prompts",
            "Both directions reuse the same references and model pair; they are not independent replications",
        ],
    }
