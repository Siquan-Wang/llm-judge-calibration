"""Pinned, offline-ready extraction of public LLMBar cached judgments.

The reference is curated instruction-following correctness, not subjective
human preference. Cached winners for BOTH presentation orders already name the
original output (see the pinned upstream ``evaluate.py``); never reverse them
again. Source text is validated in memory and omitted from the derived table.
No model inference or API credentials are used.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
from numbers import Real
from typing import Mapping
from urllib.request import Request, urlopen

import pandas as pd


LLMBAR_DATASET = "princeton-nlp/LLMBar"
LLMBAR_REVISION = "900616bff90b6c6c8e1681f7d079250637c55992"
LLMBAR_LABELS_SHA256 = "aafff984b410bd1f60e0a3196a10957c5a1f241caf8c85c4c391d8b1e2ef1f97"
LLMBAR_URL = f"https://github.com/{LLMBAR_DATASET}/tree/{LLMBAR_REVISION}"
_RAW = f"https://raw.githubusercontent.com/{LLMBAR_DATASET}/{LLMBAR_REVISION}/"
SUBSETS = ("Natural", "Neighbor", "GPTInst", "GPTOut", "Manual")
JUDGES = ("GPT-4", "ChatGPT", "LLaMA2")
LABEL_COLUMNS = (
    "sample_id", "instruction_id", "subset", "judge", "reference_label",
    "forward_prediction", "reverse_prediction",
)
_TEXT_KEYS = ("input", "output_1", "output_2")
_COUNTS = dict(zip(SUBSETS, (100, 134, 92, 47, 46)))


def _subset_path(subset: str) -> str:
    return "Dataset/LLMBar/" + ("" if subset == "Natural" else "Adversarial/") + subset


def _gold_path(subset: str) -> str:
    return f"{_subset_path(subset)}/dataset.json"


def _cache_path(subset: str, judge: str) -> str:
    return f"{_subset_path(subset)}/evaluators/{judge}/Vanilla/result.json"


def _config_path(judge: str) -> str:
    return f"LLMEvaluator/evaluators/config/{judge}/Vanilla.json"


# SHA-256 of exact source bytes, independently inspected at the pinned revision.
# Pinning both revision and digest makes accidental source replacement fail closed.
_GOLD_HASHES = (
    "f7127b7aa78104cb8e03a2b1b3ac0077e95b125714dd441a8ee63666145a3375",
    "6712c7eda6342602e0bc7c06c282fa95e2110d4e741a284026a62d937f9780a0",
    "1002425f73c2cf54b6e6de99bce3f6c60fa2a17d13194772e37d7481c1351d42",
    "1ff56e520d7607750a9866a3be0f5b850392ca4477a1f69d80f32bd3e310f561",
    "0d9598769a39cdbbd265778c37e49939d41d5f76dcfb17bd75dcfa07516d3837",
)
_CACHE_HASHES = (
    ("4e3f72a6490e2cbac3243c0a144105a1b1c34baec2a82964bf181e9bbf8d5b4b",
     "94062f9b5c4b4be7fc74ce8d23315b53b97c97bdcbe613be089a0ffa188e686c",
     "0050748ad99fcd48242f79bd5688d54b6e487d0098dde4fd9591314a93de5636"),
    ("aa0f8137af29a893f92d9c8706ad0e873cf686a340345e7c1db7591775de0275",
     "54c0b167619c36980df1c1bac259790d6fc310a5bc2c826cd995d18641f911b6",
     "e234234894ebae0c78693857cc68dd3befab5ea599e8b94833da40dd935ddcda"),
    ("31ca00033a84f9442a3fc131cb17bf9c51b2693cf48b0c0ec97a894d759e1cfb",
     "7c6122b0cb68c5b02fff840e6657d5e03dfa7a69cef0b06e7f1b1cf589a4443a",
     "141c3f3caa774309983d077f2ecbef99cd05198645b836cada2038c43f4e4886"),
    ("bf711216e700aa25172fc3177b181919d65675fd8a32f90cd9fff297dd5ed6d4",
     "f47a67afe29c81b731b58f061704d1fc192d8dc6ae0044c108b36be0ed71aba1",
     "cdd4cd4d8459a50239306a3af14237409113a2e6370674ea99453ca0e46788c5"),
    ("49785f158253f346a97e63fed70fc1434f7b54768c7f7b5f6abd7dfa7f51f606",
     "ecd62d71890ee57c92deb125e77623c553dbc7a15ad720e6129b9045edfb48a9",
     "b87be8f0972f5b6c2ec516f2ad88c5f697e6cf0514487a77ba70de08fab762d2"),
)
SOURCE_SHA256 = {_gold_path(s): h for s, h in zip(SUBSETS, _GOLD_HASHES)}
SOURCE_SHA256.update({
    _cache_path(s, j): h for s, hs in zip(SUBSETS, _CACHE_HASHES)
    for j, h in zip(JUDGES, hs)
})
SOURCE_SHA256.update(dict(zip(
    [_config_path(j) for j in JUDGES],
    ("2f03c7d7f06b9901b48f77b39ed338e057bea30ea2605b7d41a588cf767e8087",
     "0c088384399a9f3dc4c674a829a962fe461c043f67a85b38ef41b6cf95f7f088",
     "b7051fadc6d3cea9f1e664eb9502cd59c41048d93e568ee2d8f6e8c975ccb7fa"),
)))
SOURCE_SHA256.update({
    "LICENSE": "8872dbf8660b00890b5e07ce1ea1f7a44fa7ae4e1857da56caba172b51dab3cb",
    "LLMEvaluator/evaluate.py": "8f9a645239d91d5ad0ffcf977425714e285663a853eefcd21b11bf43a525229a",
    "LLMEvaluator/evaluators/prompts/comparison/Vanilla.txt":
        "77a3a8f354472bc3b39a78847246c93b824bd1c59c57070bf91e06da927aa7da",
    "README.md": "07b3f12f2a4f283ce33e475456fb548f7adb81fee674ad345d9d92f59a49bd1c",
})


def _comparison(row: object, path: str, *, cached: bool) -> tuple:
    keys = set(_TEXT_KEYS) | {"label"} | ({"results"} if cached else set())
    if not isinstance(row, dict) or set(row) != keys:
        raise ValueError(f"Unexpected comparison schema in {path}")
    if any(not isinstance(row[k], str) for k in _TEXT_KEYS):
        raise ValueError(f"Comparison text must be strings in {path}")
    if type(row["label"]) is not int or row["label"] not in (1, 2):
        raise ValueError(f"Reference label must be integer 1 or 2 in {path}")
    return tuple(row[k] for k in _TEXT_KEYS)


def _winner(result: object, path: str) -> int:
    if not isinstance(result, dict) or set(result) != {"completion", "winner"}:
        raise ValueError(f"Unexpected order-result schema in {path}")
    completion = result["completion"]
    if not isinstance(completion, list) or len(completion) != 2:
        raise ValueError(f"Unexpected cached completion schema in {path}")
    winner = result["winner"]
    if winner is None:
        return 0
    if not isinstance(winner, str) or winner not in ("1", "2"):
        raise ValueError(f"Cached winner must be '1', '2', or null in {path}")
    return int(winner)


def extract_llmbar_labels(source_documents: Mapping[str, object]) -> pd.DataFrame:
    """Validate and join the 20 parsed gold/cache JSON documents by full content.

    This pure extractor also accepts small fixtures with the same five-subset,
    three-judge schema. Only ``fetch_llmbar_labels`` certifies the pinned public
    byte hashes and its 419-comparison population. No text is included in output
    or error messages. Null winners are retained as 0 (invalid), never dropped.
    Sample IDs hash the exact original input/output tuple, excluding gold;
    instruction IDs hash the exact input, without Unicode/whitespace folding.
    """
    paths = {_gold_path(s) for s in SUBSETS} | {
        _cache_path(s, j) for s in SUBSETS for j in JUDGES
    }
    if not isinstance(source_documents, Mapping) or set(source_documents) != paths:
        raise ValueError("Expected exactly the five gold and fifteen selected cache documents")
    records, seen = [], set()
    for subset in SUBSETS:
        path = _gold_path(subset)
        gold = source_documents[path]
        if not isinstance(gold, list) or not gold:
            raise ValueError(f"Expected a nonempty comparison list in {path}")
        references = {}
        for row in gold:
            key = _comparison(row, path, cached=False)
            if key in references or key in seen:
                raise ValueError(f"Duplicate full comparison in {path}")
            references[key] = row["label"]
            seen.add(key)
        for judge in JUDGES:
            path = _cache_path(subset, judge)
            cached = source_documents[path]
            if not isinstance(cached, list) or len(cached) != len(references):
                raise ValueError(f"Cache must contain every gold comparison exactly once in {path}")
            matched = set()
            for row in cached:
                key = _comparison(row, path, cached=True)
                if key not in references or key in matched:
                    raise ValueError(f"Cache content does not match unique gold comparisons in {path}")
                if row["label"] != references[key]:
                    raise ValueError(f"Cache reference label disagrees with gold in {path}")
                matched.add(key)
                results = row["results"]
                if (not isinstance(results, list) or len(results) != 1
                        or not isinstance(results[0], dict)
                        or set(results[0]) != {"swap = False", "swap = True"}):
                    raise ValueError(f"Expected one run with both presentation orders in {path}")
                predictions = [_winner(results[0][order], path)
                               for order in ("swap = False", "swap = True")]
                encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                records.append((hashlib.sha256(encoded).hexdigest(),
                                hashlib.sha256(key[0].encode("utf-8")).hexdigest(),
                                subset, judge, references[key], *predictions))
    return pd.DataFrame(records, columns=LABEL_COLUMNS).sort_values(
        ["subset", "sample_id", "judge"], kind="mergesort"
    ).reset_index(drop=True)


def labels_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Canonical UTF-8 CSV bytes with LF lines and the fixed public schema."""
    if tuple(frame.columns) != LABEL_COLUMNS:
        raise ValueError("Unexpected derived-label columns")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(LABEL_COLUMNS)
    writer.writerows(frame.itertuples(index=False, name=None))
    return output.getvalue().encode("utf-8")


def _read_pinned_source(path: str, timeout: float) -> bytes:
    request = Request(_RAW + path, headers={"User-Agent": "judgecal-public-data-reproduction"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read(1_000_001)
    if len(payload) > 1_000_000:
        raise ValueError(f"Unexpectedly large pinned source: {path}")
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256[path]:
        raise ValueError(f"Pinned source SHA-256 mismatch: {path}")
    return payload


def fetch_llmbar_labels(*, timeout: float = 30.0) -> tuple[pd.DataFrame, dict]:
    """Fetch about 2.6 MB of public pinned files; return labels and provenance.

    Downloads exist in memory only. The official public repository's existing
    inference is reused; this function never contacts a model API. Source hashes
    are checked before parsing. The manifest and CSV encoding are deterministic.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, Real):
        raise ValueError("timeout must be finite and positive")
    try:
        timeout = float(timeout)
    except (OverflowError, ValueError):
        raise ValueError("timeout must be finite and positive") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    paths = sorted(SOURCE_SHA256)
    with ThreadPoolExecutor(max_workers=6) as pool:
        payloads = dict(zip(paths, pool.map(lambda p: _read_pinned_source(p, timeout), paths)))
    docs = {p: json.loads(b.decode("utf-8")) for p, b in payloads.items()
            if p.startswith("Dataset/")}
    frame = extract_llmbar_labels(docs)
    unique = frame.drop_duplicates("sample_id")
    counts = unique.groupby("subset").size().to_dict()
    if counts != _COUNTS or len(unique) != 419 or unique.instruction_id.nunique() != 418:
        raise ValueError("Pinned LLMBar population counts changed")
    labels_digest = hashlib.sha256(labels_csv_bytes(frame)).hexdigest()
    if labels_digest != LLMBAR_LABELS_SHA256:
        raise ValueError("Derived labels differ from the independently checked snapshot")
    configs = {j: json.loads(payloads[_config_path(j)].decode("utf-8"))[0] for j in JUDGES}
    failures = {
        j: {"forward": int((g.forward_prediction == 0).sum()),
            "reverse": int((g.reverse_prediction == 0).sum())}
        for j, g in frame.groupby("judge")
    }
    manifest = {
        "schema_version": 1,
        "dataset_id": LLMBAR_DATASET,
        "source_revision": LLMBAR_REVISION,
        "source_url": LLMBAR_URL,
        "citation": "Zeng et al. (ICLR 2024), Evaluating Large Language Models at Evaluating Instruction Following",
        "paper_url": "https://arxiv.org/abs/2310.07641",
        "labels_sha256": labels_digest,
        "rows": len(frame), "comparisons": len(unique),
        "instruction_groups": int(unique.instruction_id.nunique()),
        "subset_counts": counts, "judges": list(JUDGES), "invalid_predictions": failures,
        "reference_label_counts": {str(k): int(v) for k, v in unique.reference_label.value_counts().sort_index().items()},
        "columns": list(LABEL_COLUMNS),
        "sample_id_definition": "SHA-256 of UTF-8 JSON [input,output_1,output_2], ensure_ascii=False, separators=(',',':'); excludes gold",
        "instruction_id_definition": "SHA-256 of exact input UTF-8; no normalization; split complete instruction groups",
        "join_policy": "Every cached full input/output tuple must match gold exactly once; reference labels checked separately; no row-position join",
        "prediction_definition": "Original output ID 1 or 2; null cached winner maps to 0 invalid. Both cached orders already canonical; reverse winner is NOT flipped again.",
        "canonicalization_source": "LLMEvaluator/evaluate.py: winner if not swap else reverse_label(winner)",
        "reference_definition": "Curated objective instruction-following reference; not subjective preference or unrestricted population ground truth",
        "selected_prompt_directory": "Vanilla",
        "prompt_interpretation": "Vanilla in the repository corresponds to Vanilla+Rules in the paper; Vanilla_NoRules is a different cache",
        "model_configurations": configs,
        "model_version_limitations": "GPT-4 is an undated gpt-4 alias; ChatGPT config names gpt-3.5-turbo with engine gpt-35-turbo-0613; LLaMA2 names meta-llama/Llama-2-70b-chat-hf without a weight revision. Cached-source pins reproduce this reanalysis, not new inference.",
        "selection_limitations": "Natural examples were filtered/modified; Adversarial examples were constructed/filtered against ChatGPT evaluators. Report cohorts separately; do not infer universal judge rankings.",
        "license": "MIT (official repository notice retained in DATA_LICENSE.md)",
        "upstream_license_text": payloads["LICENSE"].decode("utf-8").strip(),
        "raw_text_published": False,
        "sources": [{"path": p, "url": _RAW+p, "sha256": SOURCE_SHA256[p],
                     "bytes": len(payloads[p])} for p in paths],
        "source_total_bytes": sum(map(len, payloads.values())),
    }
    return frame, manifest


def data_license_notice(manifest: dict) -> str:
    """Attribution and complete pinned upstream license for derived labels."""
    return (
        "# LLMBar derived-label attribution\n\n"
        "Source: [Princeton NLP LLMBar](" + manifest["source_url"] + "). "
        + manifest["citation"] + ". [Paper](" + manifest["paper_url"] + ").\n\n"
        "The table contains stable hashes, curated reference labels, and cached judge "
        "choices for both presentation orders. It contains no source prompts, responses "
        "or completions. Null parsed winners are retained as 0, not removed. "
        "The repository's `Vanilla` cache is the paper's Vanilla+Rules condition.\n\n"
        "The source is a selected instruction-following benchmark, not a representative "
        "sample of subjective human preference. Adversarial selection used ChatGPT "
        "evaluators. Exact instruction hashes keep repeated prompts together. "
        "Model versions and all exact source byte hashes are recorded in `dataset.json`. "
        "No new model calls were made.\n\n"
        "The official source repository supplies the following MIT notice. "
        "Attribution is retained separately from this project's code license; this "
        "notice does not assert new rights over underlying third-party text, which "
        "is not redistributed here.\n\n```text\n"
        + manifest["upstream_license_text"] + "\n```\n"
    )
