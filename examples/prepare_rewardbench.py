"""Rebuild a text-free RewardBench panel from pinned public source bytes.

No model is called. Reading Parquet requires the optional ``data`` extra.
Existing source files are verified rather than silently replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

from judgecal.rewardbench import (
    REWARDBENCH_JUDGES as JUDGES, SOURCE_FILES, build_rewardbench_metadata, extract_rewardbench_labels,
    labels_csv_bytes, verify_source_bytes,
)
from research_study import ROOT, write_json


def read_sources(directory: Path, download: bool = False) -> dict[str, bytes]:
    """Fetch only the enumerated pinned files, with explicit byte limits."""
    result = {}
    for name, specification in SOURCE_FILES.items():
        path = directory / name
        if path.exists():
            payload = path.read_bytes()
        elif download:
            request = Request(specification["url"], headers={"User-Agent": "judgecal-research/0.10"})
            with urlopen(request, timeout=60) as response:
                payload = response.read(specification["bytes"] + 1)
            if len(payload) != specification["bytes"] or hashlib.sha256(payload).hexdigest() != specification["sha256"]:
                raise ValueError(f"pinned source integrity failed: {name}")
            directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        else:
            raise FileNotFoundError(f"missing pinned source {name}; supply --download or a complete --source-directory")
        result[name] = payload
    verify_source_bytes(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=ROOT / ".cache/rewardbench-sources")
    parser.add_argument("--download", action="store_true", help="download missing pinned public source files only")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/rewardbench")
    args = parser.parse_args(argv)
    sources = read_sources(args.source_directory, args.download)
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError('Install the optional data extra: pip install -e ".[data]"') from error
    records = parquet.read_table(io.BytesIO(sources["filtered.parquet"])).to_pylist()
    caches = {judge: json.loads(sources[judge.rsplit("/", 1)[-1] + ".json"]) for judge in JUDGES}
    labels = extract_rewardbench_labels(records, caches)
    metadata = build_rewardbench_metadata(labels, verified_sources=verify_source_bytes(sources))
    # Independent source aggregates are an additional orientation/scoring check.
    for judge in JUDGES:
        aggregate = json.loads(sources[judge.rsplit("/", 1)[-1] + "-aggregate.json"])
        means = labels.loc[labels.judge == judge].groupby("subset").reference_correct.mean()
        for subset, mean in means.items():
            if subset not in aggregate or abs(float(aggregate[subset]) - float(mean)) > 1e-14:
                raise ValueError(f"source aggregate disagrees for {judge}: {subset}")
    metadata["aggregate_validation"] = {"judges": len(JUDGES), "subsets_per_judge": len(means),
                                         "maximum_allowed_absolute_difference": 1e-14}
    payload = labels_csv_bytes(labels)
    metadata["labels_sha256"] = hashlib.sha256(payload).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    allowed = {"labels.csv", "dataset.json", "NOTICE.md"}
    if {path.name for path in args.output.iterdir()} - allowed:
        raise ValueError("output directory contains unrelated files")
    (args.output / "labels.csv").write_bytes(payload)
    write_json(args.output / "dataset.json", metadata)
    notice = (ROOT / "reports/rewardbench/NOTICE.md").read_bytes()
    (args.output / "NOTICE.md").write_bytes(notice)
    print(f"Verified {len(records)} comparisons and wrote {len(labels)} text-free judge rows.")


if __name__ == "__main__":
    main()
