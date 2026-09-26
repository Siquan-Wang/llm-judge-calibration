"""Fetch pinned public caches and publish only derived LLMBar labels/provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from judgecal.llmbar import (
    LLMBAR_DATASET, LLMBAR_REVISION, data_license_notice, fetch_llmbar_labels,
    labels_csv_bytes,
)


def write_derived_files(frame, manifest, output: Path) -> None:
    """Write only three named artifacts; existing nonidentical files are refused.

    Identical reruns are a no-op. Unrelated directory entries are untouched.
    All conflict checks happen before any file is written.
    """
    labels = labels_csv_bytes(frame)
    if (manifest.get("dataset_id") != LLMBAR_DATASET
            or manifest.get("source_revision") != LLMBAR_REVISION
            or manifest.get("labels_sha256") != hashlib.sha256(labels).hexdigest()):
        raise ValueError("Labels and pinned dataset manifest do not match")
    artifacts = {
        "labels.csv": labels,
        "dataset.json": (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        "DATA_LICENSE.md": data_license_notice(manifest).encode("utf-8"),
    }
    output = Path(output)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise FileExistsError(f"Output must be a regular directory: {output}")
    for name, payload in artifacts.items():
        path = output / name
        if path.is_symlink() or (path.exists() and (not path.is_file() or path.read_bytes() != payload)):
            raise FileExistsError(f"Refusing to overwrite nonidentical artifact: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        path = output / name
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "reports" / "llmbar")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    frame, manifest = fetch_llmbar_labels(timeout=args.timeout)
    write_derived_files(frame, manifest, args.output)
    print(json.dumps({k: manifest[k] for k in (
        "dataset_id", "source_revision", "labels_sha256", "rows", "comparisons",
        "instruction_groups", "invalid_predictions", "source_total_bytes",
    )}, indent=2))


if __name__ == "__main__":
    main()
