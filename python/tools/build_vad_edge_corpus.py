"""Build a local real-child VAD edge corpus from pinned OpenSLR SLR98."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from scipy.io import wavfile

SOURCE = "https://www.openslr.org/98/"
LICENSE = "CC BY-NC-ND 4.0"
ARCHIVE_NAME = "Parent-ChildVocalInteraction.tar.gz"
METADATA_NAME = "Parent_Child_Vocal_Interaction.json"
PER_LOCATION = 20


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(source_root: Path, output_root: Path) -> Path:
    archive_path = source_root / ARCHIVE_NAME
    metadata_path = source_root / METADATA_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected: list[tuple[str, str, dict[str, Any]]] = []
    for location, entries in sorted(metadata.items()):
        child_only = [
            (key, value)
            for key, value in entries.items()
            if value.get("speaker") == "b"
        ]
        child_only.sort(
            key=lambda item: (
                -float(item[1].get("length", 0.0)),
                int(item[1].get("label", -1)),
                item[0],
            )
        )
        selected.extend(
            (location, key, value)
            for key, value in child_only[:PER_LOCATION]
        )

    captures: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for location, key, metadata_row in selected:
            member_name = f"{location}/{key}.wav"
            member = archive.getmember(member_name)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"could not read {member_name}")
            payload = extracted.read()
            sample_rate, audio = wavfile.read(io.BytesIO(payload))
            destination = output_root / location / f"{key}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            captures.append(
                {
                    "path": destination.relative_to(output_root).as_posix(),
                    "split": "held_out",
                    "condition": f"real_child_{location.lower()}",
                    "sample_rate": int(sample_rate),
                    "speech_intervals_samples": [[0, len(audio)]],
                    "source_member": member_name,
                    "source_sha256": _sha256_bytes(payload),
                    "age": int(metadata_row["age"]),
                    "label": int(metadata_row["label"]),
                    "duration_s": float(metadata_row["length"]),
                }
            )

    manifest = {
        "schema_version": 1,
        "description": (
            "Unmodified child-only SLR98 utterances selected by longest duration "
            "within each of three recording environments."
        ),
        "source": SOURCE,
        "license": LICENSE,
        "archive": {
            "path": str(archive_path.as_posix()),
            "sha256": _sha256(archive_path),
        },
        "selection": {
            "speaker": "b (child)",
            "age": 5,
            "per_location": PER_LOCATION,
            "capture_count": len(captures),
            "redistribution": "local evaluation only; never packaged",
        },
        "captures": captures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("models/vad_edge_corpus_sources"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/vad_edge_eval_corpus"),
    )
    args = parser.parse_args()
    manifest = build(args.sources.resolve(), args.output.resolve())
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
