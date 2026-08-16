"""Fetch a deterministic multi-speaker child VAD validation subset.

The 6.8 GB Samromur Children archive supports HTTP byte ranges. This tool reads
its central directory remotely and downloads only the selected FLAC members.
`remotezip` and `soundfile` are benchmark-only dependencies and are not part of
AudioForge's runtime package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from release_provenance import sha256_file as _sha256

import numpy as np
from scipy.io import wavfile

ARCHIVE_URL = (
    "https://openslr.trmal.net/resources/117/samromur_children_21.09.zip"
)
OPENS_LR_PAGE = "https://www.openslr.org/117/"
ARCHIVE_ROOT = "samromur_children_21.09"
LICENSE = "CC BY 4.0"
SELECTION_SEED = "audioforge-vad-child-v2"
SPEAKERS_PER_AGE_GENDER = 2
UTTERANCES_PER_SPEAKER = 2
PADDING_SECONDS = 0.5


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_key(value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{value}".encode()).hexdigest()


def _eligible_test_rows(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [
        row
        for row in rows
        if row["status"] == "test"
        and row["is_valid"] == "1.0"
        and row["empty"] == "0"
        and row["gender"] in {"female", "male"}
        and row["age"].isdigit()
        and 6 <= int(row["age"]) <= 16
        and 1.5 <= float(row["duration"]) <= 6.0
        and float(row["marosijo_score"]) >= 0.8
    ]


def _select_rows(
    rows: list[dict[str, str]],
    *,
    speakers_per_group: int = SPEAKERS_PER_AGE_GENDER,
    utterances_per_speaker: int = UTTERANCES_PER_SPEAKER,
) -> list[dict[str, str]]:
    by_group_speaker: dict[
        tuple[int, str], dict[str, list[dict[str, str]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_group_speaker[(int(row["age"]), row["gender"])][
            row["speaker_id"]
        ].append(row)

    selected: list[dict[str, str]] = []
    for group in sorted(by_group_speaker):
        speaker_rows = by_group_speaker[group]
        eligible_speakers = [
            speaker_id
            for speaker_id, candidates in speaker_rows.items()
            if len(candidates) >= utterances_per_speaker
        ]
        eligible_speakers.sort(
            key=lambda speaker_id: _stable_key(
                f"{group[0]}:{group[1]}:{speaker_id}"
            )
        )
        chosen_speakers = eligible_speakers[:speakers_per_group]
        if len(chosen_speakers) != speakers_per_group:
            raise RuntimeError(f"insufficient speakers for age/gender group {group}")
        for speaker_id in chosen_speakers:
            candidates = sorted(
                speaker_rows[speaker_id],
                key=lambda row: _stable_key(
                    f"{speaker_id}:{row['filename']}"
                ),
            )
            selected.extend(candidates[:utterances_per_speaker])
    return selected


def _decode_flac(payload: bytes) -> tuple[int, np.ndarray]:
    try:
        import soundfile as sf  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "Install benchmark-only dependency soundfile==0.13.1"
        ) from exc

    audio, sample_rate = sf.read(
        io.BytesIO(payload),
        dtype="float32",
        always_2d=False,
    )
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=1, dtype=np.float32)
    if mono.ndim != 1 or mono.size == 0:
        raise ValueError("selected FLAC is empty or has unsupported dimensions")
    return int(sample_rate), np.nan_to_num(mono)


def _render_capture(
    *,
    payload: bytes,
    row: dict[str, str],
    output_root: Path,
) -> dict[str, Any]:
    sample_rate, speech = _decode_flac(payload)
    if sample_rate != 16_000:
        raise ValueError(f"unexpected Samromur sample rate {sample_rate}")
    padding = int(round(PADDING_SECONDS * sample_rate))
    rendered = np.zeros(speech.size + 2 * padding, dtype=np.float32)
    rendered[padding : padding + speech.size] = speech
    pcm = np.round(np.clip(rendered, -1.0, 1.0) * 32767.0).astype(np.int16)

    relative_path = (
        Path(f"age_{row['age']}")
        / row["gender"]
        / row["speaker_id"]
        / Path(row["filename"]).with_suffix(".wav")
    )
    destination = output_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(destination, sample_rate, pcm)
    return {
        "path": relative_path.as_posix(),
        "split": "external_child_test",
        "condition": f"samromur_age_{row['age']}_{row['gender']}",
        "sample_rate": sample_rate,
        "speech_intervals_samples": [[padding, padding + int(speech.size)]],
        "speaker_id": row["speaker_id"],
        "age": int(row["age"]),
        "gender": row["gender"],
        "duration_s": float(row["duration"]),
        "validation_score": float(row["marosijo_score"]),
        "source_member": (
            f"{ARCHIVE_ROOT}/data_test/{row['speaker_id']}/{row['filename']}"
        ),
        "source_flac_sha256": _sha256_bytes(payload),
        "rendered_wav_sha256": _sha256(destination),
        "label_scope": (
            "The validated source is one read utterance. The interval is the "
            "source clip boundary after deterministic padding; it is not a "
            "phoneme-level annotation, so onset delay is descriptive only."
        ),
    }


def fetch(
    *,
    metadata_path: Path,
    output_root: Path,
    speakers_per_group: int = SPEAKERS_PER_AGE_GENDER,
    utterances_per_speaker: int = UTTERANCES_PER_SPEAKER,
) -> Path:
    try:
        from remotezip import RemoteZip  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "Install benchmark-only dependency remotezip==0.12.3"
        ) from exc

    selected = _select_rows(
        _eligible_test_rows(metadata_path),
        speakers_per_group=speakers_per_group,
        utterances_per_speaker=utterances_per_speaker,
    )
    captures: list[dict[str, Any]] = []
    with RemoteZip(ARCHIVE_URL) as archive:
        for index, row in enumerate(selected, start=1):
            member = (
                f"{ARCHIVE_ROOT}/data_test/"
                f"{row['speaker_id']}/{row['filename']}"
            )
            payload = archive.read(member)
            captures.append(
                _render_capture(
                    payload=payload,
                    row=row,
                    output_root=output_root,
                )
            )
            print(f"[{index}/{len(selected)}] {member}")

    manifest = {
        "schema_version": 2,
        "description": (
            "Deterministic, age/gender-stratified, speaker-disjoint external "
            "child VAD validation subset from Samromur Children 21.09."
        ),
        "source": OPENS_LR_PAGE,
        "archive_url": ARCHIVE_URL,
        "license": LICENSE,
        "metadata_sha256": _sha256(metadata_path),
        "selection": {
            "seed": SELECTION_SEED,
            "status": "test",
            "ages": list(range(6, 17)),
            "genders": ["female", "male"],
            "speakers_per_age_gender": speakers_per_group,
            "utterances_per_speaker": utterances_per_speaker,
            "speaker_count": len({row["speaker_id"] for row in selected}),
            "capture_count": len(captures),
            "duration_range_s": [1.5, 6.0],
            "minimum_validation_score": 0.8,
            "padding_seconds": PADDING_SECONDS,
            "redistribution": "local evaluation only; never packaged",
        },
        "benchmark_dependencies": {
            "remotezip": "0.12.3",
            "soundfile": "0.13.1",
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
        "--metadata",
        type=Path,
        default=Path("models/vad_edge_corpus_sources/samromur_metadata.tsv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/vad_child_multispeaker_corpus"),
    )
    args = parser.parse_args()
    manifest = fetch(
        metadata_path=args.metadata.resolve(),
        output_root=args.output.resolve(),
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
