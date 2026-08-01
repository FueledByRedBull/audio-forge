"""Fetch the native-48-kHz RAVDESS neutral repeated-take evaluation slice."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import urllib.parse
import urllib.request
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import wavfile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "models" / "cross_take_eval"
DATASET_PAGE = "https://doi.org/10.5281/zenodo.1188976"
ARCHIVE_URL = (
    "https://zenodo.org/records/1188976/files/"
    "Audio_Speech_Actors_01-24.zip?download=1"
)
OFFICIAL_MD5 = "bc696df654c87fed845eb13823edef8a"
LICENSE = "CC BY-NC-SA 4.0"
STATEMENTS = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}
EMOTIONS_AND_INTENSITIES = {
    ("01", "01"): "neutral-normal",
    ("02", "01"): "calm-normal",
    ("02", "02"): "calm-strong",
}


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validated_archive_url() -> str:
    parsed = urllib.parse.urlsplit(ARCHIVE_URL)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "zenodo.org"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.fragment
    ):
        raise ValueError("RAVDESS archive URL must use trusted Zenodo HTTPS")
    return ARCHIVE_URL


def _download(destination: Path) -> None:
    request = urllib.request.Request(
        _validated_archive_url(),
        headers={"User-Agent": "AudioForge evaluation corpus fetcher/1"},
    )
    with (
        urllib.request.urlopen(request, timeout=60) as response,
        destination.open("wb") as output,
    ):
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def _parse_member(path: str) -> dict[str, str] | None:
    name = Path(path).name
    if not name.lower().endswith(".wav"):
        return None
    parts = name.removesuffix(".wav").split("-")
    if len(parts) != 7:
        return None
    modality, channel, emotion, intensity, statement, repetition, actor = parts
    if (
        modality != "03"
        or channel != "01"
        or (emotion, intensity) not in EMOTIONS_AND_INTENSITIES
        or statement not in STATEMENTS
        or repetition not in {"01", "02"}
    ):
        return None
    return {
        "name": name,
        "statement": statement,
        "repetition": repetition,
        "actor": actor,
        "emotion": emotion,
        "intensity": intensity,
    }


def _normalize_wav(
    payload: bytes,
    name: str,
) -> tuple[int, int, float, bytes, int]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Chunk \(non-data\) not understood, skipping it\.",
        )
        sample_rate, raw = wavfile.read(io.BytesIO(payload))
    audio = np.asarray(raw)
    if int(sample_rate) != 48_000:
        raise ValueError(f"{name} is {sample_rate} Hz, expected native 48000 Hz")
    if audio.dtype != np.dtype("<i2") or audio.ndim not in {1, 2}:
        raise ValueError(f"{name} must be 16-bit PCM, got {audio.shape}/{audio.dtype}")
    source_channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    if source_channels not in {1, 2}:
        raise ValueError(f"{name} has unsupported {source_channels}-channel audio")
    if audio.ndim == 2:
        mono = np.asarray(
            np.clip(
                np.rint(np.mean(audio.astype(np.float64), axis=1)),
                -32768,
                32767,
            ),
            dtype=np.int16,
        )
    else:
        mono = np.asarray(audio, dtype=np.int16)
    normalized = io.BytesIO()
    wavfile.write(normalized, int(sample_rate), mono)
    peak = float(np.max(np.abs(mono.astype(np.float64)))) if mono.size else 0.0
    return (
        int(sample_rate),
        int(mono.size),
        peak / 32768.0,
        normalized.getvalue(),
        source_channels,
    )


def extract(
    archive_path: Path,
    output_root: Path,
    *,
    actors: Iterable[int] = range(1, 25),
) -> dict[str, Any]:
    selected_actors = {f"{actor:02d}" for actor in actors}
    if not selected_actors:
        raise ValueError("at least one RAVDESS actor is required")
    output_root = output_root.resolve()
    audio_root = output_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    captures: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.namelist()):
            parsed = _parse_member(member)
            if parsed is None or parsed["actor"] not in selected_actors:
                continue
            payload = archive.read(member)
            (
                sample_rate,
                frames,
                peak,
                normalized_payload,
                source_channels,
            ) = _normalize_wav(payload, parsed["name"])
            destination = (
                audio_root
                / f"actor-{parsed['actor']}"
                / parsed["name"]
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(normalized_payload)
            key = (
                parsed["actor"],
                parsed["statement"],
                parsed["emotion"],
                parsed["intensity"],
            )
            pair = captures.setdefault(
                key,
                {
                    "id": (
                        f"actor-{parsed['actor']}-statement-{parsed['statement']}-"
                        f"{EMOTIONS_AND_INTENSITIES[(parsed['emotion'], parsed['intensity'])]}"
                    ),
                    "speaker": f"actor-{parsed['actor']}",
                    "speaker_sex": (
                        "male" if int(parsed["actor"]) % 2 else "female"
                    ),
                    "statement_id": parsed["statement"],
                    "statement": STATEMENTS[parsed["statement"]],
                    "delivery": EMOTIONS_AND_INTENSITIES[
                        (parsed["emotion"], parsed["intensity"])
                    ],
                    "sample_rate": sample_rate,
                    "takes": {},
                },
            )
            pair["takes"][parsed["repetition"]] = {
                "path": destination.relative_to(output_root).as_posix(),
                "sha256": _sha256(normalized_payload),
                "source_sha256": _sha256(payload),
                "source_channels": source_channels,
                "frames": frames,
                "duration_seconds": frames / sample_rate,
                "peak": peak,
            }

    expected_pairs = (
        len(selected_actors)
        * len(STATEMENTS)
        * len(EMOTIONS_AND_INTENSITIES)
    )
    if len(captures) != expected_pairs or any(
        set(pair["takes"]) != {"01", "02"} for pair in captures.values()
    ):
        raise RuntimeError(
            f"archive yielded {len(captures)}/{expected_pairs} complete repeated-take pairs"
        )
    actor_numbers = sorted(int(actor) for actor in selected_actors)
    if actor_numbers == list(range(1, 25)):
        splits = {
            "train": [f"actor-{actor:02d}" for actor in range(1, 13)],
            "validation": [f"actor-{actor:02d}" for actor in range(13, 19)],
            "test": [f"actor-{actor:02d}" for actor in range(19, 25)],
        }
    else:
        splits = {"fixture": [f"actor-{actor:02d}" for actor in actor_numbers]}
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "RAVDESS neutral speech repeated-take slice",
        "dataset_page": DATASET_PAGE,
        "license": LICENSE,
        "redistribution": (
            "Local non-commercial evaluation only; audio is ignored by Git "
            "and must never enter release assets."
        ),
        "archive": {
            "url": ARCHIVE_URL,
            "official_md5": OFFICIAL_MD5,
            "actual_md5": _hash(archive_path, "md5"),
            "sha256": _hash(archive_path, "sha256"),
        },
        "selection": {
            "rule": (
                "audio-only speech, neutral emotion, normal intensity, both "
                "statements, both repetitions; plus normal/strong calm delivery "
                "for product-length independent aggregates"
            ),
            "speaker_count": len(selected_actors),
            "pair_count": len(captures),
        },
        "speaker_disjoint_splits": splits,
        "pairs": sorted(captures.values(), key=lambda value: value["id"]),
        "limitations": [
            "Actors use short fixed statements rather than the Rainbow Passage.",
            "Professional acted speech does not represent every conversational delivery.",
            "The non-commercial ShareAlike license prevents shipping this audio with AudioForge.",
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def fetch(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="audioforge-ravdess-",
        dir=output_root.parent,
    ) as raw:
        archive = Path(raw) / "Audio_Speech_Actors_01-24.zip"
        _download(archive)
        actual_md5 = _hash(archive, "md5")
        if actual_md5 != OFFICIAL_MD5:
            raise RuntimeError(
                f"RAVDESS archive MD5 mismatch: {actual_md5} != {OFFICIAL_MD5}"
            )
        return extract(archive, output_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = fetch(args.output)
    print(
        f"Fetched {len(manifest['pairs'])} native-48-kHz repeated-take pairs "
        f"to {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
