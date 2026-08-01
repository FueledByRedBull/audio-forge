"""Fetch a deterministic native-48-kHz VoiceBank-DEMAND evaluation subset."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "models" / "deepfilter_fullband_eval"
DATASET_PAGE = "https://doi.org/10.7488/ds/2117"
LICENSE = "CC BY 4.0"
ARCHIVES = {
    "clean": {
        "url": (
            "https://datashare.ed.ac.uk/bitstreams/"
            "dec213d3-bf57-4777-9663-c24bdce92d5e/download"
        ),
        "official_md5": "34eb1c0ba7ef667e9b966866c542fc16",
        "prefix": "clean_testset_wav",
    },
    "noisy": {
        "url": (
            "https://datashare.ed.ac.uk/bitstreams/"
            "13c1bfbf-14a6-41db-9b41-8f7310f01ad5/download"
        ),
        "official_md5": "fb1b86caa31e8ba5b506c0c64da9aab5",
        "prefix": "noisy_testset_wav",
    },
}
SELECTED_BASENAMES = (
    "p232_001.wav",
    "p232_002.wav",
    "p232_003.wav",
    "p232_005.wav",
    "p232_006.wav",
    "p232_007.wav",
    "p232_009.wav",
    "p232_010.wav",
    "p232_011.wav",
    "p232_012.wav",
    "p232_013.wav",
    "p232_014.wav",
    "p257_001.wav",
    "p257_002.wav",
    "p257_003.wav",
    "p257_004.wav",
    "p257_006.wav",
    "p257_007.wav",
    "p257_008.wav",
    "p257_009.wav",
    "p257_010.wav",
    "p257_011.wav",
    "p257_012.wav",
    "p257_013.wav",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_wav(payload: bytes, label: str) -> tuple[int, int, float]:
    sample_rate, raw = wavfile.read(io.BytesIO(payload))
    audio = np.asarray(raw)
    if int(sample_rate) != 48_000:
        raise ValueError(f"{label} is {sample_rate} Hz, expected native 48000 Hz")
    if audio.ndim != 1:
        raise ValueError(f"{label} must be mono, got shape {audio.shape}")
    if not np.issubdtype(audio.dtype, np.integer):
        raise ValueError(f"{label} must use integer PCM, got {audio.dtype}")
    peak = float(np.max(np.abs(audio.astype(np.float64)))) if audio.size else 0.0
    scale = float(2 ** (audio.dtype.itemsize * 8 - 1))
    return int(sample_rate), int(audio.size), peak / scale


def _member_name(kind: str, basename: str) -> str:
    return f"{ARCHIVES[kind]['prefix']}/{basename}"


def fetch(output_root: Path) -> dict[str, Any]:
    try:
        RemoteZip = importlib.import_module("remotezip").RemoteZip
    except ImportError as error:
        raise RuntimeError(
            "remotezip is required only for corpus acquisition; install remotezip==0.12.3"
        ) from error

    output_root = output_root.resolve()
    clean_root = output_root / "clean"
    noisy_root = output_root / "noisy"
    clean_root.mkdir(parents=True, exist_ok=True)
    noisy_root.mkdir(parents=True, exist_ok=True)
    captures: list[dict[str, Any]] = []

    archives = {
        kind: RemoteZip(str(metadata["url"]))
        for kind, metadata in ARCHIVES.items()
    }
    try:
        available = {
            kind: {entry.filename for entry in archive.infolist()}
            for kind, archive in archives.items()
        }
        for basename in SELECTED_BASENAMES:
            payloads: dict[str, bytes] = {}
            metadata: dict[str, tuple[int, int, float]] = {}
            for kind in ("clean", "noisy"):
                member = _member_name(kind, basename)
                if member not in available[kind]:
                    raise FileNotFoundError(f"{member} is absent from the official archive")
                payload = archives[kind].read(member)
                payloads[kind] = payload
                metadata[kind] = _validate_wav(payload, member)
            if metadata["clean"][:2] != metadata["noisy"][:2]:
                raise ValueError(
                    f"{basename} clean/noisy sample-rate or frame-count mismatch"
                )
            destinations = {
                "clean": clean_root / basename,
                "noisy": noisy_root / basename,
            }
            for kind, destination in destinations.items():
                destination.write_bytes(payloads[kind])
            captures.append(
                {
                    "id": basename.removesuffix(".wav"),
                    "speaker": basename.split("_", 1)[0],
                    "sample_rate": metadata["clean"][0],
                    "frames": metadata["clean"][1],
                    "duration_seconds": metadata["clean"][1] / 48_000.0,
                    "clean": {
                        "path": f"clean/{basename}",
                        "sha256": _sha256(payloads["clean"]),
                        "peak": metadata["clean"][2],
                    },
                    "noisy": {
                        "path": f"noisy/{basename}",
                        "sha256": _sha256(payloads["noisy"]),
                        "peak": metadata["noisy"][2],
                    },
                }
            )
    finally:
        for archive in archives.values():
            archive.close()

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "VoiceBank-DEMAND native 48 kHz test subset",
        "dataset_page": DATASET_PAGE,
        "license": LICENSE,
        "selection": {
            "rule": "first 12 lexicographic utterances from each test speaker",
            "speakers": ["p232", "p257"],
            "pair_count": len(captures),
        },
        "archives": {
            kind: {
                **metadata,
                "archive_hash_status": (
                    "publisher-reported; ranged member retrieval does not download "
                    "the complete archive"
                ),
            }
            for kind, metadata in ARCHIVES.items()
        },
        "captures": captures,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = fetch(args.output)
    print(
        f"Fetched {len(manifest['captures'])} native-48-kHz pairs to "
        f"{args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
