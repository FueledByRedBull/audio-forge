"""Build a deterministic labeled VAD corpus from pinned speech/noise sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from release_provenance import sha256_file as _sha256

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

from mic_eq.analysis.wav_io import read_mono_wav

SPEECH_LICENSE = "CC BY-SA 4.0"
SPEECH_SOURCE = (
    "https://github.com/Jakobovski/free-spoken-digit-dataset/tree/v1.0.10"
)
NOISE_LICENSE = "CC BY-NC 3.0"
NOISE_SOURCE = "https://github.com/karolpiczak/ESC-50/tree/master"
TARGET_SPEECH_RMS_DBFS = -22.0
PADDING_SECONDS = 0.4
SEED = 0xA0D10F0

SPEAKER_SPLITS = {
    "george": "calibration",
    "jackson": "calibration",
    "lucas": "development",
    "nicolas": "development",
    "theo": "held_out",
    "yweweler": "held_out",
}
NOISE_CATEGORIES = {
    "1-100210-A-36.wav": "vacuum_cleaner",
    "1-11687-A-47.wav": "airplane",
    "1-137-A-32.wav": "keyboard_typing",
    "1-17367-A-10.wav": "rain",
    "1-18527-A-44.wav": "engine",
    "1-21934-A-38.wav": "clock_tick",
}
REPO_ROOT = Path(__file__).resolve().parents[2]


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float64)
    divisor = math.gcd(source_rate, target_rate)
    return np.asarray(
        resample_poly(audio, target_rate // divisor, source_rate // divisor),
        dtype=np.float64,
    )


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def _set_rms(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    current = _rms(audio)
    if current <= 1e-12:
        return np.zeros_like(audio)
    target = 10.0 ** (target_dbfs / 20.0)
    return audio * (target / current)


def _limit_peak(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    current = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio * (peak / current) if current > peak else audio


def _deterministic_noise_segment(
    noise: np.ndarray,
    length: int,
    key: str,
) -> np.ndarray:
    if noise.size == 0:
        return np.zeros(length, dtype=np.float64)
    if noise.size < length:
        noise = np.tile(noise, int(np.ceil(length / noise.size)))
    maximum_start = max(noise.size - length, 0)
    offset_seed = int.from_bytes(
        hashlib.sha256(f"{SEED}:{key}".encode()).digest()[:8],
        "little",
    )
    start = offset_seed % (maximum_start + 1)
    segment = np.asarray(noise[start : start + length], dtype=np.float64)
    return segment - float(np.mean(segment))


def _phone_band(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    high_hz = min(3400.0, sample_rate * 0.45)
    sos = butter(6, [300.0, high_hz], btype="bandpass", fs=sample_rate, output="sos")
    return np.asarray(sosfilt(sos, audio), dtype=np.float64)


def _pitch_up(audio: np.ndarray) -> np.ndarray:
    # Speed/pitch transform is intentionally simple and deterministic. It tests
    # high-pitch robustness without introducing a phase-vocoder dependency.
    return np.asarray(resample_poly(audio, 10, 13), dtype=np.float64)


def _write_capture(
    *,
    output_root: Path,
    relative_path: Path,
    sample_rate: int,
    audio: np.ndarray,
    split: str,
    condition: str,
    speech_interval: tuple[int, int] | None,
    source_paths: list[Path],
    captures: list[dict[str, Any]],
) -> None:
    destination = output_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = _limit_peak(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0))
    pcm = np.round(np.clip(rendered, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(destination, sample_rate, pcm)
    captures.append(
        {
            "path": destination.relative_to(output_root).as_posix(),
            "split": split,
            "condition": condition,
            "sample_rate": sample_rate,
            "speech_intervals_samples": (
                [list(speech_interval)] if speech_interval is not None else []
            ),
            "source_paths": [_portable_path(path) for path in source_paths],
            "sha256": _sha256(destination),
        }
    )


def _speech_identity(path: Path) -> tuple[str, str]:
    parts = path.stem.split("_")
    if len(parts) != 3 or parts[1] not in SPEAKER_SPLITS:
        raise ValueError(f"unexpected FSDD filename: {path.name}")
    return parts[1], SPEAKER_SPLITS[parts[1]]


def build_corpus(source_root: Path, output_root: Path) -> Path:
    speech_paths = sorted((source_root / "fsdd").glob("*.wav"))
    noise_paths = sorted((source_root / "esc50").glob("*.wav"))
    if not speech_paths:
        raise FileNotFoundError(f"no FSDD WAV files under {source_root / 'fsdd'}")
    if not noise_paths:
        raise FileNotFoundError(f"no ESC-50 WAV files under {source_root / 'esc50'}")

    noise_sources = []
    for noise_path in noise_paths:
        source_rate, noise = read_mono_wav(noise_path, dtype=np.float64)
        noise_sources.append((noise_path, source_rate, noise))

    captures: list[dict[str, Any]] = []
    source_records = [
        {
            "path": path.relative_to(source_root).as_posix(),
            "sha256": _sha256(path),
            "source": SPEECH_SOURCE,
            "license": SPEECH_LICENSE,
        }
        for path in speech_paths
    ]
    source_records.extend(
        {
            "path": path.relative_to(source_root).as_posix(),
            "sha256": _sha256(path),
            "source": NOISE_SOURCE,
            "license": NOISE_LICENSE,
            "category": NOISE_CATEGORIES.get(path.name, "unknown"),
        }
        for path in noise_paths
    )

    for speech_index, speech_path in enumerate(speech_paths):
        speaker, split = _speech_identity(speech_path)
        source_rate, source_speech = read_mono_wav(speech_path, dtype=np.float64)
        noise_path, noise_rate, source_noise = noise_sources[
            speech_index % len(noise_sources)
        ]

        conditions = [
            ("clean", 16000, None),
            ("clean_48k", 48000, None),
            ("noise_0db", 16000, 0.0),
            ("noise_5db_48k", 48000, 5.0),
            ("noise_10db", 16000, 10.0),
            ("noise_20db_48k", 48000, 20.0),
            ("quiet", 48000, None),
            ("phone_band", 16000, None),
            ("high_pitch", 16000, None),
        ]

        for condition, sample_rate, snr_db in conditions:
            speech = _resample(source_speech, source_rate, sample_rate)
            speech = speech - float(np.mean(speech))
            if condition == "high_pitch":
                speech = _pitch_up(speech)
            if condition == "phone_band":
                speech = _phone_band(speech, sample_rate)
            target_rms = -40.0 if condition == "quiet" else TARGET_SPEECH_RMS_DBFS
            speech = _set_rms(speech, target_rms)

            padding = int(round(PADDING_SECONDS * sample_rate))
            speech_start = padding
            speech_end = speech_start + speech.size
            rendered = np.zeros(speech.size + 2 * padding, dtype=np.float64)
            rendered[speech_start:speech_end] = speech

            sources = [speech_path]
            if snr_db is not None:
                noise = _resample(source_noise, noise_rate, sample_rate)
                noise = _deterministic_noise_segment(
                    noise,
                    rendered.size,
                    f"{speech_path.name}:{condition}:{noise_path.name}",
                )
                speech_rms = _rms(speech)
                noise_rms = _rms(noise)
                if noise_rms > 1e-12:
                    noise *= speech_rms / (10.0 ** (snr_db / 20.0) * noise_rms)
                rendered += noise
                sources.append(noise_path)

            relative_path = (
                Path(split)
                / condition
                / f"{speech_path.stem}_{sample_rate}.wav"
            )
            _write_capture(
                output_root=output_root,
                relative_path=relative_path,
                sample_rate=sample_rate,
                audio=rendered,
                split=split,
                condition=condition,
                speech_interval=(speech_start, speech_end),
                source_paths=sources,
                captures=captures,
            )

    for sample_rate in (16000, 48000):
        for noise_path, noise_rate, source_noise in noise_sources:
            noise = _resample(source_noise, noise_rate, sample_rate)
            noise = _deterministic_noise_segment(
                noise,
                5 * sample_rate,
                f"noise-only:{noise_path.name}:{sample_rate}",
            )
            noise = _set_rms(noise, -30.0)
            category = NOISE_CATEGORIES.get(noise_path.name, "unknown")
            _write_capture(
                output_root=output_root,
                relative_path=Path("held_out")
                / "noise_only"
                / f"{noise_path.stem}_{sample_rate}.wav",
                sample_rate=sample_rate,
                audio=noise,
                split="held_out",
                condition=f"noise_only_{category}",
                speech_interval=None,
                source_paths=[noise_path],
                captures=captures,
            )

    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "description": (
            "Deterministic FSDD speech mixed with ESC-50 environmental noise. "
            "Labels are exact inserted-speech intervals."
        ),
        "licenses": {
            "speech": SPEECH_LICENSE,
            "noise": NOISE_LICENSE,
            "redistribution": (
                "Generated corpus is local benchmark data and is not a release asset."
            ),
        },
        "sources": source_records,
        "captures": captures,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("models/vad_corpus_sources"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/vad_eval_corpus"),
    )
    args = parser.parse_args()

    manifest_path = build_corpus(
        args.sources.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "capture_count": len(payload["captures"]),
                "source_count": len(payload["sources"]),
                "manifest_sha256": _sha256(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
