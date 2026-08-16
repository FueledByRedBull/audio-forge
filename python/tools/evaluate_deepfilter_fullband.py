"""Revalidate DeepFilter attenuation on native 48 kHz full-band material."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from release_provenance import sha256_file as _sha256

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, welch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "models" / "deepfilter_fullband_eval"
DEFAULT_MIC_NOISE = DEFAULT_CORPUS / "microphone-noise.wav"
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "deepfilter_benchmark.exe"
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "deepfilter-fullband-report.json"
SAMPLE_RATE = 48_000
FRAME_SIZE = 480
MODELS = ("ll", "standard")
ATTENUATION_ARMS_DB = (12.0, 20.0, 30.0, 80.0)
PRODUCT_STRENGTHS = (0.5, 1.0)
HISS_BANDS_HZ = ((8_000.0, 12_000.0), (12_000.0, 16_000.0), (16_000.0, 20_000.0))
ABSOLUTE_GATES = {
    "min_clean_p10_si_sdr_db": 20.0,
    "max_clean_p90_lsd_db": 3.0,
    "max_clean_p90_hf_lsd_db": 6.0,
    "max_clean_dropout_rate": 0.02,
    "min_environmental_median_improvement_db_full": 2.0,
    "min_environmental_median_improvement_db_half": 0.75,
    "min_environmental_p10_improvement_db": -1.0,
    "min_hiss_median_improvement_db": 0.0,
    "min_hiss_p10_improvement_db": -1.0,
    "min_noise_only_attenuation_db_full": 3.0,
    "min_noise_only_attenuation_db_half": 1.0,
    "max_p99_frame_seconds": 0.008,
    "max_clipped_samples": 0,
    "max_non_finite_samples": 0,
}
RELATIVE_TO_80_DB_GATES = {
    "max_clean_p10_si_sdr_regression_db": 3.0,
    "max_clean_p90_hf_lsd_regression_db": 0.5,
    "max_environmental_median_improvement_regression_db": 0.5,
    "max_hiss_median_improvement_regression_db": 0.5,
    "max_noise_only_attenuation_regression_db": 3.0,
}
RELATIVE_DECISION_CHECKS = frozenset(
    {"clean_lower_tail", "clean_hf_lsd", "environmental_median", "hiss_median"}
)


@dataclass(frozen=True)
class Segment:
    identifier: str
    condition: str
    start: int
    end: int
    reference: np.ndarray | None
    vad_probabilities: np.ndarray | None
    metadata: dict[str, Any]


def _read_mono(path: Path) -> np.ndarray:
    sample_rate, raw = wavfile.read(path)
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"{path} is {sample_rate} Hz, expected native 48000 Hz")
    audio = np.asarray(raw)
    if audio.ndim != 1:
        raise ValueError(f"{path} must be mono, got shape {audio.shape}")
    if np.issubdtype(audio.dtype, np.integer):
        bits = audio.dtype.itemsize * 8
        scale = float(2 ** (bits - 1))
        if np.issubdtype(audio.dtype, np.unsignedinteger):
            converted = (np.asarray(audio, dtype=np.float64) - scale) / scale
        else:
            converted = np.asarray(audio, dtype=np.float64) / scale
    else:
        converted = np.asarray(audio, dtype=np.float64)
    if converted.size == 0 or not np.all(np.isfinite(converted)):
        raise ValueError(f"{path} must contain finite audio")
    return converted


def _manifest_audio(
    corpus_root: Path,
    metadata: Any,
    identifier: str,
    expected_frames: Any,
) -> np.ndarray:
    if not isinstance(metadata, dict):
        raise ValueError(f"{identifier} manifest entry must be an object")
    raw_path = metadata.get("path")
    expected_hash = metadata.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{identifier} manifest path is invalid")
    path = (corpus_root / raw_path).resolve(strict=True)
    if not path.is_relative_to(corpus_root):
        raise ValueError(f"{identifier} manifest path escapes the corpus")
    if not isinstance(expected_hash, str) or _sha256(path) != expected_hash.lower():
        raise ValueError(f"{identifier} source hash mismatch")
    audio = _read_mono(path)
    if (
        isinstance(expected_frames, bool)
        or not isinstance(expected_frames, int)
        or expected_frames != audio.size
    ):
        raise ValueError(f"{identifier} frame count mismatch")
    return audio


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-15))


def _db_ratio(numerator: float, denominator: float) -> float:
    return 20.0 * math.log10(max(numerator, 1e-15) / max(denominator, 1e-15))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64, copy=False) - np.mean(reference)
    estimate = estimate.astype(np.float64, copy=False) - np.mean(estimate)
    energy = float(np.dot(reference, reference))
    if energy <= 1e-12:
        return 0.0
    scale = float(np.dot(estimate, reference)) / energy
    target = scale * reference
    residual = estimate - target
    return 10.0 * math.log10(
        (float(np.dot(target, target)) + 1e-12)
        / (float(np.dot(residual, residual)) + 1e-12)
    )


def _frame_rms(audio: np.ndarray) -> np.ndarray:
    if audio.size < 960:
        return np.empty(0, dtype=np.float64)
    frames = np.lib.stride_tricks.sliding_window_view(audio, 960)[::480]
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)


def _speech_active(
    reference: np.ndarray,
    vad_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    levels = _frame_rms(reference)
    if levels.size == 0:
        return np.zeros(0, dtype=bool)
    if vad_probabilities is not None and vad_probabilities.size:
        # Frame RMS uses 20 ms windows at a 10 ms hop. Silero produces one
        # posterior per non-overlapping 32 ms native-input window. Interpolate
        # between posterior centres using the same temporal convention as the
        # product analysis path instead of treating quiet phonetic gaps and
        # utterance tails as speech dropouts.
        frame_centres = np.arange(levels.size, dtype=np.int64) * 480 + 480
        posterior_centres = (
            np.arange(vad_probabilities.size, dtype=np.float64) + 0.5
        ) * 1_536
        interpolated = np.interp(
            frame_centres,
            posterior_centres,
            np.clip(vad_probabilities, 0.0, 1.0),
            left=float(np.clip(vad_probabilities[0], 0.0, 1.0)),
            right=float(np.clip(vad_probabilities[-1], 0.0, 1.0)),
        )
        return np.asarray(interpolated >= 0.48, dtype=bool)
    peak_db = 20.0 * math.log10(max(float(np.percentile(levels, 95)), 1e-9))
    return 20.0 * np.log10(np.maximum(levels, 1e-9)) >= max(
        -50.0, peak_db - 35.0
    )


def _dropout_rate(
    reference: np.ndarray,
    estimate: np.ndarray,
    vad_probabilities: np.ndarray | None = None,
) -> float:
    reference_rms = _frame_rms(reference)
    estimate_rms = _frame_rms(estimate)
    count = min(reference_rms.size, estimate_rms.size)
    active = _speech_active(reference, vad_probabilities)[:count]
    if not np.any(active):
        return 0.0
    ratio = estimate_rms[:count][active] / np.maximum(
        reference_rms[:count][active], 1e-9
    )
    return float(np.mean(ratio < 0.1))


def _speech_lsd(
    reference: np.ndarray,
    estimate: np.ndarray,
    low_hz: float,
    high_hz: float,
    vad_probabilities: np.ndarray | None = None,
) -> float:
    frequencies, _, reference_stft = stft(
        reference,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=960,
        noverlap=480,
        boundary=cast(str, None),
        padded=False,
    )
    _, _, estimate_stft = stft(
        estimate,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=960,
        noverlap=480,
        boundary=cast(str, None),
        padded=False,
    )
    frame_count = min(reference_stft.shape[1], estimate_stft.shape[1])
    band = (frequencies >= low_hz) & (frequencies <= high_hz)
    if frame_count == 0 or not np.any(band):
        return 0.0
    reference_magnitude = np.abs(reference_stft[band, :frame_count])
    estimate_magnitude = np.abs(estimate_stft[band, :frame_count])
    active = _speech_active(reference, vad_probabilities)[:frame_count]
    band_level = np.sqrt(np.mean(np.square(reference_magnitude), axis=0) + 1e-15)
    informative = band_level >= max(float(np.percentile(band_level, 95)) * 0.01, 1e-9)
    selected = active & informative
    if not np.any(selected):
        return 0.0
    floor = max(float(np.max(reference_magnitude)) * 1e-4, 1e-9)
    reference_db = 20.0 * np.log10(
        np.maximum(reference_magnitude[:, selected], floor)
    )
    estimate_db = 20.0 * np.log10(
        np.maximum(estimate_magnitude[:, selected], floor)
    )
    per_frame = np.sqrt(np.mean(np.square(estimate_db - reference_db), axis=0))
    return float(np.median(per_frame))


def _band_rms(audio: np.ndarray, low_hz: float, high_hz: float) -> float:
    if audio.size < 2_048:
        return 0.0
    frequencies, density = welch(
        audio,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=min(4_096, audio.size),
        noverlap=min(2_048, max(0, audio.size // 2)),
        scaling="density",
    )
    band = (frequencies >= low_hz) & (frequencies <= high_hz)
    if np.count_nonzero(band) < 2:
        return 0.0
    return float(np.sqrt(max(float(np.trapezoid(density[band], frequencies[band])), 0.0)))


def _band_noise(
    frames: int,
    low_hz: float,
    high_hz: float,
    seed: int,
) -> np.ndarray:
    frequencies = np.fft.rfftfreq(frames, d=1.0 / SAMPLE_RATE)
    selected = (frequencies >= low_hz) & (frequencies <= high_hz)
    rng = np.random.default_rng(seed)
    spectrum = np.zeros(frequencies.size, dtype=np.complex128)
    spectrum[selected] = rng.standard_normal(
        np.count_nonzero(selected)
    ) + 1j * rng.standard_normal(np.count_nonzero(selected))
    noise = np.fft.irfft(spectrum, n=frames)
    return noise / max(_rms(noise), 1e-15)


def _mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    scaled_noise = noise * (
        _rms(clean) / max(_rms(noise), 1e-15) / (10.0 ** (snr_db / 20.0))
    )
    mixture = clean + scaled_noise
    scale = min(1.0, 0.98 / max(float(np.max(np.abs(mixture))), 1e-12))
    return clean * scale, mixture * scale


def _build_stream(
    corpus_root: Path,
    microphone_noise_path: Path,
) -> tuple[np.ndarray, list[Segment], dict[str, Any]]:
    manifest = json.loads(
        (corpus_root / "manifest.json").read_text(encoding="utf-8")
    )
    captures = manifest.get("captures")
    if not isinstance(captures, list) or len(captures) < 12:
        raise RuntimeError("full-band manifest must contain at least 12 paired captures")
    separator = np.zeros(SAMPLE_RATE // 4, dtype=np.float64)
    parts: list[np.ndarray] = []
    segments: list[Segment] = []
    cursor = 0

    def append(
        identifier: str,
        condition: str,
        audio: np.ndarray,
        reference: np.ndarray | None,
        vad_probabilities: np.ndarray | None,
        metadata: dict[str, Any],
    ) -> None:
        nonlocal cursor
        start = cursor
        end = start + audio.size
        parts.extend((audio, separator))
        segments.append(
            Segment(
                identifier,
                condition,
                start,
                end,
                reference,
                vad_probabilities,
                metadata,
            )
        )
        cursor = end + separator.size

    from mic_eq import analyze_vad_probabilities

    clean_by_id: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    seen_identifiers: set[str] = set()
    for row in captures:
        if not isinstance(row, dict):
            raise ValueError("full-band capture entries must be objects")
        identifier = str(row["id"])
        if identifier in seen_identifiers:
            raise ValueError(f"duplicate full-band capture identifier {identifier}")
        seen_identifiers.add(identifier)
        if row.get("sample_rate") != SAMPLE_RATE:
            raise ValueError(f"{identifier} manifest sample rate is not 48 kHz")
        clean = _manifest_audio(
            corpus_root,
            row.get("clean"),
            f"{identifier} clean",
            row.get("frames"),
        )
        noisy = _manifest_audio(
            corpus_root,
            row.get("noisy"),
            f"{identifier} noisy",
            row.get("frames"),
        )
        if clean.size != noisy.size:
            raise RuntimeError(f"{identifier} clean/noisy length mismatch")
        clean_vad = np.asarray(
            analyze_vad_probabilities(
                np.ascontiguousarray(clean, dtype=np.float32),
                SAMPLE_RATE,
                0.48,
            ),
            dtype=np.float64,
        )
        clean_by_id[identifier] = (clean, clean_vad)
        append(
            identifier + "-clean",
            "clean",
            clean,
            clean,
            clean_vad,
            {"source": identifier},
        )
        append(
            identifier + "-environmental",
            "environmental",
            noisy,
            clean,
            clean_vad,
            {"source": identifier},
        )

    hiss_sources = [
        str(capture["id"])
        for capture in captures
        if str(capture["id"]).split("_", 1)[1] in {"001", "002", "003"}
    ]
    for source_index, identifier in enumerate(hiss_sources):
        clean, clean_vad = clean_by_id[identifier]
        for band_index, (low_hz, high_hz) in enumerate(HISS_BANDS_HZ):
            noise = _band_noise(
                clean.size,
                low_hz,
                high_hz,
                0xD33F_0000 + source_index * 10 + band_index,
            )
            reference, mixture = _mix_at_snr(clean, noise, 10.0)
            append(
                f"{identifier}-hiss-{int(low_hz)}-{int(high_hz)}",
                "hiss",
                mixture,
                reference,
                clean_vad,
                {
                    "source": identifier,
                    "noise_band_hz": [low_hz, high_hz],
                    "snr_db": 10.0,
                },
            )

    for band_index, (low_hz, high_hz) in enumerate(HISS_BANDS_HZ):
        noise = 0.05 * _band_noise(
            SAMPLE_RATE * 8,
            low_hz,
            high_hz,
            0xD33F_1000 + band_index,
        )
        append(
            f"noise-only-{int(low_hz)}-{int(high_hz)}",
            "noise_only",
            noise,
            None,
            None,
            {"noise_band_hz": [low_hz, high_hz]},
        )

    microphone_noise = _read_mono(microphone_noise_path)
    microphone_sidecar = microphone_noise_path.with_suffix(".json")
    microphone_metadata = json.loads(
        microphone_sidecar.resolve(strict=True).read_text(encoding="utf-8")
    )
    if (
        not isinstance(microphone_metadata, dict)
        or microphone_metadata.get("schema_version") != 1
        or microphone_metadata.get("sha256") != _sha256(microphone_noise_path)
        or microphone_metadata.get("sample_rate") != SAMPLE_RATE
        or microphone_metadata.get("frames") != microphone_noise.size
        or microphone_metadata.get("assessment", {}).get("passed") is not True
    ):
        raise ValueError("microphone-noise sidecar does not match the accepted WAV")
    microphone_noise = microphone_noise[: min(microphone_noise.size, SAMPLE_RATE * 30)]
    if microphone_noise.size < SAMPLE_RATE * 10:
        raise RuntimeError("microphone-noise capture must contain at least 10 seconds")
    append(
        "real-microphone-noise",
        "microphone_noise",
        microphone_noise,
        None,
        None,
        {"source": "local physical microphone quiet capture"},
    )
    return (
        np.concatenate(parts).astype(np.float32),
        segments,
        manifest,
    )


def _write_raw(path: Path, audio: np.ndarray) -> None:
    np.asarray(audio, dtype="<f4").tofile(path)


def _run_backend(
    binary: Path,
    input_path: Path,
    output_path: Path,
    model: str,
    attenuation_db: float,
    strength: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    completed = subprocess.run(
        [
            str(binary),
            "--model",
            model,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--library",
            str(REPO_ROOT / "df.dll"),
            "--model-dir",
            str(REPO_ROOT / "models"),
            "--attenuation-db",
            str(attenuation_db),
            "--post-filter-beta",
            "0",
            "--strength",
            str(strength),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("DeepFilter benchmark emitted no JSON metadata")
    return (
        np.fromfile(output_path, dtype="<f4").astype(np.float64),
        json.loads(lines[-1]),
    )


def _speech_metrics(
    reference: np.ndarray,
    input_audio: np.ndarray,
    output: np.ndarray,
    vad_probabilities: np.ndarray | None,
) -> dict[str, float]:
    input_si_sdr = _si_sdr(reference, input_audio)
    output_si_sdr = _si_sdr(reference, output)
    return {
        "input_si_sdr_db": input_si_sdr,
        "output_si_sdr_db": output_si_sdr,
        "si_sdr_improvement_db": output_si_sdr - input_si_sdr,
        "speech_lsd_db": _speech_lsd(
            reference,
            output,
            80.0,
            20_000.0,
            vad_probabilities,
        ),
        "hf_speech_lsd_db": _speech_lsd(
            reference,
            output,
            8_000.0,
            20_000.0,
            vad_probabilities,
        ),
        "dropout_rate": _dropout_rate(
            reference,
            output,
            vad_probabilities,
        ),
        "vad_active_frame_count": float(
            np.count_nonzero(
                _speech_active(reference, vad_probabilities)
            )
        ),
    }


def _segment_metrics(
    stream: np.ndarray,
    aligned_output: np.ndarray,
    segments: list[Segment],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        input_audio = stream[segment.start : segment.end].astype(np.float64)
        output = aligned_output[segment.start : segment.end]
        row: dict[str, Any] = {
            "id": segment.identifier,
            "condition": segment.condition,
            "metadata": segment.metadata,
            "input_peak_dbfs": _db_ratio(
                float(np.max(np.abs(input_audio))), 1.0
            ),
            "output_peak_dbfs": _db_ratio(float(np.max(np.abs(output))), 1.0),
            "clipped_samples": int(np.count_nonzero(np.abs(output) > 1.0)),
            "non_finite_samples": int(np.count_nonzero(~np.isfinite(output))),
        }
        if segment.reference is not None:
            row.update(
                _speech_metrics(
                    segment.reference,
                    input_audio,
                    output,
                    segment.vad_probabilities,
                )
            )
        else:
            row["total_attenuation_db"] = -_db_ratio(_rms(output), _rms(input_audio))
            row["band_attenuation_db"] = {
                f"{int(low_hz)}-{int(high_hz)}": -_db_ratio(
                    _band_rms(output, low_hz, high_hz),
                    _band_rms(input_audio, low_hz, high_hz),
                )
                for low_hz, high_hz in (
                    (80.0, 8_000.0),
                    *HISS_BANDS_HZ,
                )
            }
            target_band = segment.metadata.get("noise_band_hz")
            if isinstance(target_band, list) and len(target_band) == 2:
                row["target_band_attenuation_db"] = -_db_ratio(
                    _band_rms(output, float(target_band[0]), float(target_band[1])),
                    _band_rms(
                        input_audio, float(target_band[0]), float(target_band[1])
                    ),
                )
            else:
                row["target_band_attenuation_db"] = row["total_attenuation_db"]
        rows.append(row)
    return rows


def _aggregate(rows: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, float]:
    clean = [row for row in rows if row["condition"] == "clean"]
    environmental = [
        row for row in rows if row["condition"] == "environmental"
    ]
    hiss = [row for row in rows if row["condition"] == "hiss"]
    noise_only = [
        row
        for row in rows
        if row["condition"] in {"noise_only", "microphone_noise"}
    ]
    return {
        "clean_p10_si_sdr_db": _percentile(
            [float(row["output_si_sdr_db"]) for row in clean], 10
        ),
        "clean_p90_lsd_db": _percentile(
            [float(row["speech_lsd_db"]) for row in clean], 90
        ),
        "clean_p90_hf_lsd_db": _percentile(
            [float(row["hf_speech_lsd_db"]) for row in clean], 90
        ),
        "clean_max_dropout_rate": max(
            (float(row["dropout_rate"]) for row in clean), default=0.0
        ),
        "environmental_median_improvement_db": _percentile(
            [float(row["si_sdr_improvement_db"]) for row in environmental], 50
        ),
        "environmental_p10_improvement_db": _percentile(
            [float(row["si_sdr_improvement_db"]) for row in environmental], 10
        ),
        "hiss_median_improvement_db": _percentile(
            [float(row["si_sdr_improvement_db"]) for row in hiss], 50
        ),
        "hiss_p10_improvement_db": _percentile(
            [float(row["si_sdr_improvement_db"]) for row in hiss], 10
        ),
        "noise_only_median_attenuation_db": _percentile(
            [float(row["target_band_attenuation_db"]) for row in noise_only], 50
        ),
        "clipped_samples": float(
            sum(int(row["clipped_samples"]) for row in rows)
        ),
        "non_finite_samples": float(
            sum(int(row["non_finite_samples"]) for row in rows)
        ),
        "rtf": float(runtime["rtf"]),
        "p99_frame_seconds": float(runtime["p99_frame_seconds"]),
        "max_frame_seconds": float(runtime["max_frame_seconds"]),
    }


def _absolute_checks(aggregate: dict[str, float], strength: float) -> dict[str, bool]:
    minimum_environmental = (
        ABSOLUTE_GATES["min_environmental_median_improvement_db_full"]
        if strength == 1.0
        else ABSOLUTE_GATES["min_environmental_median_improvement_db_half"]
    )
    minimum_noise_attenuation = (
        ABSOLUTE_GATES["min_noise_only_attenuation_db_full"]
        if strength == 1.0
        else ABSOLUTE_GATES["min_noise_only_attenuation_db_half"]
    )
    return {
        "clean_lower_tail": aggregate["clean_p10_si_sdr_db"]
        >= ABSOLUTE_GATES["min_clean_p10_si_sdr_db"],
        "clean_lsd": aggregate["clean_p90_lsd_db"]
        <= ABSOLUTE_GATES["max_clean_p90_lsd_db"],
        "clean_hf_lsd": aggregate["clean_p90_hf_lsd_db"]
        <= ABSOLUTE_GATES["max_clean_p90_hf_lsd_db"],
        "clean_dropout": aggregate["clean_max_dropout_rate"]
        <= ABSOLUTE_GATES["max_clean_dropout_rate"],
        "environmental_median": aggregate["environmental_median_improvement_db"]
        >= minimum_environmental,
        "environmental_lower_tail": aggregate["environmental_p10_improvement_db"]
        >= ABSOLUTE_GATES["min_environmental_p10_improvement_db"],
        "hiss_median": aggregate["hiss_median_improvement_db"]
        >= ABSOLUTE_GATES["min_hiss_median_improvement_db"],
        "hiss_lower_tail": aggregate["hiss_p10_improvement_db"]
        >= ABSOLUTE_GATES["min_hiss_p10_improvement_db"],
        "noise_only_attenuation": aggregate["noise_only_median_attenuation_db"]
        >= minimum_noise_attenuation,
        "p99_realtime": aggregate["p99_frame_seconds"]
        <= ABSOLUTE_GATES["max_p99_frame_seconds"],
        "clipping": aggregate["clipped_samples"]
        <= ABSOLUTE_GATES["max_clipped_samples"],
        "finite": aggregate["non_finite_samples"]
        <= ABSOLUTE_GATES["max_non_finite_samples"],
    }


def _relative_checks(
    aggregate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, bool]:
    return {
        "clean_lower_tail": aggregate["clean_p10_si_sdr_db"]
        >= baseline["clean_p10_si_sdr_db"]
        - RELATIVE_TO_80_DB_GATES["max_clean_p10_si_sdr_regression_db"],
        "clean_hf_lsd": aggregate["clean_p90_hf_lsd_db"]
        <= baseline["clean_p90_hf_lsd_db"]
        + RELATIVE_TO_80_DB_GATES["max_clean_p90_hf_lsd_regression_db"],
        "environmental_median": aggregate["environmental_median_improvement_db"]
        >= baseline["environmental_median_improvement_db"]
        - RELATIVE_TO_80_DB_GATES[
            "max_environmental_median_improvement_regression_db"
        ],
        "hiss_median": aggregate["hiss_median_improvement_db"]
        >= baseline["hiss_median_improvement_db"]
        - RELATIVE_TO_80_DB_GATES["max_hiss_median_improvement_regression_db"],
        "noise_only_attenuation": aggregate["noise_only_median_attenuation_db"]
        >= baseline["noise_only_median_attenuation_db"]
        - RELATIVE_TO_80_DB_GATES[
            "max_noise_only_attenuation_regression_db"
        ],
    }


def _select_objective_attenuation(decisions: list[dict[str, Any]]) -> float | None:
    if not decisions:
        return None
    common = set(ATTENUATION_ARMS_DB)
    for decision in decisions:
        arms = decision.get("arms")
        if not isinstance(arms, dict):
            return None
        common &= {
            float(raw_attenuation)
            for raw_attenuation, metadata in arms.items()
            if isinstance(metadata, dict) and metadata.get("passed") is True
        }
    return min(common) if common else None


def evaluate(
    corpus_root: Path,
    microphone_noise_path: Path,
    binary: Path,
) -> dict[str, Any]:
    corpus_root = corpus_root.resolve(strict=True)
    microphone_noise_path = microphone_noise_path.resolve(strict=True)
    binary = binary.resolve(strict=True)
    stream, segments, manifest = _build_stream(
        corpus_root, microphone_noise_path
    )
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="audioforge-deepfilter-fullband-") as raw:
        temporary = Path(raw)
        input_path = temporary / "input.f32"
        _write_raw(input_path, stream)
        for model in MODELS:
            for strength in PRODUCT_STRENGTHS:
                for attenuation_db in ATTENUATION_ARMS_DB:
                    output, runtime = _run_backend(
                        binary,
                        input_path,
                        temporary
                        / f"{model}-{strength:g}-{attenuation_db:g}.f32",
                        model,
                        attenuation_db,
                        strength,
                    )
                    latency = int(runtime["latency_samples"])
                    aligned = output[latency : latency + stream.size]
                    if aligned.size != stream.size:
                        raise RuntimeError("DeepFilter output is too short after alignment")
                    rows = _segment_metrics(stream, aligned, segments)
                    aggregate = _aggregate(rows, runtime)
                    cases.append(
                        {
                            "model": model,
                            "strength": strength,
                            "attenuation_limit_db": attenuation_db,
                            "post_filter_beta": 0.0,
                            "declared_latency_samples": latency,
                            "runtime": runtime,
                            "aggregate": aggregate,
                            "segments": rows,
                        }
                    )

    decisions: list[dict[str, Any]] = []
    for model in MODELS:
        for strength in PRODUCT_STRENGTHS:
            stratum = [
                case
                for case in cases
                if case["model"] == model and case["strength"] == strength
            ]
            baseline = next(
                case for case in stratum if case["attenuation_limit_db"] == 80.0
            )
            arms: dict[str, Any] = {}
            passing: list[float] = []
            for case in stratum:
                absolute = _absolute_checks(case["aggregate"], strength)
                relative = _relative_checks(
                    case["aggregate"], baseline["aggregate"]
                )
                decision_relative = {
                    name: passed
                    for name, passed in relative.items()
                    if name in RELATIVE_DECISION_CHECKS
                }
                passed = all(absolute.values()) and all(decision_relative.values())
                arms[str(case["attenuation_limit_db"])] = {
                    "metrics": case["aggregate"],
                    "absolute_checks": absolute,
                    "relative_to_80_db_checks": relative,
                    "release_decision_relative_checks": decision_relative,
                    "passed": passed,
                }
                if passed:
                    passing.append(float(case["attenuation_limit_db"]))
            decisions.append(
                {
                    "model": model,
                    "strength": strength,
                    "arms": arms,
                    "objective_selected_attenuation_limit_db": (
                        min(passing) if passing else None
                    ),
                }
            )

    objective_selected = _select_objective_attenuation(decisions)
    current_30_passed = all(
        bool(decision["arms"]["30.0"]["passed"]) for decision in decisions
    )
    release_selected = objective_selected if objective_selected is not None else 30.0
    if objective_selected is None:
        decision_reason = "No attenuation arm passed every objective stratum."
    elif objective_selected == 30.0:
        decision_reason = (
            "The 30 dB default is the least aggressive arm that passes every "
            "absolute and speech-present benchmark stratum. Extra noise-only "
            "attenuation is reported diagnostically but cannot alone justify a "
            "more aggressive production default."
        )
    else:
        decision_reason = (
            f"{objective_selected:g} dB is the least aggressive arm that passes "
            "every absolute and speech-present benchmark stratum."
        )
    tracked_source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "rust-core/src/bin/deepfilter_benchmark.rs",
        REPO_ROOT / "rust-core/src/dsp/deepfilter_ffi.rs",
    )
    source_paths = (
        *tracked_source_paths,
        binary,
        corpus_root / "manifest.json",
        microphone_noise_path,
        microphone_noise_path.with_suffix(".json"),
        REPO_ROOT / "df.dll",
        REPO_ROOT / "models/DeepFilterNet3_ll_onnx.tar.gz",
        REPO_ROOT / "models/DeepFilterNet3_onnx.tar.gz",
        REPO_ROOT / "models/silero_vad.onnx",
    )
    source_hashes = {
        (
            path.relative_to(REPO_ROOT).as_posix()
            if path.is_relative_to(REPO_ROOT)
            else path.name
        ): _sha256(path)
        for path in source_paths
    }
    tracked_source_hashes = {
        path.relative_to(REPO_ROOT).as_posix(): _sha256(path)
        for path in tracked_source_paths
    }
    maximum_p99 = max(
        float(case["aggregate"]["p99_frame_seconds"]) for case in cases
    )
    report = {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audible_change": True,
        "status": (
            "objective-failed"
            if objective_selected is None
            else "objective-retained-incumbent"
            if objective_selected == 30.0
            else "objective-challenger-adopted"
        ),
        "decision": {
            "current_30_db_passed_objective_gates": current_30_passed,
            "objective_selected_attenuation_limit_db": objective_selected,
            "release_selected_attenuation_limit_db": release_selected,
            "objective_candidate_adopted": bool(
                objective_selected is not None and objective_selected != 30.0
            ),
            "retained": objective_selected == 30.0,
            "reason": decision_reason,
        },
        "configuration": {
            "sample_rate": SAMPLE_RATE,
            "models": list(MODELS),
            "attenuation_arms_db": list(ATTENUATION_ARMS_DB),
            "product_strengths": list(PRODUCT_STRENGTHS),
            "post_filter_beta": 0.0,
            "hiss_bands_hz": [list(band) for band in HISS_BANDS_HZ],
            "corpus_pairs": len(manifest["captures"]),
            "stream_seconds": stream.size / SAMPLE_RATE,
            "dropout_activity_definition": (
                "Silero posterior linearly interpolated between 32 ms native "
                "window centres and thresholded at 0.48 on 20 ms RMS-frame centres"
            ),
        },
        "metric_revision": {
            "replaces": "energy-only peak-minus-35-dB activity mask",
            "reason": (
                "The preliminary report counted quiet phonetic gaps and "
                "utterance tails as active speech, inflating dropout. The "
                "retained metric uses the product VAD posterior for event scope."
            ),
            "runtime_tail_policy": (
                "P99 frame time and whole-stream RTF are release gates. The hard "
                "maximum is retained as a diagnostic because a single value across "
                "roughly 32,000 frames is dominated by host scheduling and grows "
                "less stable as the benchmark corpus grows; exact-artifact 30-minute "
                "underrun/drop/recovery checks provide the sustained realtime gate."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "gates": {
            "absolute": ABSOLUTE_GATES,
            "relative_to_80_db": RELATIVE_TO_80_DB_GATES,
            "release_decision_relative_checks": sorted(RELATIVE_DECISION_CHECKS),
            "noise_only_attenuation_policy": (
                "diagnostic only; it cannot independently select a more aggressive default"
            ),
        },
        "decisions_by_model_and_strength": decisions,
        "cases": cases,
        "source_sha256": tracked_source_hashes,
        "evaluation_contract": {
            "configuration": {
                "attenuation_limit_db": release_selected,
                "post_filter_beta": 0.0,
                "strengths": list(PRODUCT_STRENGTHS),
            },
            "asset_hashes": source_hashes,
            "runtime": {"max_p99_frame_seconds": maximum_p99},
            "latency": {
                f"{model}_{strength:g}": next(
                    int(case["declared_latency_samples"])
                    for case in cases
                    if case["model"] == model
                    and case["strength"] == strength
                    and case["attenuation_limit_db"] == 30.0
                )
                for model in MODELS
                for strength in PRODUCT_STRENGTHS
            },
            "clean_preservation": {
                "metric": "per-stratum lower-tail SI-SDR, P90 full/HF LSD, and dropout",
                "passed": current_30_passed,
            },
        },
        "limitations": [
            "Two official VoiceBank-DEMAND test speakers do not represent every voice.",
            "The local quiet capture contains microphone, preamp, and room noise; it is not a laboratory-isolated self-noise measurement.",
            "Noise-only attenuation is retained as a diagnostic rather than a release selector because stronger silence suppression alone does not establish better speech-present output.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--microphone-noise", type=Path, default=DEFAULT_MIC_NOISE)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--details-output",
        type=Path,
        help="Optional full per-case report; the tracked report stays compact.",
    )
    args = parser.parse_args()
    report = evaluate(
        args.corpus,
        args.microphone_noise,
        args.binary,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.details_output is not None:
        details_output = args.details_output.resolve()
        details_output.parent.mkdir(parents=True, exist_ok=True)
        details_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    report.pop("cases", None)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"DeepFilter full-band evaluation status={report['status']} "
        f"objective_selected={report['decision']['objective_selected_attenuation_limit_db']} dB"
    )
    return 0 if report["status"] != "objective-failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
