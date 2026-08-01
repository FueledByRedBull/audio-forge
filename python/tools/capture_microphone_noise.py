"""Capture and validate a quiet native-48-kHz physical-microphone noise sample."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.io import wavfile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT / "models/deepfilter_fullband_eval/microphone-noise.wav"
)
MAX_ACTIVE_FRACTION = 0.02
MAX_VAD_P95 = 0.30
MAX_PEAK_DBFS = -12.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(abs(value), 1e-12))


def assess_capture(
    audio: np.ndarray,
    vad_probabilities: Sequence[float],
) -> dict[str, Any]:
    probabilities = np.asarray(vad_probabilities, dtype=np.float64)
    peak_dbfs = _dbfs(float(np.max(np.abs(audio))))
    rms_dbfs = _dbfs(
        float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-15))
    )
    active_fraction = (
        float(np.mean(probabilities >= 0.48)) if probabilities.size else 1.0
    )
    vad_p95 = float(np.percentile(probabilities, 95)) if probabilities.size else 1.0
    checks = {
        "finite": bool(np.all(np.isfinite(audio))),
        "duration": audio.size >= 10 * 48_000,
        "speech_activity": active_fraction <= MAX_ACTIVE_FRACTION,
        "vad_p95": vad_p95 <= MAX_VAD_P95,
        "peak": peak_dbfs <= MAX_PEAK_DBFS,
        "non_silent": rms_dbfs >= -120.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "vad_active_fraction": active_fraction,
        "vad_p95": vad_p95,
    }


def capture(
    processor: Any,
    analyze_vad: Callable[[np.ndarray, int, float], Sequence[float]],
    *,
    duration_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    processor.set_output_mute(True)
    processor.start_raw_recording(duration_seconds)
    deadline = time.monotonic() + duration_seconds + 5.0
    while not processor.is_recording_complete():
        if time.monotonic() >= deadline:
            raise TimeoutError("microphone-noise recording did not complete")
        time.sleep(0.05)
    audio = np.asarray(processor.stop_raw_recording(), dtype=np.float32)
    probabilities = analyze_vad(audio, 48_000, 0.48)
    return audio, assess_capture(audio, probabilities)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-device", required=True)
    parser.add_argument("--output-device", required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 10.0 <= args.duration <= 30.0:
        raise ValueError("--duration must be between 10 and 30 seconds")

    from mic_eq import AudioProcessor, analyze_vad_probabilities

    processor = AudioProcessor()
    try:
        processor.start(args.input_device, args.output_device)
        if int(processor.sample_rate()) != 48_000:
            raise RuntimeError(
                f"selected microphone runs at {processor.sample_rate()} Hz, expected 48000"
            )
        time.sleep(1.0)
        audio, assessment = capture(
            processor,
            analyze_vad_probabilities,
            duration_seconds=args.duration,
        )
    finally:
        try:
            processor.set_output_mute(False)
        except Exception:
            pass
        processor.stop()

    if not assessment["passed"]:
        print(json.dumps(assessment, indent=2))
        return 1
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output, 48_000, audio)
    native_spec = importlib.util.find_spec("mic_eq.mic_eq_core")
    native_path = (
        Path(native_spec.origin)
        if native_spec is not None and native_spec.origin is not None
        else None
    )
    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_rate": 48_000,
        "frames": int(audio.size),
        "duration_seconds": audio.size / 48_000.0,
        "input_device": args.input_device,
        "output_device": args.output_device,
        "sha256": _sha256(output),
        "assessment": assessment,
        "runtime": (
            {
                "native_extension_name": native_path.name,
                "native_extension_sha256": _sha256(native_path),
            }
            if native_path is not None
            else None
        ),
        "limitation": (
            "A quiet physical-microphone capture includes microphone, preamp, "
            "electrical, and room noise; it is not isolated laboratory self-noise."
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
