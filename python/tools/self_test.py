"""
AudioForge self-test (headless).

Plays a probe signal, captures raw input, and verifies correlation.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import wave

import numpy as np

from mic_eq import AudioProcessor
from mic_eq.analysis.latency_calibration import analyze_latency, generate_probe_signal


def _play_probe_blocking(probe: np.ndarray, sample_rate: int) -> None:
    """Play probe signal using platform-available APIs."""
    if os.name == "nt":
        import winsound

        pcm = np.clip(probe, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            with wave.open(wav_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm16.tobytes())
            play_flags = winsound.SND_FILENAME
            play_flags |= getattr(winsound, "SND_SYNC", 0)
            winsound.PlaySound(wav_path, play_flags)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass
        return

    # Best-effort fallback for non-Windows hosts.
    time.sleep(len(probe) / float(sample_rate))


def main() -> int:
    parser = argparse.ArgumentParser(description="AudioForge self-test (probe capture).")
    parser.add_argument("--duration", type=float, default=2.5, help="Recording duration in seconds.")
    parser.add_argument("--delay", type=float, default=0.45, help="Probe playback delay in seconds.")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum correlation confidence to pass.",
    )
    parser.add_argument("--input-device", type=str, default=None, help="Input device name.")
    parser.add_argument("--output-device", type=str, default=None, help="Output device name.")
    args = parser.parse_args()

    processor = AudioProcessor()
    try:
        result = processor.start(args.input_device, args.output_device)
        print(f"Started processor: {result}")

        processor.start_raw_recording(args.duration)
        start = time.time()
        played = False

        probe = generate_probe_signal(sample_rate=48_000, duration_ms=80.0)

        while True:
            elapsed = time.time() - start
            if (not played) and elapsed >= args.delay:
                print("Playing probe...")
                _play_probe_blocking(probe, 48_000)
                played = True
            if elapsed >= args.duration:
                break
            time.sleep(0.02)

        raw = processor.stop_raw_recording()
        if raw is None:
            print("Self-test failed: no recording captured.")
            return 2

        recording = np.asarray(raw, dtype=np.float32)
        analysis = analyze_latency(
            reference_probe=probe,
            recorded_signal=recording,
            sample_rate=48_000,
            min_search_ms=5.0,
            max_search_ms=500.0,
        )

        if not analysis.success or analysis.confidence < args.confidence:
            print(
                "Self-test failed: low confidence "
                f"(confidence={analysis.confidence:.3f})."
            )
            return 1

        print(
            "Self-test passed: "
            f"rt={analysis.measured_round_trip_ms:.2f}ms "
            f"confidence={analysis.confidence:.3f}"
        )
        return 0
    except Exception as exc:
        print(f"Self-test error: {type(exc).__name__}: {exc}")
        return 3
    finally:
        try:
            processor.set_output_mute(False)
        except Exception:
            pass
        try:
            processor.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
