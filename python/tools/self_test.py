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
from dataclasses import dataclass

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


@dataclass
class SelfTestAttempt:
    confidence: float
    round_trip_ms: float
    message: str
    diagnostics: dict
    playback_delay_ms: float


def _run_attempt(
    processor: AudioProcessor,
    *,
    duration: float,
    delay: float,
    output_sample_rate: int,
    probe_duration_ms: float,
    expected_latency_min_ms: float,
    expected_latency_max_ms: float,
    expected_playback_jitter_ms: float,
) -> SelfTestAttempt:
    processor.start_raw_recording(duration)
    start = time.time()
    played = False
    probe = generate_probe_signal(sample_rate=output_sample_rate, duration_ms=probe_duration_ms)

    while True:
        elapsed = time.time() - start
        if (not played) and elapsed >= delay:
            print("Playing probe...")
            _play_probe_blocking(probe, output_sample_rate)
            played = True
        if elapsed >= duration:
            break
        time.sleep(0.02)

    raw = processor.stop_raw_recording()
    if raw is None:
        return SelfTestAttempt(
            confidence=0.0,
            round_trip_ms=0.0,
            message="no recording captured",
            diagnostics=processor.get_runtime_diagnostics(),
            playback_delay_ms=delay * 1000.0,
        )

    recording = np.asarray(raw, dtype=np.float32)
    analysis = analyze_latency(
        reference_probe=probe,
        recorded_signal=recording,
        sample_rate=output_sample_rate,
        min_search_ms=5.0,
        max_search_ms=500.0,
        expected_playback_start_ms=delay * 1000.0,
        expected_playback_jitter_ms=expected_playback_jitter_ms,
        expected_latency_min_ms=expected_latency_min_ms,
        expected_latency_max_ms=expected_latency_max_ms,
    )
    return SelfTestAttempt(
        confidence=float(analysis.confidence),
        round_trip_ms=float(analysis.measured_round_trip_ms),
        message=analysis.message or ("ok" if analysis.success else "low confidence"),
        diagnostics=processor.get_runtime_diagnostics(),
        playback_delay_ms=delay * 1000.0,
    )


def _format_diagnostics(diagnostics: dict) -> str:
    return (
        "backend={backend} failed={failed} dropped_in={dropped_in} "
        "underruns={underruns} restarts={restarts} non_finite={non_finite}".format(
            backend=diagnostics.get("noise_model", "unknown"),
            failed=diagnostics.get("noise_backend_failed", False),
            dropped_in=diagnostics.get("input_dropped_samples", 0),
            underruns=diagnostics.get("output_underrun_total", 0),
            restarts=diagnostics.get("stream_restart_count", 0),
            non_finite=diagnostics.get("suppressor_non_finite_count", 0),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AudioForge self-test (probe capture).")
    parser.add_argument("--duration", type=float, default=3.0, help="Recording duration in seconds.")
    parser.add_argument("--delay", type=float, default=0.45, help="Probe playback delay in seconds.")
    parser.add_argument("--warmup", type=float, default=0.3, help="Stream warmup time in seconds.")
    parser.add_argument("--retries", type=int, default=7, help="Additional retries after the first failed attempt.")
    parser.add_argument(
        "--probe-duration-ms",
        type=float,
        default=120.0,
        help="Probe chirp duration in milliseconds.",
    )
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
        output_sample_rate = int(processor.output_sample_rate() or 48_000)
        print(f"Output sample rate: {output_sample_rate} Hz")
        if args.warmup > 0.0:
            print(f"Warming up for {args.warmup:.2f}s...")
            time.sleep(args.warmup)

        processor.set_recovery_suppressed(True)
        attempts = max(1, args.retries + 1)
        delay_ladder = [args.delay, args.delay + 0.25, args.delay + 0.50]
        attempt_delays = [delay_ladder[min(index, len(delay_ladder) - 1)] for index in range(attempts)]
        best_attempt: SelfTestAttempt | None = None

        for attempt_index in range(attempts):
            if attempt_index > 0:
                print(f"Retrying self-test ({attempt_index + 1}/{attempts})...")
                time.sleep(0.15)

            expected_latency_min_ms = 20.0
            expected_latency_max_ms = 500.0
            if best_attempt is not None and best_attempt.round_trip_ms > 0.0:
                expected_latency_min_ms = max(20.0, best_attempt.round_trip_ms - 120.0)
                expected_latency_max_ms = min(500.0, best_attempt.round_trip_ms + 120.0)

            attempt = _run_attempt(
                processor,
                duration=args.duration,
                delay=attempt_delays[attempt_index],
                output_sample_rate=output_sample_rate,
                probe_duration_ms=args.probe_duration_ms,
                expected_latency_min_ms=expected_latency_min_ms,
                expected_latency_max_ms=expected_latency_max_ms,
                expected_playback_jitter_ms=150.0,
            )
            attempt_passed = attempt.confidence >= args.confidence and attempt.round_trip_ms > 0.0
            attempt_message = "ok" if attempt_passed else attempt.message
            print(
                f"Attempt {attempt_index + 1}: "
                f"delay={attempt.playback_delay_ms:.0f}ms "
                f"rt={attempt.round_trip_ms:.2f}ms "
                f"confidence={attempt.confidence:.3f} "
                f"message={attempt_message} "
                f"{_format_diagnostics(attempt.diagnostics)}"
            )

            if best_attempt is None or attempt.confidence > best_attempt.confidence:
                best_attempt = attempt
            if attempt_passed:
                print(
                    "Self-test passed: "
                    f"rt={attempt.round_trip_ms:.2f}ms "
                    f"confidence={attempt.confidence:.3f}"
                )
                return 0

        if best_attempt is None:
            print("Self-test failed: no recording captured.")
            return 2

        print(
            "Self-test failed: low confidence "
            f"(confidence={best_attempt.confidence:.3f})."
        )
        return 1
    except Exception as exc:
        print(f"Self-test error: {type(exc).__name__}: {exc}")
        return 3
    finally:
        try:
            processor.set_output_mute(False)
        except Exception:
            pass
        try:
            processor.set_recovery_suppressed(False)
        except Exception:
            pass
        try:
            processor.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
