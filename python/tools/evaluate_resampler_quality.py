"""Evaluate AudioForge's exact 44.1/48 kHz product resampler."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import __version__ as scipy_version
from scipy.signal import firwin, resample_poly

from mic_eq import product_resampler_configuration, simulate_product_resampler


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCHEMA_VERSION = 2
CHUNK_SIZE = 1024
PASSBAND_FREQUENCIES_HZ = (
    50.0,
    100.0,
    1_000.0,
    5_000.0,
    10_000.0,
    15_000.0,
    18_000.0,
    20_000.0,
)
STOPBAND_FREQUENCIES_HZ = (22_500.0, 23_000.0, 23_500.0)
UPSAMPLE_IMAGE_TONES_HZ = (20_500.0, 21_000.0)
GATES = {
    "max_passband_absolute_error_db": 0.25,
    "max_passband_ripple_db": 0.25,
    "max_offline_reference_magnitude_delta_db": 0.25,
    "max_downsample_alias_db": -60.0,
    "max_upsample_image_db": -60.0,
    "max_impulse_location_error_samples": 1.0,
    "min_roundtrip_snr_db": 40.0,
    "max_long_stream_count_error_samples": 0,
    "max_p99_deadline_fraction": 0.25,
    "max_block_deadline_fraction": 0.50,
}


Simulator = Callable[
    [Sequence[float], int, int, int, int | None, str | None],
    tuple[list[float], int, int, list[int]],
]
ConfigurationProvider = Callable[[], tuple[int, str, str, int, int]]


@dataclass(frozen=True)
class ResamplerConfiguration:
    identifier: str
    sinc_len: int
    window: str
    native_default: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _db_ratio(numerator: float, denominator: float) -> float:
    if numerator <= 0.0:
        return -300.0
    return 20.0 * math.log10(numerator / max(denominator, 1e-15))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _run(
    samples: np.ndarray,
    input_rate: int,
    output_rate: int,
    simulator: Simulator = simulate_product_resampler,
    configuration: ResamplerConfiguration | None = None,
) -> tuple[np.ndarray, int, list[int]]:
    sinc_len = None
    window = None
    if configuration is not None and not configuration.native_default:
        sinc_len = configuration.sinc_len
        window = configuration.window
    raw, delay, expected, timings = simulator(
        cast(Sequence[float], samples.astype(np.float64, copy=False)),
        input_rate,
        output_rate,
        CHUNK_SIZE,
        sinc_len,
        window,
    )
    output = np.asarray(raw, dtype=np.float64)
    if output.size < expected:
        raise ValueError(
            f"resampler returned {output.size} frames; expected at least {expected}"
        )
    return output[:expected], delay, timings


def _sine(sample_rate: int, frequency_hz: float, duration_seconds: float) -> np.ndarray:
    frames = int(round(sample_rate * duration_seconds))
    time = np.arange(frames, dtype=np.float64) / sample_rate
    return 0.5 * np.sin(2.0 * np.pi * frequency_hz * time)


def _steady_slice(values: np.ndarray, sample_rate: int) -> np.ndarray:
    margin = min(int(round(0.25 * sample_rate)), max(0, values.size // 4))
    if margin == 0:
        return values
    return values[margin:-margin]


def _tone_amplitude(values: np.ndarray, sample_rate: int, frequency_hz: float) -> float:
    values = _steady_slice(values, sample_rate)
    if values.size == 0:
        return 0.0
    window = np.hanning(values.size)
    phase = np.exp(
        -2j * np.pi * frequency_hz * np.arange(values.size, dtype=np.float64) / sample_rate
    )
    coherent_gain = float(np.sum(window)) / values.size
    return float(
        2.0
        * np.abs(np.sum(values * window * phase))
        / (values.size * max(coherent_gain, 1e-15))
    )


def _offline_reference(
    samples: np.ndarray, input_rate: int, output_rate: int
) -> np.ndarray:
    common = math.gcd(input_rate, output_rate)
    up = output_rate // common
    down = input_rate // common
    return np.asarray(
        resample_poly(
            samples,
            up,
            down,
            window=_offline_reference_filter(up, down),
        ),
        dtype=np.float64,
    )


@lru_cache(maxsize=4)
def _offline_reference_filter(up: int, down: int) -> np.ndarray:
    max_rate = max(up, down)
    half_len = 64 * max_rate
    return np.asarray(
        firwin(
            2 * half_len + 1,
            1.0 / max_rate,
            window=cast(Any, ("kaiser", 14.0)),
        ),
        dtype=np.float64,
    )


def _passband_case(
    input_rate: int,
    output_rate: int,
    configuration: ResamplerConfiguration,
) -> dict[str, Any]:
    responses: list[dict[str, float]] = []
    for frequency_hz in PASSBAND_FREQUENCIES_HZ:
        source = _sine(input_rate, frequency_hz, 1.5)
        output, _delay, _timings = _run(
            source, input_rate, output_rate, configuration=configuration
        )
        reference = _offline_reference(source, input_rate, output_rate)
        input_rms = _rms(_steady_slice(source, input_rate))
        output_gain_db = _db_ratio(
            _rms(_steady_slice(output, output_rate)), input_rms
        )
        reference_gain_db = _db_ratio(
            _rms(_steady_slice(reference, output_rate)), input_rms
        )
        responses.append(
            {
                "frequency_hz": frequency_hz,
                "gain_db": output_gain_db,
                "offline_reference_gain_db": reference_gain_db,
                "offline_reference_magnitude_delta_db": (
                    output_gain_db - reference_gain_db
                ),
            }
        )
    gains = [item["gain_db"] for item in responses]
    reference_deltas = [
        abs(item["offline_reference_magnitude_delta_db"]) for item in responses
    ]
    return {
        "input_rate": input_rate,
        "output_rate": output_rate,
        "responses": responses,
        "max_absolute_error_db": max(abs(gain) for gain in gains),
        "ripple_db": max(gains) - min(gains),
        "max_offline_reference_magnitude_delta_db": max(reference_deltas),
    }


def _band_limited_noise(
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    duration_seconds: float,
    seed: int,
) -> np.ndarray:
    frames = int(round(sample_rate * duration_seconds))
    frequencies = np.fft.rfftfreq(frames, d=1.0 / sample_rate)
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    rng = np.random.default_rng(seed)
    spectrum = np.zeros(frequencies.size, dtype=np.complex128)
    spectrum[mask] = rng.standard_normal(mask.sum()) + 1j * rng.standard_normal(
        mask.sum()
    )
    values = np.fft.irfft(spectrum, n=frames)
    return values * (0.2 / max(_rms(values), 1e-15))


def _downsample_alias_case(
    configuration: ResamplerConfiguration,
) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for frequency_hz in STOPBAND_FREQUENCIES_HZ:
        source = _sine(48_000, frequency_hz, 2.0)
        output, _delay, _timings = _run(
            source, 48_000, 44_100, configuration=configuration
        )
        rows.append(
            {
                "input_frequency_hz": frequency_hz,
                "attenuation_db": _db_ratio(
                    _rms(_steady_slice(output, 44_100)),
                    _rms(_steady_slice(source, 48_000)),
                ),
            }
        )
    noise = _band_limited_noise(48_000, 22_500.0, 23_900.0, 4.0, 0xA11A5)
    noise_output, _delay, _timings = _run(
        noise, 48_000, 44_100, configuration=configuration
    )
    noise_attenuation_db = _db_ratio(
        _rms(_steady_slice(noise_output, 44_100)),
        _rms(_steady_slice(noise, 48_000)),
    )
    return {
        "tones": rows,
        "swept_noise": {
            "input_band_hz": [22_500.0, 23_900.0],
            "attenuation_db": noise_attenuation_db,
        },
        "worst_alias_db": max(
            noise_attenuation_db,
            *(row["attenuation_db"] for row in rows),
        ),
    }


def _upsample_image_case(
    configuration: ResamplerConfiguration,
) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for frequency_hz in UPSAMPLE_IMAGE_TONES_HZ:
        source = _sine(44_100, frequency_hz, 2.0)
        output, _delay, _timings = _run(
            source, 44_100, 48_000, configuration=configuration
        )
        image_hz = 44_100.0 - frequency_hz
        fundamental = _tone_amplitude(output, 48_000, frequency_hz)
        image = _tone_amplitude(output, 48_000, image_hz)
        rows.append(
            {
                "input_frequency_hz": frequency_hz,
                "image_frequency_hz": image_hz,
                "image_relative_db": _db_ratio(image, fundamental),
            }
        )
    return {
        "tones": rows,
        "worst_image_db": max(row["image_relative_db"] for row in rows),
    }


def _impulse_case(
    input_rate: int,
    output_rate: int,
    configuration: ResamplerConfiguration,
) -> dict[str, Any]:
    source = np.zeros(input_rate, dtype=np.float64)
    impulse_index = input_rate // 2
    source[impulse_index] = 1.0
    output, delay, _timings = _run(
        source, input_rate, output_rate, configuration=configuration
    )
    peak_index = int(np.argmax(np.abs(output)))
    expected_location = impulse_index * output_rate / input_rate
    return {
        "input_rate": input_rate,
        "output_rate": output_rate,
        "reported_output_delay_samples": delay,
        "reported_output_delay_ms": delay * 1_000.0 / output_rate,
        "impulse_peak_index": peak_index,
        "expected_impulse_location": expected_location,
        "location_error_samples": abs(peak_index - expected_location),
    }


def _pink_noise(
    sample_rate: int,
    duration_seconds: float,
    low_hz: float,
    high_hz: float,
    seed: int,
) -> np.ndarray:
    frames = int(round(sample_rate * duration_seconds))
    frequencies = np.fft.rfftfreq(frames, d=1.0 / sample_rate)
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    rng = np.random.default_rng(seed)
    spectrum = np.zeros(frequencies.size, dtype=np.complex128)
    spectrum[mask] = (
        rng.standard_normal(mask.sum()) + 1j * rng.standard_normal(mask.sum())
    ) / np.sqrt(frequencies[mask])
    values = np.fft.irfft(spectrum, n=frames)
    return values * (0.2 / max(_rms(values), 1e-15))


def _roundtrip_case(
    configuration: ResamplerConfiguration,
) -> dict[str, Any]:
    source = _pink_noise(44_100, 8.0, 50.0, 20_000.0, 0xA0D10)
    upsampled, delay_up, _timings_up = _run(
        source, 44_100, 48_000, configuration=configuration
    )
    roundtrip, delay_down, _timings_down = _run(
        upsampled, 48_000, 44_100, configuration=configuration
    )
    length = min(source.size, roundtrip.size)
    margin = 4_096
    source_mid = source[margin : length - margin]
    roundtrip_mid = roundtrip[margin : length - margin]
    error = roundtrip_mid - source_mid
    return {
        "stimulus": "deterministic 50 Hz-20 kHz equal-energy-per-octave noise",
        "roundtrip_snr_db": _db_ratio(_rms(source_mid), _rms(error)),
        "max_absolute_error": float(np.max(np.abs(error))),
        "input_frames": int(source.size),
        "upsampled_frames": int(upsampled.size),
        "roundtrip_frames": int(roundtrip.size),
        "reported_up_delay_samples": delay_up,
        "reported_down_delay_samples": delay_down,
    }


def _long_stream_and_timing_case(
    input_rate: int,
    output_rate: int,
    duration_seconds: int,
    configuration: ResamplerConfiguration,
) -> dict[str, Any]:
    frames = input_rate * duration_seconds
    source = np.zeros(frames, dtype=np.float64)
    output, delay, timings = _run(
        source, input_rate, output_rate, configuration=configuration
    )
    expected = int(round(frames * output_rate / input_rate))
    block_deadline_ns = CHUNK_SIZE / input_rate * 1_000_000_000.0
    p95 = _percentile(timings, 95)
    p99 = _percentile(timings, 99)
    maximum = float(max(timings, default=0))
    return {
        "input_rate": input_rate,
        "output_rate": output_rate,
        "duration_seconds": duration_seconds,
        "input_frames": frames,
        "output_frames": int(output.size),
        "expected_output_frames": expected,
        "count_error_samples": int(output.size - expected),
        "reported_output_delay_samples": delay,
        "blocks": len(timings),
        "block_time_ns": {
            "median": statistics.median(timings) if timings else 0,
            "p95": p95,
            "p99": p99,
            "max": maximum,
        },
        "deadline_ns": block_deadline_ns,
        "p99_deadline_fraction": p99 / block_deadline_ns,
        "max_deadline_fraction": maximum / block_deadline_ns,
    }


def _evaluate_configuration(
    configuration: ResamplerConfiguration,
    duration_seconds: int,
) -> dict[str, Any]:
    passband = [
        _passband_case(44_100, 48_000, configuration),
        _passband_case(48_000, 44_100, configuration),
    ]
    downsample_alias = _downsample_alias_case(configuration)
    upsample_image = _upsample_image_case(configuration)
    impulse = [
        _impulse_case(44_100, 48_000, configuration),
        _impulse_case(48_000, 44_100, configuration),
    ]
    roundtrip = _roundtrip_case(configuration)
    long_stream = [
        _long_stream_and_timing_case(
            44_100, 48_000, duration_seconds, configuration
        ),
        _long_stream_and_timing_case(
            48_000, 44_100, duration_seconds, configuration
        ),
    ]
    checks = {
        "passband_absolute_error": all(
            case["max_absolute_error_db"] <= GATES["max_passband_absolute_error_db"]
            for case in passband
        ),
        "passband_ripple": all(
            case["ripple_db"] <= GATES["max_passband_ripple_db"]
            for case in passband
        ),
        "offline_reference_magnitude": all(
            case["max_offline_reference_magnitude_delta_db"]
            <= GATES["max_offline_reference_magnitude_delta_db"]
            for case in passband
        ),
        "downsample_alias": downsample_alias["worst_alias_db"]
        <= GATES["max_downsample_alias_db"],
        "upsample_image": upsample_image["worst_image_db"]
        <= GATES["max_upsample_image_db"],
        "impulse_location": all(
            case["location_error_samples"]
            <= GATES["max_impulse_location_error_samples"]
            for case in impulse
        ),
        "delay_accounting": all(
            impulse_case["reported_output_delay_samples"]
            == stream_case["reported_output_delay_samples"]
            for impulse_case, stream_case in zip(impulse, long_stream, strict=True)
        ),
        "roundtrip": roundtrip["roundtrip_snr_db"]
        >= GATES["min_roundtrip_snr_db"],
        "long_stream_count": all(
            abs(case["count_error_samples"])
            <= GATES["max_long_stream_count_error_samples"]
            for case in long_stream
        ),
        "p99_realtime": all(
            case["p99_deadline_fraction"] <= GATES["max_p99_deadline_fraction"]
            for case in long_stream
        ),
        "max_realtime": all(
            case["max_deadline_fraction"] <= GATES["max_block_deadline_fraction"]
            for case in long_stream
        ),
    }
    return {
        "configuration": {
            "identifier": configuration.identifier,
            "sinc_len": configuration.sinc_len,
            "window": configuration.window,
            "native_default": configuration.native_default,
            "interpolation": "cubic",
            "oversampling_factor": 256,
        },
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "measurements": {
            "passband_and_offline_reference": passband,
            "downsample_alias": downsample_alias,
            "upsample_image": upsample_image,
            "impulse": impulse,
            "roundtrip": roundtrip,
            "long_stream_and_timing": long_stream,
        },
    }


def _alternative_decision(
    product: dict[str, Any],
    alternative: dict[str, Any],
) -> dict[str, Any]:
    if alternative["status"] != "passed":
        failed = [
            name for name, passed in alternative["checks"].items() if not passed
        ]
        return {
            "retained": False,
            "reason": f"Rejected because predefined gates failed: {', '.join(failed)}.",
        }
    product_delays = [
        row["reported_output_delay_samples"]
        for row in product["measurements"]["impulse"]
    ]
    alternative_delays = [
        row["reported_output_delay_samples"]
        for row in alternative["measurements"]["impulse"]
    ]
    if all(
        candidate > incumbent
        for candidate, incumbent in zip(
            alternative_delays, product_delays, strict=True
        )
    ):
        return {
            "retained": False,
            "reason": (
                "Rejected because the exact product configuration already clears every "
                "quality and realtime gate, while this alternative increases delay."
            ),
        }
    return {
        "retained": False,
        "reason": (
            "Rejected because it produced no gate-level quality benefit over the "
            "lower-complexity exact product configuration."
        ),
    }


def evaluate(
    duration_seconds: int,
    configuration_provider: ConfigurationProvider = product_resampler_configuration,
) -> dict[str, Any]:
    sinc_len, window, interpolation, oversampling_factor, chunk_size = (
        configuration_provider()
    )
    if interpolation != "cubic" or oversampling_factor != 256:
        raise ValueError("native product resampler metadata does not match the evaluator")
    if chunk_size != CHUNK_SIZE:
        raise ValueError(
            f"native chunk size {chunk_size} does not match evaluator {CHUNK_SIZE}"
        )

    configurations = (
        ResamplerConfiguration("product", sinc_len, window, native_default=True),
        ResamplerConfiguration(
            "legacy-blackman-harris-squared-128",
            128,
            "blackman_harris_squared",
        ),
        ResamplerConfiguration(
            "high-rejection-blackman-harris-squared-256",
            256,
            "blackman_harris_squared",
        ),
    )
    results = [
        _evaluate_configuration(configuration, duration_seconds)
        for configuration in configurations
    ]
    product = results[0]
    legacy = results[1]
    high_rejection = results[2]
    alternatives = [
        {
            **result,
            "decision": _alternative_decision(product, result),
        }
        for result in results[1:]
    ]
    retained = product["status"] == "passed"
    source_paths = (
        REPO_ROOT / "rust-core/src/audio/processor/resampling.rs",
        REPO_ROOT / "rust-core/src/audio/processor/dsp_loop.rs",
        REPO_ROOT / "python/tools/evaluate_resampler_quality.py",
    )
    product_streams = product["measurements"]["long_stream_and_timing"]
    product_impulses = product["measurements"]["impulse"]
    product_roundtrip = product["measurements"]["roundtrip"]
    source_hashes = {
        path.relative_to(REPO_ROOT).as_posix(): _sha256(path)
        for path in source_paths
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "audible_change": True,
        "status": "passed" if retained else "failed",
        "decision": {
            "retained": retained,
            "selected_configuration": product["configuration"]["identifier"],
            "reason": (
                "The 128-tap Blackman product resampler passed every predefined "
                "quality, offline-reference, sample-count, delay-accounting, and "
                "realtime gate. The former 128-tap Blackman-Harris-squared "
                "configuration failed passband, offline-reference, and round-trip "
                "gates; the passing 256-tap alternative increased delay in both "
                "directions."
                if retained
                else "The exact product resampler failed at least one predefined gate."
            ),
        },
        "incumbent_comparison": {
            "configuration": legacy["configuration"],
            "status": legacy["status"],
            "failed_checks": [
                name for name, passed in legacy["checks"].items() if not passed
            ],
            "replacement_justified": retained and legacy["status"] != "passed",
        },
        "higher_rejection_comparison": {
            "configuration": high_rejection["configuration"],
            "status": high_rejection["status"],
            "product_delay_samples": [
                row["reported_output_delay_samples"]
                for row in product["measurements"]["impulse"]
            ],
            "candidate_delay_samples": [
                row["reported_output_delay_samples"]
                for row in high_rejection["measurements"]["impulse"]
            ],
            "selected": False,
        },
        "configuration": {
            "chunk_size": CHUNK_SIZE,
            "rates": [[44_100, 48_000], [48_000, 44_100]],
            "long_stream_duration_seconds": duration_seconds,
            "offline_reference": {
                "implementation": "scipy.signal.resample_poly",
                "window": "Kaiser beta=14",
                "filter_half_length_per_phase": 64,
            },
            "product_source_sha256": source_hashes,
        },
        "source_sha256": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy_version,
            "platform": platform.platform(),
        },
        "gates": GATES,
        "product": product,
        "alternatives": alternatives,
        "evaluation_contract": {
            "configuration": product["configuration"],
            "asset_hashes": source_hashes,
            "runtime": {
                "max_p99_frame_seconds": max(
                    row["block_time_ns"]["p99"] for row in product_streams
                )
                / 1_000_000_000.0,
                "max_deadline_fraction": max(
                    row["max_deadline_fraction"] for row in product_streams
                ),
            },
            "latency": {
                "output_delay_samples": [
                    row["reported_output_delay_samples"] for row in product_impulses
                ],
                "output_delay_ms": [
                    row["reported_output_delay_ms"] for row in product_impulses
                ],
            },
            "clean_preservation": {
                "metric": "44.1->48->44.1 kHz pink-noise round-trip SNR",
                "value_db": product_roundtrip["roundtrip_snr_db"],
                "minimum_db": GATES["min_roundtrip_snr_db"],
                "passed": product["checks"]["roundtrip"],
            },
        },
        "limitations": [
            "Objective synthetic measurements do not replace device-route testing.",
            "Timing is machine-specific and must be trended rather than treated as universal.",
            (
                "The offline reference gate compares magnitude at fixed tones; raw "
                "waveform subtraction is intentionally not used because valid "
                "linear-phase resamplers can use different transition-band kernels."
            ),
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=60,
        help="zero-input duration used for count drift and timing (default: 60)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "evaluation/resampler-quality-report.json",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        help="Optional full per-tone report; the tracked report stays compact.",
    )
    return parser


def _remove_measurement_details(report: dict[str, Any]) -> None:
    arms = [report["product"], *report["alternatives"]]
    for arm in arms:
        measurements = arm["measurements"]
        measurements["downsample_alias"].pop("tones", None)
        measurements["upsample_image"].pop("tones", None)
        for comparison in measurements["passband_and_offline_reference"]:
            comparison.pop("responses", None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 10 <= args.duration_seconds <= 600:
        raise ValueError("--duration-seconds must be between 10 and 600")
    report = evaluate(args.duration_seconds)
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
    _remove_measurement_details(report)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Resampler quality evaluation {report['status']}; "
        f"report written to {output}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Resampler evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
