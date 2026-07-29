"""Measure 48 kHz dynamics artifacts against a 192 kHz reference render."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import correlate, correlation_lags, resample_poly

from mic_eq import simulate_auto_eq_chain


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "dynamics-aliasing-report.json"
BASE_RATE = 48_000
REFERENCE_RATE = 192_000
CASES = (
    ("carrier_8k", 8_000.0, 37.0),
    ("carrier_11k", 11_000.0, 73.0),
    ("carrier_15k", 15_000.0, 113.0),
    ("carrier_18k", 18_000.0, 157.0),
)


def _signal(sample_rate: int, carrier_hz: float, modulation_hz: float) -> np.ndarray:
    duration = 4.0
    time = np.arange(int(duration * sample_rate), dtype=np.float64) / sample_rate
    slow_envelope = 0.08 + 0.72 * np.square(
        0.5 + 0.5 * np.sin(2.0 * np.pi * modulation_hz * time)
    )
    transient_period = max(1, int(round(0.173 * sample_rate)))
    transient_phase = np.arange(time.size) % transient_period
    transient = np.exp(-transient_phase / max(1.0, 0.0015 * sample_rate))
    envelope = np.clip(slow_envelope + 0.35 * transient, 0.0, 0.95)
    return np.asarray(
        envelope * np.sin(2.0 * np.pi * carrier_hz * time),
        dtype=np.float32,
    )


def _settings() -> dict[str, object]:
    return {
        "deesser_enabled": False,
        "compressor_enabled": True,
        "compressor_threshold_db": -24.0,
        "compressor_ratio": 8.0,
        "compressor_attack_ms": 0.5,
        "compressor_release_ms": 50.0,
        "compressor_makeup_gain_db": 0.0,
        "compressor_adaptive_release": False,
        "compressor_sidechain_highpass_enabled": False,
        "limiter_enabled": False,
        "return_output_audio": True,
    }


def _render(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
    neutral_bands = [(100.0 * 1.7**index, 0.0, 1.0) for index in range(10)]
    result = simulate_auto_eq_chain(audio, sample_rate, neutral_bands, _settings())
    return np.asarray(result.pop("output_audio"), dtype=np.float64), result


def _align(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    probe_count = min(reference.size, candidate.size, BASE_RATE)
    values = correlate(
        candidate[:probe_count] - np.mean(candidate[:probe_count]),
        reference[:probe_count] - np.mean(reference[:probe_count]),
        mode="full",
        method="fft",
    )
    lags = correlation_lags(probe_count, probe_count, mode="full")
    allowed = np.abs(lags) <= 256
    lag = int(lags[allowed][int(np.argmax(np.abs(values[allowed])))])
    if lag >= 0:
        count = min(reference.size, candidate.size - lag)
        return reference[:count], candidate[lag : lag + count], lag
    count = min(reference.size + lag, candidate.size)
    return reference[-lag : -lag + count], candidate[:count], lag


def _relative_error_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    error = candidate - reference
    reference_rms = float(np.sqrt(np.mean(np.square(reference))))
    error_rms = float(np.sqrt(np.mean(np.square(error))))
    return float(20.0 * np.log10(max(error_rms, 1e-12) / max(reference_rms, 1e-12)))


def _folded_error_db(
    reference: np.ndarray,
    candidate: np.ndarray,
    carrier_hz: float,
    modulation_hz: float,
) -> float:
    window = np.hanning(reference.size)
    reference_spectrum = np.fft.rfft(reference * window)
    error_spectrum = np.fft.rfft((candidate - reference) * window)
    frequencies = np.fft.rfftfreq(reference.size, 1.0 / BASE_RATE)
    expected = np.zeros(frequencies.shape, dtype=bool)
    for sideband in range(-12, 13):
        center = carrier_hz + sideband * modulation_hz
        if 0.0 <= center <= BASE_RATE / 2:
            expected |= np.abs(frequencies - center) <= max(8.0, modulation_hz * 0.12)
    folded_energy = float(np.sum(np.square(np.abs(error_spectrum[~expected]))))
    reference_energy = float(np.sum(np.square(np.abs(reference_spectrum))))
    return float(
        10.0
        * np.log10(max(folded_energy, 1e-24) / max(reference_energy, 1e-24))
    )


def _case(case_id: str, carrier_hz: float, modulation_hz: float) -> dict[str, Any]:
    base_input = _signal(BASE_RATE, carrier_hz, modulation_hz)
    reference_input = _signal(REFERENCE_RATE, carrier_hz, modulation_hz)
    base_output, base_diagnostics = _render(base_input, BASE_RATE)
    reference_output, reference_diagnostics = _render(
        reference_input, REFERENCE_RATE
    )
    downsampled = resample_poly(reference_output, 1, REFERENCE_RATE // BASE_RATE)
    reference, candidate, lag = _align(downsampled, base_output)
    trim = BASE_RATE // 2
    if reference.size > 2 * trim:
        reference = reference[trim:-trim]
        candidate = candidate[trim:-trim]
    return {
        "id": case_id,
        "carrier_hz": carrier_hz,
        "modulation_hz": modulation_hz,
        "alignment_lag_samples": lag,
        "relative_waveform_error_db": _relative_error_db(reference, candidate),
        "folded_out_of_expected_error_db": _folded_error_db(
            reference, candidate, carrier_hz, modulation_hz
        ),
        "base_runtime_ms": float(base_diagnostics["candidate_runtime_ms"]),
        "reference_runtime_ms": float(reference_diagnostics["candidate_runtime_ms"]),
        "base_peak_gain_reduction_db": float(
            base_diagnostics["compressor_gain_reduction_db"]
        ),
        "reference_peak_gain_reduction_db": float(
            reference_diagnostics["compressor_gain_reduction_db"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    rows = [_case(*case) for case in CASES]
    metrics = {
        "median_relative_waveform_error_db": float(
            np.median([row["relative_waveform_error_db"] for row in rows])
        ),
        "worst_relative_waveform_error_db": max(
            float(row["relative_waveform_error_db"]) for row in rows
        ),
        "median_folded_out_of_expected_error_db": float(
            np.median([row["folded_out_of_expected_error_db"] for row in rows])
        ),
        "worst_folded_out_of_expected_error_db": max(
            float(row["folded_out_of_expected_error_db"]) for row in rows
        ),
    }
    material = bool(metrics["worst_folded_out_of_expected_error_db"] > -30.0)
    control_law_mismatch = bool(
        metrics["median_relative_waveform_error_db"] > -35.0
    )
    report = {
        "schema_version": 2,
        "audible_change": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "48 kHz dynamics aliasing against aligned 192 kHz reference",
        "predefined_materiality_gate": {
            "worst_folded_out_of_expected_error_db_max": -30.0,
        },
        "material_artifact_detected": material,
        "high_rate_control_law_mismatch_detected": control_law_mismatch,
        "decision": (
            "prototype compressor oversampling"
            if material
            else "retain current dynamics path; no oversampling justified"
        ),
        "metrics": metrics,
        "cases": rows,
        "configuration": {
            **_settings(),
            "base_sample_rate": BASE_RATE,
            "reference_sample_rate": REFERENCE_RATE,
            "reference_downsampler": "scipy.signal.resample_poly",
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "limitations": [
            "This is a focused worst-case diagnostic, not a perceptual listening result.",
            "The 192 kHz render is a high-rate implementation reference, not an analytic alias-free compressor.",
            "Broadband waveform error is reported separately because it includes sample-peak detector and envelope differences, not only folded inharmonic energy.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"material_artifact_detected": material, **metrics}, sort_keys=True))
    print(f"Wrote {args.report.resolve().relative_to(REPO_ROOT.resolve()).as_posix()}")
    return 1 if material else 0


if __name__ == "__main__":
    raise SystemExit(main())
