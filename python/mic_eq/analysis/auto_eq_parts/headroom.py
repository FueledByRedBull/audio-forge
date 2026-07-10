"""Headroom-aware Auto-EQ validation and offline chain simulation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
from scipy.signal import lfilter, resample_poly

from .constants import NUM_EQ_BANDS

HEADROOM_TARGET_DB = 1.0
LIMITER_GAIN_REDUCTION_WARN_DB = 1.0
TRUE_PEAK_GAIN_REDUCTION_WARN_DB = 0.5
HEADROOM_SCALES = (1.0, 0.85, 0.70, 0.55, 0.40, 0.25, 0.0)


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1.0e-12)))


def _linear(db_value: float) -> float:
    return float(10.0 ** (float(db_value) / 20.0))


def _as_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _flatten_chain_settings(chain_settings: dict[str, Any] | None) -> dict[str, Any]:
    chain_settings = chain_settings or {}
    deesser = chain_settings.get("deesser") or {}
    compressor = chain_settings.get("compressor") or {}
    limiter = chain_settings.get("limiter") or {}

    return {
        "deesser_enabled": _as_bool(deesser.get("enabled"), False),
        "deesser_auto_enabled": _as_bool(deesser.get("auto_enabled"), True),
        "deesser_auto_amount": _as_float(deesser.get("auto_amount"), 0.5),
        "deesser_low_cut_hz": _as_float(deesser.get("low_cut_hz"), 4000.0),
        "deesser_high_cut_hz": _as_float(deesser.get("high_cut_hz"), 11000.0),
        "deesser_threshold_db": _as_float(deesser.get("threshold_db"), -28.0),
        "deesser_ratio": _as_float(deesser.get("ratio"), 4.0),
        "deesser_attack_ms": _as_float(deesser.get("attack_ms"), 2.0),
        "deesser_release_ms": _as_float(deesser.get("release_ms"), 80.0),
        "deesser_max_reduction_db": _as_float(deesser.get("max_reduction_db"), 6.0),
        "compressor_enabled": _as_bool(compressor.get("enabled"), True),
        "compressor_threshold_db": _as_float(compressor.get("threshold_db"), -20.0),
        "compressor_ratio": _as_float(compressor.get("ratio"), 4.0),
        "compressor_attack_ms": _as_float(compressor.get("attack_ms"), 10.0),
        "compressor_release_ms": _as_float(compressor.get("release_ms"), 200.0),
        "compressor_makeup_gain_db": _as_float(compressor.get("makeup_gain_db"), 0.0),
        "compressor_adaptive_release": _as_bool(compressor.get("adaptive_release"), False),
        "compressor_base_release_ms": _as_float(compressor.get("base_release_ms"), 50.0),
        "compressor_auto_makeup_enabled": _as_bool(compressor.get("auto_makeup_enabled"), False),
        "compressor_target_lufs": _as_float(compressor.get("target_lufs"), -18.0),
        "compressor_sidechain_highpass_enabled": _as_bool(
            compressor.get("sidechain_highpass_enabled"),
            True,
        ),
        "limiter_enabled": _as_bool(limiter.get("enabled"), True),
        "limiter_ceiling_db": _as_float(limiter.get("ceiling_db"), -0.5),
        "limiter_release_ms": _as_float(limiter.get("release_ms"), 50.0),
        "limiter_careful_output_enabled": _as_bool(
            limiter.get("careful_output_enabled"),
            True,
        ),
    }


def _bands_from_settings(eq_settings: dict[str, Any]) -> list[tuple[float, float, float]]:
    freqs = list(eq_settings.get("band_freqs") or [])
    gains = list(eq_settings.get("band_gains") or [])
    qs = list(eq_settings.get("band_qs") or [])
    if not (len(freqs) == len(gains) == len(qs) == NUM_EQ_BANDS):
        raise ValueError("Auto-EQ settings must contain 10 frequencies, gains, and Q values")
    return [
        (_as_float(freq, 1000.0), _as_float(gain, 0.0), _as_float(q, 1.41))
        for freq, gain, q in zip(freqs, gains, qs)
    ]


def _native_simulate(
    audio_data: np.ndarray,
    sample_rate: int,
    bands: list[tuple[float, float, float]],
    flat_settings: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        from mic_eq import CORE_AVAILABLE, simulate_auto_eq_chain
    except ImportError:
        return None
    if not CORE_AVAILABLE:
        return None
    try:
        audio = np.ascontiguousarray(audio_data, dtype=np.float32)
        result: Any = simulate_auto_eq_chain(audio, float(sample_rate), bands, flat_settings)
        if not isinstance(result, Mapping):
            return None
        return {str(key): value for key, value in result.items()}
    except Exception:
        return None


def _biquad_coefficients(
    kind: str,
    frequency_hz: float,
    gain_db: float,
    q: float,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi * np.clip(frequency_hz, 20.0, sample_rate * 0.45) / sample_rate
    sin_omega = np.sin(omega)
    cos_omega = np.cos(omega)
    q = max(float(q), 1.0e-6)
    alpha = sin_omega / (2.0 * q)
    a = 10.0 ** (gain_db / 40.0)

    if kind == "peaking":
        b0 = 1.0 + alpha * a
        b1 = -2.0 * cos_omega
        b2 = 1.0 - alpha * a
        a0 = 1.0 + alpha / a
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha / a
    elif kind == "low_shelf":
        two_sqrt_a_alpha = 2.0 * np.sqrt(a) * alpha
        b0 = a * ((a + 1.0) - (a - 1.0) * cos_omega + two_sqrt_a_alpha)
        b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_omega)
        b2 = a * ((a + 1.0) - (a - 1.0) * cos_omega - two_sqrt_a_alpha)
        a0 = (a + 1.0) + (a - 1.0) * cos_omega + two_sqrt_a_alpha
        a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_omega)
        a2 = (a + 1.0) + (a - 1.0) * cos_omega - two_sqrt_a_alpha
    else:
        two_sqrt_a_alpha = 2.0 * np.sqrt(a) * alpha
        b0 = a * ((a + 1.0) + (a - 1.0) * cos_omega + two_sqrt_a_alpha)
        b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_omega)
        b2 = a * ((a + 1.0) + (a - 1.0) * cos_omega - two_sqrt_a_alpha)
        a0 = (a + 1.0) - (a - 1.0) * cos_omega + two_sqrt_a_alpha
        a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_omega)
        a2 = (a + 1.0) - (a - 1.0) * cos_omega - two_sqrt_a_alpha

    return (
        np.array([b0 / a0, b1 / a0, b2 / a0], dtype=float),
        np.array([1.0, a1 / a0, a2 / a0], dtype=float),
    )


def _apply_eq_fallback(
    audio_data: np.ndarray,
    sample_rate: int,
    bands: list[tuple[float, float, float]],
) -> np.ndarray:
    output = np.asarray(audio_data, dtype=np.float64).copy()
    for index, (frequency_hz, gain_db, q) in enumerate(bands):
        kind = "low_shelf" if index == 0 else "high_shelf" if index == NUM_EQ_BANDS - 1 else "peaking"
        b, a = _biquad_coefficients(kind, frequency_hz, gain_db, q, float(sample_rate))
        output = np.asarray(lfilter(b, a, output), dtype=np.float64)
    return np.asarray(output, dtype=np.float32)


def _true_peak_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    oversampled = resample_poly(np.asarray(samples, dtype=np.float64), 4, 1)
    return _db(float(np.max(np.abs(oversampled))) if oversampled.size else 0.0)


def _simulate_fallback(
    audio_data: np.ndarray,
    sample_rate: int,
    bands: list[tuple[float, float, float]],
    flat_settings: dict[str, Any],
) -> dict[str, Any]:
    input_audio = np.asarray(audio_data, dtype=np.float32)
    eq_output = _apply_eq_fallback(input_audio, sample_rate, bands)
    processed = eq_output.astype(np.float64, copy=True)

    compressor_gr = 0.0
    if flat_settings.get("compressor_enabled", True):
        rms_db = _db(float(np.sqrt(np.mean(np.square(processed)))) if processed.size else 0.0)
        threshold = _as_float(flat_settings.get("compressor_threshold_db"), -20.0)
        ratio = max(_as_float(flat_settings.get("compressor_ratio"), 4.0), 1.0)
        over_db = max(0.0, rms_db - threshold)
        compressor_gr = over_db * (1.0 - 1.0 / ratio)
        makeup = _as_float(flat_settings.get("compressor_makeup_gain_db"), 0.0)
        processed *= _linear(makeup - compressor_gr)

    careful = bool(flat_settings.get("limiter_careful_output_enabled", True))
    ceiling_db = _as_float(flat_settings.get("limiter_ceiling_db"), -0.5)
    effective_ceiling_db = min(ceiling_db, -1.5) if careful else min(ceiling_db, 0.0)
    pre_true_peak_db = _true_peak_db(processed.astype(np.float32, copy=False))
    limiter_gr = 0.0
    true_peak_gr = 0.0
    limited_events = 0
    if flat_settings.get("limiter_enabled", True) and pre_true_peak_db > effective_ceiling_db:
        true_peak_gr = pre_true_peak_db - effective_ceiling_db
        limiter_gr = max(0.0, _db(float(np.max(np.abs(processed)))) - effective_ceiling_db)
        limited_events = 1
        processed *= _linear(-true_peak_gr)
        ceiling = _linear(effective_ceiling_db)
        processed = np.clip(processed, -ceiling, ceiling)

    output_sample_peak_db = _db(float(np.max(np.abs(processed))) if processed.size else 0.0)
    output_true_peak_db = _true_peak_db(processed.astype(np.float32, copy=False))

    return {
        "input_sample_peak_db": _db(float(np.max(np.abs(input_audio))) if input_audio.size else 0.0),
        "input_rms_db": _db(float(np.sqrt(np.mean(np.square(input_audio)))) if input_audio.size else 0.0),
        "output_sample_peak_db": output_sample_peak_db,
        "pre_limiter_true_peak_db": pre_true_peak_db,
        "output_true_peak_db": output_true_peak_db,
        "output_rms_db": _db(float(np.sqrt(np.mean(np.square(processed)))) if processed.size else 0.0),
        "limiter_effective_ceiling_db": effective_ceiling_db,
        "sample_headroom_db": effective_ceiling_db - output_sample_peak_db,
        "pre_limiter_true_peak_headroom_db": effective_ceiling_db - pre_true_peak_db,
        "true_peak_headroom_db": effective_ceiling_db - output_true_peak_db,
        "limiter_gain_reduction_db": limiter_gr,
        "true_peak_limiter_gain_reduction_db": true_peak_gr,
        "true_peak_limited_events": limited_events,
        "compressor_gain_reduction_db": compressor_gr,
        "deesser_gain_reduction_db": 0.0,
        "processed_samples": int(processed.size),
    }


def simulate_candidate_chain(
    audio_data: np.ndarray,
    sample_rate: int,
    eq_settings: dict[str, Any],
    chain_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate the deterministic downstream chain for a candidate EQ."""

    bands = _bands_from_settings(eq_settings)
    flat_settings = _flatten_chain_settings(chain_settings)
    native = _native_simulate(audio_data, sample_rate, bands, flat_settings)
    if native is not None:
        native["simulation_backend"] = "rust"
        native["safety_authority"] = "authoritative"
        return native

    fallback = _simulate_fallback(audio_data, sample_rate, bands, flat_settings)
    fallback["simulation_backend"] = "python"
    fallback["safety_authority"] = "advisory"
    fallback["limitations"] = [
        "de-esser behavior is not simulated",
        "compression uses whole-capture RMS instead of the live envelope",
        "the live lookahead limiter is not simulated",
    ]
    return fallback


def _is_headroom_safe(simulation: dict[str, Any]) -> bool:
    pre_true_peak_headroom = _as_float(
        simulation.get("pre_limiter_true_peak_headroom_db"),
        simulation.get("true_peak_headroom_db", 120.0),
    )
    limiter_gr = _as_float(simulation.get("limiter_gain_reduction_db"), 0.0)
    true_peak_gr = _as_float(simulation.get("true_peak_limiter_gain_reduction_db"), 0.0)
    return (
        pre_true_peak_headroom >= HEADROOM_TARGET_DB
        and limiter_gr <= LIMITER_GAIN_REDUCTION_WARN_DB
        and true_peak_gr <= TRUE_PEAK_GAIN_REDUCTION_WARN_DB
    )


def apply_headroom_validation(
    audio_data: np.ndarray,
    sample_rate: int,
    eq_settings: dict[str, Any],
    chain_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Scale Auto-EQ boosts/cuts when offline chain simulation predicts headroom risk."""

    audio = np.asarray(audio_data, dtype=np.float32)
    result = deepcopy(eq_settings)
    original_gains = np.asarray(result.get("band_gains", []), dtype=float)
    if original_gains.size != NUM_EQ_BANDS:
        return result

    before = simulate_candidate_chain(audio, sample_rate, result, chain_settings)
    selected = before
    selected_scale = 1.0
    selected_gains = original_gains.copy()
    if not _is_headroom_safe(before):
        for scale in HEADROOM_SCALES[1:]:
            candidate = deepcopy(result)
            candidate_gains = (original_gains * scale).tolist()
            candidate["band_gains"] = candidate_gains
            simulation = simulate_candidate_chain(audio, sample_rate, candidate, chain_settings)
            selected = simulation
            selected_scale = scale
            selected_gains = np.asarray(candidate_gains, dtype=float)
            if _is_headroom_safe(simulation):
                break

    result["band_gains"] = selected_gains.tolist()
    existing_scale = _as_float(result.get("validation_gain_scale"), 1.0)
    result["validation_gain_scale"] = float(existing_scale * selected_scale)

    meets_thresholds = _is_headroom_safe(selected)
    authoritative = selected.get("simulation_backend") == "rust"
    safe = bool(meets_thresholds and authoritative)
    if not safe:
        result["validation_confidence"] = float(
            min(_as_float(result.get("validation_confidence"), 1.0), 0.42)
        )
        result["analysis_confidence"] = float(
            min(_as_float(result.get("analysis_confidence"), 1.0), 0.58)
        )
    elif selected_scale < 1.0:
        result["validation_confidence"] = float(
            min(_as_float(result.get("validation_confidence"), 1.0), 0.72)
        )

    result["headroom_validation"] = {
        "safe": safe,
        "authoritative": authoritative,
        "advisory": not authoritative,
        "meets_advisory_thresholds": meets_thresholds,
        "gain_scale": selected_scale,
        "before": before,
        "after": selected,
        "status": "safe" if safe else "risk" if authoritative else "advisory",
    }
    result["headroom_safe"] = safe
    result["headroom_advisory"] = not authoritative
    result["headroom_gain_scale"] = selected_scale
    return result
