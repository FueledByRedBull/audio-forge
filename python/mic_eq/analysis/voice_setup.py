"""Heuristic analysis for the Auto Voice Setup wizard."""

from __future__ import annotations

from typing import Any

import numpy as np

from .auto_eq import analyze_auto_eq
from .spectrum import analyze_voice_spectrum, smooth_spectrum_perceptual

NOISE_MIN_DURATION_S = 1.0
SPEECH_MIN_DURATION_S = 3.0
FRAME_MS = 40.0
HOP_MS = 20.0

GATE_MODE_LABELS = {
    0: "Threshold Only",
    1: "VAD Assisted",
    2: "VAD Only",
}

TARGET_LUFS_BY_CURVE = {
    "broadcast": -16.0,
    "streaming": -16.0,
    "podcast": -17.0,
    "flat": -18.0,
}


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _rms_db(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=float)
    if audio.size == 0:
        return -120.0
    return float(20.0 * np.log10(np.sqrt(np.mean(audio * audio)) + 1e-9))


def _peak_db(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=float)
    if audio.size == 0:
        return -120.0
    return float(20.0 * np.log10(np.max(np.abs(audio)) + 1e-9))


def _frame_rms_db(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_size = max(256, int(sample_rate * FRAME_MS / 1000.0))
    hop_size = max(128, int(sample_rate * HOP_MS / 1000.0))
    if audio.size < frame_size:
        return np.asarray([_rms_db(audio)], dtype=float)

    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_size)[::hop_size]
    frame_power = np.mean(frames * frames, axis=1)
    return 10.0 * np.log10(frame_power + 1e-12)


def _band_mean(freqs: np.ndarray, spectrum_db: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return float(np.mean(spectrum_db))
    return float(np.mean(spectrum_db[mask]))


def _recommend_gate_settings(
    *,
    vad_available: bool,
    noise_rms_db: float,
    speech_floor_db: float,
    speech_body_db: float,
    speech_snr_db: float,
    speech_dynamic_range_db: float,
) -> dict[str, Any]:
    margin_db = _clamp(speech_floor_db - noise_rms_db - 3.0, 4.0, 12.0)
    threshold_db = _clamp(noise_rms_db + margin_db, -80.0, -10.0)
    vad_threshold = _clamp(0.54 - (speech_snr_db - 10.0) / 30.0, 0.34, 0.58)
    quietness_gap_db = max(0.0, -22.0 - speech_body_db)
    vad_pre_gain = _clamp(10.0 ** (quietness_gap_db / 20.0), 1.0, 3.0)
    vad_hold_time_ms = _clamp(140.0 + speech_dynamic_range_db * 6.0, 140.0, 260.0)

    # Prefer VAD-assisted gating over VAD-only. It is safer when speech is soft
    # or the backend becomes temporarily unavailable.
    gate_mode = 1 if vad_available else 0

    return {
        "enabled": True,
        "threshold_db": threshold_db,
        "attack_ms": 5.0,
        "release_ms": 120.0,
        "gate_mode": gate_mode,
        "vad_threshold": vad_threshold,
        "vad_hold_time_ms": vad_hold_time_ms,
        "vad_pre_gain": vad_pre_gain,
        "auto_threshold_enabled": bool(vad_available),
        "gate_margin_db": margin_db,
    }


def _recommend_deesser_settings(
    *,
    freqs: np.ndarray,
    spectrum_db: np.ndarray,
    capture_confidence: float,
) -> tuple[dict[str, Any], dict[str, float | bool]]:
    presence_db = _band_mean(freqs, spectrum_db, 2500.0, 4500.0)
    sibilance_db = _band_mean(freqs, spectrum_db, 5000.0, 9000.0)
    sib_mask = (freqs >= 4500.0) & (freqs <= 9500.0)
    if np.any(sib_mask):
        sib_freqs = freqs[sib_mask]
        sib_spec = spectrum_db[sib_mask]
        peak_index = int(np.argmax(sib_spec))
        peak_hz = float(sib_freqs[peak_index])
        sibilance_peak_db = float(sib_spec[peak_index])
    else:
        peak_hz = 6500.0
        sibilance_peak_db = sibilance_db

    sibilance_excess_db = sibilance_db - presence_db
    peak_prominence_db = sibilance_peak_db - sibilance_db
    enabled = bool(
        capture_confidence >= 0.40
        and (sibilance_excess_db >= 1.5 or peak_prominence_db >= 2.5)
    )
    auto_amount = _clamp((sibilance_excess_db + 1.5) / 8.0, 0.25, 0.85)
    low_cut_hz = _clamp(peak_hz - 1700.0, 3500.0, 7000.0)
    high_cut_hz = _clamp(peak_hz + 2100.0, low_cut_hz + 1500.0, 11000.0)
    ratio = _clamp(2.5 + max(0.0, sibilance_excess_db) * 0.45, 2.0, 5.5)
    max_reduction_db = _clamp(3.5 + max(0.0, sibilance_excess_db) * 0.65, 3.0, 8.0)

    settings = {
        "enabled": enabled,
        "auto_enabled": True,
        "auto_amount": auto_amount,
        "low_cut_hz": low_cut_hz,
        "high_cut_hz": high_cut_hz,
        "threshold_db": -28.0,
        "ratio": ratio,
        "attack_ms": 2.0,
        "release_ms": 80.0,
        "max_reduction_db": max_reduction_db,
    }
    diagnostics = {
        "enabled": enabled,
        "sibilance_excess_db": float(sibilance_excess_db),
        "peak_hz": peak_hz,
    }
    return settings, diagnostics


def _recommend_compressor_settings(
    *,
    target_preset: str,
    speech_body_db: float,
    speech_dynamic_range_db: float,
    speech_snr_db: float,
    capture_confidence: float,
) -> tuple[dict[str, Any], dict[str, float | bool]]:
    target_lufs = TARGET_LUFS_BY_CURVE.get(target_preset, -18.0)
    threshold_db = _clamp(speech_body_db - 5.5, -30.0, -14.0)
    ratio = _clamp(2.5 + (speech_dynamic_range_db - 10.0) / 6.0, 2.2, 4.5)
    attack_ms = _clamp(10.0 - (speech_dynamic_range_db - 8.0) / 4.0, 4.0, 12.0)
    release_ms = _clamp(140.0 + speech_dynamic_range_db * 7.0, 120.0, 260.0)
    base_release_ms = _clamp(55.0 + speech_dynamic_range_db * 4.0, 50.0, 140.0)
    auto_makeup_enabled = bool(capture_confidence >= 0.55 and speech_snr_db >= 10.0)
    makeup_gain_db = 0.0
    if not auto_makeup_enabled:
        makeup_gain_db = _clamp(target_lufs - speech_body_db - 8.0, 0.0, 6.0)

    settings = {
        "enabled": True,
        "threshold_db": threshold_db,
        "ratio": ratio,
        "attack_ms": attack_ms,
        "release_ms": release_ms,
        "makeup_gain_db": makeup_gain_db,
        "adaptive_release": True,
        "base_release_ms": base_release_ms,
        "auto_makeup_enabled": auto_makeup_enabled,
        "target_lufs": target_lufs,
    }
    diagnostics = {
        "auto_makeup_enabled": auto_makeup_enabled,
        "target_lufs": target_lufs,
    }
    return settings, diagnostics


def analyze_voice_setup(
    noise_audio: np.ndarray,
    speech_audio: np.ndarray,
    sample_rate: int,
    target_preset: str = "broadcast",
    *,
    vad_available: bool = True,
) -> dict[str, Any]:
    """Analyze room noise plus speech and recommend a full voice chain."""
    noise_arr = np.asarray(noise_audio, dtype=float)
    speech_arr = np.asarray(speech_audio, dtype=float)

    if noise_arr.size < int(sample_rate * NOISE_MIN_DURATION_S):
        raise ValueError("Room-noise capture was too short for setup.")
    if speech_arr.size < int(sample_rate * SPEECH_MIN_DURATION_S):
        raise ValueError("Voice capture was too short for setup.")

    noise_rms_db = _rms_db(noise_arr)
    noise_peak_db = _peak_db(noise_arr)
    speech_rms_db = _rms_db(speech_arr)
    speech_peak_db = _peak_db(speech_arr)
    frame_rms = _frame_rms_db(speech_arr, sample_rate)

    speech_gate_db = max(noise_rms_db + 3.0, float(np.percentile(frame_rms, 35.0)))
    active_frames = frame_rms[frame_rms >= speech_gate_db]
    if active_frames.size < 6:
        active_frames = frame_rms

    speech_floor_db = float(np.percentile(active_frames, 20.0))
    speech_body_db = float(np.percentile(active_frames, 60.0))
    speech_frame_peak_db = float(np.percentile(active_frames, 95.0))
    speech_dynamic_range_db = max(0.0, speech_frame_peak_db - speech_floor_db)
    speech_snr_db = speech_body_db - noise_rms_db

    spectrum_result = analyze_voice_spectrum(speech_arr, sample_rate)
    smoothed_spectrum = smooth_spectrum_perceptual(
        spectrum_result.freqs,
        spectrum_result.median_spectrum_db,
    )
    capture_confidence = float(spectrum_result.residual_confidence)

    gate_settings = _recommend_gate_settings(
        vad_available=vad_available,
        noise_rms_db=noise_rms_db,
        speech_floor_db=speech_floor_db,
        speech_body_db=speech_body_db,
        speech_snr_db=speech_snr_db,
        speech_dynamic_range_db=speech_dynamic_range_db,
    )
    deesser_settings, deesser_diag = _recommend_deesser_settings(
        freqs=spectrum_result.freqs,
        spectrum_db=smoothed_spectrum,
        capture_confidence=capture_confidence,
    )
    compressor_settings, compressor_diag = _recommend_compressor_settings(
        target_preset=target_preset,
        speech_body_db=speech_body_db,
        speech_dynamic_range_db=speech_dynamic_range_db,
        speech_snr_db=speech_snr_db,
        capture_confidence=capture_confidence,
    )

    eq_settings: dict[str, Any] | None = None
    eq_error: str | None = None
    try:
        eq_settings, _validation = analyze_auto_eq(speech_arr, sample_rate, target_preset)
    except Exception as exc:  # pragma: no cover - exercised through return shape
        eq_error = str(exc)

    snr_confidence = _clamp((speech_snr_db - 6.0) / 12.0, 0.0, 1.0)
    dynamics_confidence = _clamp((speech_dynamic_range_db - 8.0) / 10.0, 0.0, 1.0)
    quiet_room_confidence = _clamp((-32.0 - noise_rms_db) / 18.0, 0.0, 1.0)
    eq_confidence = float(eq_settings.get("analysis_confidence", capture_confidence)) if eq_settings else capture_confidence
    gate_confidence = float(np.clip(0.55 * capture_confidence + 0.45 * snr_confidence, 0.0, 1.0))
    deesser_confidence = float(
        np.clip(
            0.60 * capture_confidence
            + 0.40 * (0.8 if not deesser_diag["enabled"] else _clamp((deesser_diag["sibilance_excess_db"] + 1.0) / 5.0, 0.25, 1.0)),
            0.0,
            1.0,
        )
    )
    compressor_confidence = float(np.clip(0.55 * capture_confidence + 0.45 * dynamics_confidence, 0.0, 1.0))
    setup_confidence = float(
        np.clip(
            0.35 * eq_confidence
            + 0.25 * gate_confidence
            + 0.15 * deesser_confidence
            + 0.15 * compressor_confidence
            + 0.10 * quiet_room_confidence,
            0.0,
            1.0,
        )
    )

    return {
        "eq_settings": eq_settings,
        "eq_error": eq_error,
        "gate_settings": gate_settings,
        "deesser_settings": deesser_settings,
        "compressor_settings": compressor_settings,
        "diagnostics": {
            "setup_confidence": setup_confidence,
            "capture_confidence": capture_confidence,
            "eq_confidence": eq_confidence,
            "gate_confidence": gate_confidence,
            "deesser_confidence": deesser_confidence,
            "compressor_confidence": compressor_confidence,
            "noise_rms_db": noise_rms_db,
            "noise_peak_db": noise_peak_db,
            "speech_rms_db": speech_rms_db,
            "speech_peak_db": speech_peak_db,
            "speech_floor_db": speech_floor_db,
            "speech_body_db": speech_body_db,
            "speech_dynamic_range_db": speech_dynamic_range_db,
            "speech_snr_db": speech_snr_db,
            "gate_mode_label": GATE_MODE_LABELS[gate_settings["gate_mode"]],
            "sibilance_excess_db": deesser_diag["sibilance_excess_db"],
            "sibilance_peak_hz": deesser_diag["peak_hz"],
            "deesser_enabled": deesser_diag["enabled"],
            "compressor_auto_makeup_enabled": compressor_diag["auto_makeup_enabled"],
            "compressor_target_lufs": compressor_diag["target_lufs"],
            "vad_available": bool(vad_available),
        },
    }


__all__ = ["GATE_MODE_LABELS", "analyze_voice_setup"]
