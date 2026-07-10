"""Uncertainty-aware Auto Voice Setup analysis and native-chain validation.

Recommendations use energy-VAD-masked speech, BS.1770 K-weighted short-term
loudness, loudness range, and robust band-energy summaries. Candidate settings
are checked through the offline DSP simulator before the UI offers to apply them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import lfilter, resample_poly

from .auto_eq import analyze_auto_eq, simulate_candidate_chain
from .spectrum import analyze_voice_spectrum, smooth_spectrum_perceptual
from ..config import EQ_FREQUENCIES

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


def _k_weighted_48k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return the BS.1770 K-weighted signal at 48 kHz."""
    signal = np.asarray(audio, dtype=np.float64)
    if sample_rate != 48_000:
        divisor = int(np.gcd(sample_rate, 48_000))
        signal = resample_poly(signal, 48_000 // divisor, sample_rate // divisor)
    shelf_b = np.asarray([1.53512485958697, -2.69169618940638, 1.19839281085285])
    shelf_a = np.asarray([1.0, -1.69065929318241, 0.73248077421585])
    highpass_b = np.asarray([1.0, -2.0, 1.0])
    highpass_a = np.asarray([1.0, -1.99004745483398, 0.99007225036621])
    return np.asarray(
        lfilter(highpass_b, highpass_a, lfilter(shelf_b, shelf_a, signal)),
        dtype=np.float64,
    )


def _vad_masked_speech_features(
    speech: np.ndarray,
    sample_rate: int,
    noise_rms_db: float,
) -> dict[str, Any]:
    """Extract energy-VAD-masked loudness, range, and robust band features."""
    signal = np.asarray(speech, dtype=np.float64)
    frame_size = max(256, int(sample_rate * FRAME_MS / 1000.0))
    hop_size = max(128, int(sample_rate * HOP_MS / 1000.0))
    if signal.size < frame_size:
        frames = signal[np.newaxis, :]
        starts = np.asarray([0], dtype=int)
    else:
        frames = np.lib.stride_tricks.sliding_window_view(signal, frame_size)[::hop_size]
        starts = np.arange(frames.shape[0], dtype=int) * hop_size
    frame_power = np.mean(frames * frames, axis=1)
    frame_db = 10.0 * np.log10(frame_power + 1e-12)
    adaptive_floor = max(noise_rms_db + 6.0, float(np.percentile(frame_db, 30.0)) + 2.0)
    active_frames = frame_db >= adaptive_floor
    if active_frames.size >= 3:
        active_frames = np.convolve(active_frames.astype(int), np.ones(3, dtype=int), mode="same") > 0

    sample_mask = np.zeros(signal.size, dtype=bool)
    for start, active in zip(starts, active_frames):
        if active:
            sample_mask[start : min(signal.size, start + frame_size)] = True
    active_duration_s = float(np.count_nonzero(sample_mask) / max(sample_rate, 1))
    active_ratio = float(np.mean(sample_mask)) if sample_mask.size else 0.0

    weighted = _k_weighted_48k(signal, sample_rate)
    if sample_rate == 48_000:
        weighted_mask = sample_mask
    else:
        divisor = int(np.gcd(sample_rate, 48_000))
        weighted_mask = (
            resample_poly(
                sample_mask.astype(np.float64),
                48_000 // divisor,
                sample_rate // divisor,
            )
            >= 0.5
        )
    if weighted_mask.size < weighted.size:
        weighted_mask = np.pad(weighted_mask, (0, weighted.size - weighted_mask.size))
    elif weighted_mask.size > weighted.size:
        weighted_mask = weighted_mask[: weighted.size]
    short_size = int(0.400 * 48_000)
    short_hop = int(0.100 * 48_000)
    loudness_values: list[float] = []
    if weighted.size >= short_size:
        for start in range(0, weighted.size - short_size + 1, short_hop):
            stop = start + short_size
            if float(np.mean(weighted_mask[start:stop])) < 0.55:
                continue
            mean_square = float(np.mean(np.square(weighted[start:stop])))
            loudness_values.append(float(-0.691 + 10.0 * np.log10(mean_square + 1e-12)))
    if not loudness_values:
        active_weighted = weighted[weighted_mask]
        mean_square = float(np.mean(np.square(active_weighted))) if active_weighted.size else 0.0
        loudness_values = [float(-0.691 + 10.0 * np.log10(mean_square + 1e-12))]
    loudness = np.asarray(loudness_values, dtype=float)
    short_term_lufs = float(np.median(loudness))
    loudness_range_db = (
        float(np.percentile(loudness, 95.0) - np.percentile(loudness, 10.0))
        if loudness.size >= 4
        else 0.0
    )

    window = np.hanning(frame_size)
    frequencies = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
    active_indices = np.flatnonzero(active_frames)
    band_ranges = {
        "low": (80.0, 250.0),
        "body": (250.0, 2000.0),
        "presence": (2000.0, 5000.0),
        "sibilance": (5000.0, min(10_000.0, sample_rate * 0.45)),
    }
    band_rows: dict[str, list[float]] = {name: [] for name in band_ranges}
    for frame_index in active_indices:
        power = np.square(np.abs(np.fft.rfft(frames[frame_index] * window))) + 1e-18
        for name, (low_hz, high_hz) in band_ranges.items():
            mask = (frequencies >= low_hz) & (frequencies <= high_hz)
            band_rows[name].append(float(10.0 * np.log10(np.sum(power[mask]) + 1e-18)))
    robust_bands = {
        name: float(np.median(values)) if values else -120.0
        for name, values in band_rows.items()
    }

    return {
        "frame_db": frame_db,
        "active_frame_mask": active_frames,
        "active_duration_s": active_duration_s,
        "active_ratio": active_ratio,
        "short_term_lufs": short_term_lufs,
        "loudness_range_db": loudness_range_db,
        "loudness_window_count": int(loudness.size),
        "band_energy_db": robust_bands,
        "sibilance_excess_db": robust_bands["sibilance"] - robust_bands["presence"],
    }


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
    robust_sibilance_excess_db: float | None = None,
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

    spectral_excess_db = sibilance_db - presence_db
    sibilance_excess_db = (
        spectral_excess_db
        if robust_sibilance_excess_db is None
        else float(0.35 * spectral_excess_db + 0.65 * robust_sibilance_excess_db)
    )
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
    speech_loudness_lufs: float,
    loudness_range_db: float,
    speech_snr_db: float,
    capture_confidence: float,
) -> tuple[dict[str, Any], dict[str, float | bool]]:
    target_lufs = TARGET_LUFS_BY_CURVE.get(target_preset, -18.0)
    threshold_db = _clamp(speech_body_db - 5.5, -30.0, -14.0)
    ratio = _clamp(2.2 + loudness_range_db / 5.0, 2.2, 4.5)
    attack_ms = _clamp(11.0 - loudness_range_db / 2.5, 4.0, 12.0)
    release_ms = _clamp(135.0 + loudness_range_db * 11.0, 120.0, 260.0)
    base_release_ms = _clamp(50.0 + loudness_range_db * 6.0, 50.0, 140.0)
    auto_makeup_enabled = bool(capture_confidence >= 0.55 and speech_snr_db >= 10.0)
    makeup_gain_db = 0.0
    if not auto_makeup_enabled:
        makeup_gain_db = _clamp(target_lufs - speech_loudness_lufs, 0.0, 6.0)

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
        "measured_short_term_lufs": speech_loudness_lufs,
        "measured_loudness_range_db": loudness_range_db,
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
    features = _vad_masked_speech_features(speech_arr, sample_rate, noise_rms_db)
    frame_rms = np.asarray(features["frame_db"], dtype=float)
    active_frames = frame_rms[np.asarray(features["active_frame_mask"], dtype=bool)]
    if active_frames.size < 6:
        active_frames = frame_rms

    speech_floor_db = float(np.percentile(active_frames, 20.0))
    speech_body_db = float(np.percentile(active_frames, 60.0))
    speech_frame_peak_db = float(np.percentile(active_frames, 95.0))
    frame_dynamic_range_db = max(0.0, speech_frame_peak_db - speech_floor_db)
    speech_dynamic_range_db = float(features["loudness_range_db"])
    speech_snr_db = speech_body_db - noise_rms_db

    spectrum_result = analyze_voice_spectrum(speech_arr, sample_rate)
    smoothed_spectrum = smooth_spectrum_perceptual(
        spectrum_result.freqs,
        spectrum_result.median_spectrum_db,
    )
    spectral_confidence = float(spectrum_result.residual_confidence)
    snr_confidence = _clamp((speech_snr_db - 6.0) / 12.0, 0.0, 1.0)
    active_duration_confidence = _clamp(float(features["active_duration_s"]) / 3.0, 0.0, 1.0)
    loudness_confidence = _clamp(float(features["loudness_window_count"]) / 8.0, 0.0, 1.0)
    capture_confidence = float(
        np.clip(
            0.45 * spectral_confidence
            + 0.25 * snr_confidence
            + 0.20 * active_duration_confidence
            + 0.10 * loudness_confidence,
            0.0,
            1.0,
        )
    )

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
        robust_sibilance_excess_db=float(features["sibilance_excess_db"]),
    )
    compressor_settings, compressor_diag = _recommend_compressor_settings(
        target_preset=target_preset,
        speech_body_db=speech_body_db,
        speech_loudness_lufs=float(features["short_term_lufs"]),
        loudness_range_db=speech_dynamic_range_db,
        speech_snr_db=speech_snr_db,
        capture_confidence=capture_confidence,
    )

    eq_settings: dict[str, Any] | None = None
    eq_error: str | None = None
    try:
        eq_settings, _validation = analyze_auto_eq(speech_arr, sample_rate, target_preset)
    except Exception as exc:  # pragma: no cover - exercised through return shape
        eq_error = str(exc)

    dynamics_confidence = _clamp(speech_dynamic_range_db / 8.0, 0.0, 1.0)
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

    offline_validation: dict[str, Any] | None = None
    offline_validation_passed = False
    simulation_eq_settings = eq_settings or {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [0.0] * len(EQ_FREQUENCIES),
        "band_qs": [1.41] * len(EQ_FREQUENCIES),
    }
    try:
        offline_validation = simulate_candidate_chain(
            speech_arr.astype(np.float32, copy=False),
            sample_rate,
            simulation_eq_settings,
            {
                "deesser": deesser_settings,
                "compressor": compressor_settings,
                "limiter": {
                    "enabled": True,
                    "ceiling_db": -1.5,
                    "release_ms": 80.0,
                    "careful_output_enabled": True,
                },
            },
        )
        output_true_peak = float(offline_validation.get("output_true_peak_db", 120.0))
        ceiling = float(offline_validation.get("limiter_effective_ceiling_db", -1.5))
        compressor_gr = float(offline_validation.get("compressor_gain_reduction_db", 120.0))
        deesser_gr = float(offline_validation.get("deesser_gain_reduction_db", 120.0))
        offline_validation_passed = bool(
            np.isfinite([output_true_peak, compressor_gr, deesser_gr]).all()
            and output_true_peak <= ceiling + 0.15
            and compressor_gr <= 12.0
            and deesser_gr <= 10.0
        )
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        offline_validation = {"error": str(exc), "simulation_backend": "unavailable"}

    uncertainty_reasons: list[str] = []
    if float(features["active_duration_s"]) < 2.0:
        uncertainty_reasons.append("too little VAD-active speech")
    if speech_snr_db < 8.0:
        uncertainty_reasons.append("speech-to-noise ratio is weak")
    if capture_confidence < 0.50:
        uncertainty_reasons.append("spectral feature stability is weak")
    if not offline_validation_passed:
        uncertainty_reasons.append("offline DSP validation did not pass")
    if offline_validation and offline_validation.get("simulation_backend") != "rust":
        uncertainty_reasons.append("offline DSP validation is advisory without the Rust extension")
        setup_confidence *= 0.90
    weak_capture = bool(
        float(features["active_duration_s"]) < 2.0
        or speech_snr_db < 8.0
        or capture_confidence < 0.50
    )
    apply_recommended = bool(not weak_capture and offline_validation_passed)
    if weak_capture:
        setup_confidence = min(setup_confidence, 0.49)
    setup_confidence = float(np.clip(setup_confidence, 0.0, 1.0))

    return {
        "eq_settings": eq_settings,
        "eq_error": eq_error,
        "gate_settings": gate_settings,
        "deesser_settings": deesser_settings,
        "compressor_settings": compressor_settings,
        "diagnostics": {
            "setup_confidence": setup_confidence,
            "recommendation_uncertainty": 1.0 - setup_confidence,
            "uncertainty_reasons": uncertainty_reasons,
            "weak_capture": weak_capture,
            "apply_recommended": apply_recommended,
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
            "speech_frame_dynamic_range_db": frame_dynamic_range_db,
            "speech_snr_db": speech_snr_db,
            "vad_active_duration_s": features["active_duration_s"],
            "vad_active_ratio": features["active_ratio"],
            "short_term_lufs": features["short_term_lufs"],
            "loudness_range_db": features["loudness_range_db"],
            "robust_band_energy_db": features["band_energy_db"],
            "gate_mode_label": GATE_MODE_LABELS[gate_settings["gate_mode"]],
            "sibilance_excess_db": deesser_diag["sibilance_excess_db"],
            "sibilance_peak_hz": deesser_diag["peak_hz"],
            "deesser_enabled": deesser_diag["enabled"],
            "compressor_auto_makeup_enabled": compressor_diag["auto_makeup_enabled"],
            "compressor_target_lufs": compressor_diag["target_lufs"],
            "vad_available": bool(vad_available),
            "offline_validation_passed": offline_validation_passed,
            "offline_validation": offline_validation,
        },
    }


__all__ = ["GATE_MODE_LABELS", "analyze_voice_setup"]
