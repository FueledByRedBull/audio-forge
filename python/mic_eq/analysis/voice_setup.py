"""Uncertainty-aware Auto Voice Setup analysis and native-chain validation.

Recommendations use energy-VAD-masked speech, BS.1770 K-weighted momentary and
three-second short-term loudness, active loudness spread, and robust band-energy
summaries. Candidate settings are checked through the offline DSP simulator
before the UI offers to apply them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import lfilter, resample_poly

from .auto_eq import analyze_auto_eq, simulate_candidate_chain
from .spectrum import (
    _interpolate_vad_probabilities,
    analyze_voice_spectrum,
    smooth_spectrum_perceptual,
)
from .vad import analyze_offline_vad
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


def _bounded_quality_score(
    components: list[tuple[float, float]],
) -> float:
    """Combine bounded quality evidence without letting one metric dominate."""
    if not components:
        return 0.0
    values = np.asarray([np.clip(value, 0.0, 1.0) for value, _weight in components])
    weights = np.asarray([max(0.0, weight) for _value, weight in components])
    if float(np.sum(weights)) <= 0.0:
        return 0.0
    weights /= float(np.sum(weights))
    # A weighted geometric mean makes a genuinely weak prerequisite visible,
    # unlike an arithmetic score that can hide it behind unrelated strengths.
    return float(np.exp(np.sum(weights * np.log(np.maximum(values, 0.03)))))


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


def _active_loudness_windows(
    weighted: np.ndarray,
    active_mask: np.ndarray,
    *,
    window_samples: int,
    hop_samples: int,
) -> np.ndarray:
    values: list[float] = []
    if weighted.size >= window_samples:
        for start in range(0, weighted.size - window_samples + 1, hop_samples):
            stop = start + window_samples
            if float(np.mean(active_mask[start:stop])) < 0.55:
                continue
            mean_square = float(np.mean(np.square(weighted[start:stop])))
            values.append(float(-0.691 + 10.0 * np.log10(mean_square + 1e-12)))
    return np.asarray(values, dtype=float)


def _vad_masked_speech_features(
    speech: np.ndarray,
    sample_rate: int,
    noise_rms_db: float,
    vad_probabilities: np.ndarray | None = None,
    noise_audio: np.ndarray | None = None,
) -> dict[str, Any]:
    """Extract posterior/energy-masked loudness, range, and band features."""
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
    energy_active_frames = frame_db >= adaptive_floor
    frame_starts = np.arange(frame_db.size, dtype=int) * hop_size
    frame_vad_probabilities = _interpolate_vad_probabilities(
        vad_probabilities,
        frame_starts,
        frame_size,
        sample_rate,
    )
    active_frames = energy_active_frames
    if frame_vad_probabilities is not None:
        supported_energy = frame_db >= max(noise_rms_db + 2.0, adaptive_floor - 4.0)
        posterior_active = (
            ((frame_vad_probabilities >= 0.35) & supported_energy)
            | (frame_vad_probabilities >= 0.65)
        )
        # Require enough posterior-supported material to replace the energy
        # mask; otherwise a failed/under-confident model must not erase a
        # usable capture.
        if int(np.count_nonzero(posterior_active)) >= 6:
            active_frames = posterior_active
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
    momentary_loudness = _active_loudness_windows(
        weighted,
        weighted_mask,
        window_samples=int(0.400 * 48_000),
        hop_samples=int(0.100 * 48_000),
    )
    short_term_loudness = _active_loudness_windows(
        weighted,
        weighted_mask,
        window_samples=int(3.000 * 48_000),
        hop_samples=int(1.000 * 48_000),
    )
    if momentary_loudness.size == 0:
        active_weighted = weighted[weighted_mask]
        mean_square = float(np.mean(np.square(active_weighted))) if active_weighted.size else 0.0
        momentary_loudness = np.asarray(
            [float(-0.691 + 10.0 * np.log10(mean_square + 1e-12))],
            dtype=float,
        )
    if short_term_loudness.size == 0:
        short_term_loudness = momentary_loudness
    momentary_lufs = float(np.median(momentary_loudness))
    short_term_lufs = float(np.median(short_term_loudness))
    active_loudness_spread_db = (
        float(
            np.percentile(momentary_loudness, 95.0)
            - np.percentile(momentary_loudness, 10.0)
        )
        if momentary_loudness.size >= 4
        else 0.0
    )

    window = np.hanning(frame_size)
    frequencies = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
    # Silero correctly treats many sibilants as unvoiced and can assign them a
    # low speech posterior. Keep energy-supported unvoiced frames in the
    # spectral/de-esser analysis while retaining the stricter posterior mask
    # for loudness and speech-duration measurements.
    spectral_active_frames = active_frames | energy_active_frames
    active_indices = np.flatnonzero(spectral_active_frames)
    band_ranges = {
        "low": (80.0, 250.0),
        "body": (250.0, 2000.0),
        "presence": (2000.0, 5000.0),
        "sibilance": (5000.0, min(10_000.0, sample_rate * 0.45)),
    }
    band_rows: dict[str, list[float]] = {name: [] for name in band_ranges}
    active_power_rows: list[np.ndarray] = []
    for frame_index in active_indices:
        frame = frames[frame_index] - float(np.mean(frames[frame_index]))
        power = np.square(np.abs(np.fft.rfft(frame * window))) + 1e-18
        active_power_rows.append(power)
        for name, (low_hz, high_hz) in band_ranges.items():
            mask = (frequencies >= low_hz) & (frequencies <= high_hz)
            band_rows[name].append(float(10.0 * np.log10(np.sum(power[mask]) + 1e-18)))
    robust_bands = {
        name: float(np.median(values)) if values else -120.0
        for name, values in band_rows.items()
    }

    deesser_evidence: dict[str, float | bool] = {
        "available": False,
        "confidence": 0.0,
        "excess_p90_db": -120.0,
        "temporal_contrast_db": 0.0,
        "candidate_frame_ratio": 0.0,
        "candidate_snr_db": 0.0,
        "peak_hz": 6500.0,
    }
    if active_power_rows:
        active_power = np.asarray(active_power_rows, dtype=float)
        voice_reference_mask = (frequencies >= 250.0) & (frequencies <= 4500.0)
        sibilance_mask = (frequencies >= 5000.0) & (
            frequencies <= min(9500.0, sample_rate * 0.45)
        )
        if np.any(voice_reference_mask) and np.any(sibilance_mask):
            voice_reference_rows = 10.0 * np.log10(
                np.sum(active_power[:, voice_reference_mask], axis=1) + 1e-18
            )
            sibilance_rows = 10.0 * np.log10(
                np.sum(active_power[:, sibilance_mask], axis=1) + 1e-18
            )
            excess_rows = sibilance_rows - voice_reference_rows
            noise_sibilance_db = float(np.percentile(sibilance_rows, 10.0))
            noise_arr = (
                np.asarray(noise_audio, dtype=float).reshape(-1)
                if noise_audio is not None
                else np.empty(0, dtype=float)
            )
            if noise_arr.size >= frame_size:
                noise_frames = np.lib.stride_tricks.sliding_window_view(
                    noise_arr,
                    frame_size,
                )[::hop_size]
                noise_band_levels: list[float] = []
                for noise_frame in noise_frames:
                    centered = noise_frame - float(np.mean(noise_frame))
                    noise_power = (
                        np.square(np.abs(np.fft.rfft(centered * window))) + 1e-18
                    )
                    noise_band_levels.append(
                        float(
                            10.0
                            * np.log10(np.sum(noise_power[sibilance_mask]) + 1e-18)
                        )
                    )
                if noise_band_levels:
                    noise_sibilance_db = float(np.median(noise_band_levels))

            sibilance_snr_rows = sibilance_rows - noise_sibilance_db
            excess_median = float(np.median(excess_rows))
            excess_p90 = float(np.percentile(excess_rows, 90.0))
            temporal_contrast = max(0.0, excess_p90 - excess_median)
            candidate_threshold = max(1.0, excess_median + 1.25)
            candidate_mask = (
                (excess_rows >= candidate_threshold)
                & (sibilance_snr_rows >= 6.0)
            )
            candidate_ratio = float(np.mean(candidate_mask))
            candidate_snr = (
                float(np.median(sibilance_snr_rows[candidate_mask]))
                if np.any(candidate_mask)
                else float(np.percentile(sibilance_snr_rows, 75.0))
            )
            peak_hz = 6500.0
            if np.any(candidate_mask):
                candidate_spectrum = np.median(active_power[candidate_mask], axis=0)
                local_freqs = frequencies[sibilance_mask]
                peak_hz = float(
                    local_freqs[
                        int(np.argmax(candidate_spectrum[sibilance_mask]))
                    ]
                )

            strength_score = _clamp((excess_p90 - 0.75) / 4.0, 0.0, 1.0)
            temporal_score = _clamp((temporal_contrast - 0.50) / 2.5, 0.0, 1.0)
            support_score = min(
                _clamp(candidate_ratio / 0.06, 0.0, 1.0),
                _clamp((0.65 - candidate_ratio) / 0.20, 0.0, 1.0),
            )
            snr_score = _clamp((candidate_snr - 4.0) / 12.0, 0.0, 1.0)
            evidence_confidence = _bounded_quality_score(
                [
                    (strength_score, 0.35),
                    (temporal_score, 0.25),
                    (support_score, 0.20),
                    (snr_score, 0.20),
                ]
            )
            if (
                temporal_contrast < 1.50
                or candidate_ratio < 0.03
                or candidate_ratio > 0.65
            ):
                evidence_confidence *= 0.25
            deesser_evidence = {
                "available": True,
                "confidence": float(np.clip(evidence_confidence, 0.0, 1.0)),
                "excess_p90_db": excess_p90,
                "temporal_contrast_db": temporal_contrast,
                "candidate_frame_ratio": candidate_ratio,
                "candidate_snr_db": candidate_snr,
                "peak_hz": peak_hz,
            }

    return {
        "frame_db": frame_db,
        "active_frame_mask": active_frames,
        "active_duration_s": active_duration_s,
        "active_ratio": active_ratio,
        "vad_probability_used": frame_vad_probabilities is not None,
        "vad_active_frame_ratio": (
            float(np.mean(frame_vad_probabilities >= 0.35))
            if frame_vad_probabilities is not None
            else 0.0
        ),
        "short_term_lufs": short_term_lufs,
        "short_term_window_count": int(short_term_loudness.size),
        "momentary_lufs": momentary_lufs,
        "active_loudness_spread_db": active_loudness_spread_db,
        # Compatibility key retained for persisted/UI consumers. The value is
        # an active 400 ms loudness spread, not standards-defined EBU LRA.
        "loudness_range_db": active_loudness_spread_db,
        "loudness_window_count": int(momentary_loudness.size),
        "band_energy_db": robust_bands,
        "sibilance_excess_db": robust_bands["sibilance"] - robust_bands["presence"],
        "deesser_frame_evidence": deesser_evidence,
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
    frame_evidence: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, float | bool]]:
    presence_db = _band_mean(freqs, spectrum_db, 2500.0, 4500.0)
    sibilance_db = _band_mean(freqs, spectrum_db, 5000.0, 9000.0)
    sib_mask = (freqs >= 4500.0) & (freqs <= 9500.0)
    if np.any(sib_mask):
        sib_freqs = freqs[sib_mask]
        sib_spec = spectrum_db[sib_mask]
        peak_index = int(np.argmax(sib_spec))
        peak_hz = float(sib_freqs[peak_index])
    else:
        peak_hz = 6500.0

    spectral_excess_db = sibilance_db - presence_db
    aggregate_excess_db = (
        spectral_excess_db
        if robust_sibilance_excess_db is None
        else float(0.35 * spectral_excess_db + 0.65 * robust_sibilance_excess_db)
    )
    frame_data = frame_evidence or {}
    frame_available = bool(frame_data.get("available"))
    evidence_confidence = (
        float(frame_data.get("confidence", 0.0))
        if frame_available
        else 0.0
    )
    sibilance_excess_db = (
        float(frame_data.get("excess_p90_db", aggregate_excess_db))
        if frame_available
        else aggregate_excess_db
    )
    if frame_available:
        peak_hz = float(frame_data.get("peak_hz", peak_hz))
    enabled = bool(
        capture_confidence >= 0.40
        and frame_available
        and evidence_confidence >= 0.48
        and sibilance_excess_db >= 1.0
        and float(frame_data.get("temporal_contrast_db", 0.0)) >= 1.50
        and 0.03
        <= float(frame_data.get("candidate_frame_ratio", 0.0))
        <= 0.65
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
        "frame_evidence_available": frame_available,
        "frame_evidence_confidence": evidence_confidence,
        "temporal_contrast_db": float(
            frame_data.get("temporal_contrast_db", 0.0)
        ),
        "candidate_frame_ratio": float(
            frame_data.get("candidate_frame_ratio", 0.0)
        ),
        "candidate_snr_db": float(
            frame_data.get("candidate_snr_db", 0.0)
        ),
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
    threshold_db = _clamp(speech_body_db - 5.5, -48.0, -14.0)
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
        "sidechain_highpass_enabled": True,
        "measured_short_term_lufs": speech_loudness_lufs,
        "measured_loudness_range_db": loudness_range_db,
    }
    diagnostics = {
        "auto_makeup_enabled": auto_makeup_enabled,
        "target_lufs": target_lufs,
    }
    return settings, diagnostics


def _calibrate_compressor_threshold(
    *,
    speech_audio: np.ndarray,
    sample_rate: int,
    eq_settings: dict[str, Any],
    deesser_settings: dict[str, Any],
    compressor_settings: dict[str, Any],
    active_loudness_spread_db: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Match peak compressor reduction against the authoritative Rust chain."""
    calibrated = dict(compressor_settings)
    target_reduction_db = _clamp(
        1.5 + 0.35 * active_loudness_spread_db,
        1.5,
        4.5,
    )
    diagnostics: dict[str, Any] = {
        "backend": "unavailable",
        "target_gain_reduction_db": target_reduction_db,
        "measured_gain_reduction_db": 0.0,
        "iterations": 0,
    }
    lower_threshold = -55.0
    upper_threshold = -6.0
    best_threshold = float(calibrated["threshold_db"])
    best_reduction = 0.0
    best_error = float("inf")

    for iteration in range(7):
        threshold = (
            best_threshold
            if iteration == 0
            else 0.5 * (lower_threshold + upper_threshold)
        )
        candidate = dict(calibrated)
        candidate["threshold_db"] = threshold
        simulation = simulate_candidate_chain(
            speech_audio.astype(np.float32, copy=False),
            sample_rate,
            eq_settings,
            {
                "deesser": deesser_settings,
                "compressor": candidate,
                "limiter": {
                    "enabled": False,
                    "careful_output_enabled": False,
                },
            },
        )
        if simulation.get("simulation_backend") != "rust":
            return calibrated, diagnostics
        reduction = float(simulation.get("compressor_gain_reduction_db", 0.0))
        if not np.isfinite(reduction):
            return calibrated, diagnostics
        error = abs(reduction - target_reduction_db)
        if error < best_error:
            best_error = error
            best_threshold = threshold
            best_reduction = reduction
        if reduction > target_reduction_db:
            lower_threshold = threshold
        else:
            upper_threshold = threshold
        diagnostics["iterations"] = iteration + 1

    calibrated["threshold_db"] = _clamp(best_threshold, -55.0, -6.0)
    diagnostics.update(
        {
            "backend": "rust",
            "measured_gain_reduction_db": best_reduction,
            "threshold_db": calibrated["threshold_db"],
        }
    )
    return calibrated, diagnostics


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
    vad_probabilities = None
    vad_analysis_backend = "energy_fallback"
    if vad_available:
        vad_probabilities, vad_analysis_backend = analyze_offline_vad(
            speech_arr,
            sample_rate,
        )
    features = _vad_masked_speech_features(
        speech_arr,
        sample_rate,
        noise_rms_db,
        vad_probabilities=vad_probabilities,
        noise_audio=noise_arr,
    )
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

    spectrum_result = analyze_voice_spectrum(
        speech_arr,
        sample_rate,
        vad_probabilities=vad_probabilities,
        noise_audio=noise_arr,
    )
    smoothed_spectrum = smooth_spectrum_perceptual(
        spectrum_result.freqs,
        spectrum_result.median_spectrum_db,
    )
    spectral_confidence = float(spectrum_result.residual_confidence)
    noise_referenced_snr_db = float(spectrum_result.snr_db)
    snr_confidence = _clamp((noise_referenced_snr_db - 6.0) / 12.0, 0.0, 1.0)
    active_duration_confidence = _clamp(float(features["active_duration_s"]) / 3.0, 0.0, 1.0)
    loudness_confidence = _clamp(float(features["loudness_window_count"]) / 8.0, 0.0, 1.0)
    capture_confidence = _bounded_quality_score(
        [
            (spectral_confidence, 0.40),
            (snr_confidence, 0.30),
            (active_duration_confidence, 0.20),
            (loudness_confidence, 0.10),
        ]
    )
    if noise_referenced_snr_db < 6.0:
        capture_confidence = min(capture_confidence, 0.40)
    if float(features["active_duration_s"]) < 2.0:
        capture_confidence = min(capture_confidence, 0.45)
    if spectrum_result.used_single_spectrum_fallback:
        capture_confidence = min(capture_confidence, 0.40)

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
        frame_evidence=features["deesser_frame_evidence"],
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
        eq_settings, _validation = analyze_auto_eq(
            speech_arr,
            sample_rate,
            target_preset,
            vad_probabilities=vad_probabilities,
            noise_audio=noise_arr,
        )
    except Exception as exc:  # pragma: no cover - exercised through return shape
        eq_error = str(exc)

    compressor_calibration: dict[str, Any] = {
        "backend": "unavailable",
        "target_gain_reduction_db": 0.0,
        "measured_gain_reduction_db": 0.0,
        "iterations": 0,
    }
    if eq_settings is not None:
        compressor_settings, compressor_calibration = (
            _calibrate_compressor_threshold(
                speech_audio=speech_arr,
                sample_rate=sample_rate,
                eq_settings=eq_settings,
                deesser_settings=deesser_settings,
                compressor_settings=compressor_settings,
                active_loudness_spread_db=float(
                    features["active_loudness_spread_db"]
                ),
            )
        )

    dynamics_confidence = _clamp(speech_dynamic_range_db / 8.0, 0.0, 1.0)
    quiet_room_confidence = _clamp((-32.0 - noise_rms_db) / 18.0, 0.0, 1.0)
    eq_confidence = float(eq_settings.get("analysis_confidence", capture_confidence)) if eq_settings else capture_confidence
    gate_confidence = float(np.clip(0.55 * capture_confidence + 0.45 * snr_confidence, 0.0, 1.0))
    deesser_confidence = _bounded_quality_score(
        [
            (capture_confidence, 0.55),
            (float(deesser_diag["frame_evidence_confidence"]), 0.45),
        ]
    )
    compressor_confidence = float(np.clip(0.55 * capture_confidence + 0.45 * dynamics_confidence, 0.0, 1.0))
    setup_confidence = _bounded_quality_score(
        [
            (eq_confidence, 0.35),
            (gate_confidence, 0.25),
            (max(deesser_confidence, 0.50) if not deesser_diag["enabled"] else deesser_confidence, 0.15),
            (compressor_confidence, 0.15),
            (quiet_room_confidence, 0.10),
        ]
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
    if noise_referenced_snr_db < 8.0:
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
        or noise_referenced_snr_db < 8.0
        or capture_confidence < 0.50
    )
    eq_apply_recommended = bool(
        eq_settings is not None
        and eq_settings.get("apply_recommended", True)
    )
    if not eq_apply_recommended:
        uncertainty_reasons.append("Auto-EQ abstained from this capture")
    apply_recommended = bool(
        not weak_capture
        and eq_apply_recommended
        and offline_validation_passed
    )
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
            "confidence_semantics": "bounded_quality_score",
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
            "noise_referenced_snr_db": noise_referenced_snr_db,
            "noise_reference_source": spectrum_result.noise_reference_source,
            "vad_active_duration_s": features["active_duration_s"],
            "vad_active_ratio": features["active_ratio"],
            "short_term_lufs": features["short_term_lufs"],
            "short_term_loudness_window_count": features[
                "short_term_window_count"
            ],
            "momentary_lufs": features["momentary_lufs"],
            "active_loudness_spread_db": features[
                "active_loudness_spread_db"
            ],
            "loudness_range_db": features["loudness_range_db"],
            "robust_band_energy_db": features["band_energy_db"],
            "gate_mode_label": GATE_MODE_LABELS[gate_settings["gate_mode"]],
            "sibilance_excess_db": deesser_diag["sibilance_excess_db"],
            "sibilance_peak_hz": deesser_diag["peak_hz"],
            "deesser_enabled": deesser_diag["enabled"],
            "deesser_frame_evidence_confidence": deesser_diag[
                "frame_evidence_confidence"
            ],
            "deesser_temporal_contrast_db": deesser_diag[
                "temporal_contrast_db"
            ],
            "deesser_candidate_frame_ratio": deesser_diag[
                "candidate_frame_ratio"
            ],
            "deesser_candidate_snr_db": deesser_diag["candidate_snr_db"],
            "compressor_auto_makeup_enabled": compressor_diag["auto_makeup_enabled"],
            "compressor_target_lufs": compressor_diag["target_lufs"],
            "compressor_calibration": compressor_calibration,
            "vad_available": bool(vad_available),
            "vad_analysis_backend": vad_analysis_backend,
            "vad_probability_used": bool(features["vad_probability_used"]),
            "vad_active_frame_ratio": float(features["vad_active_frame_ratio"]),
            "offline_validation_passed": offline_validation_passed,
            "offline_validation": offline_validation,
        },
    }


__all__ = ["GATE_MODE_LABELS", "analyze_voice_setup"]
