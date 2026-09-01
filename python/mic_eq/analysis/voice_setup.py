"""Uncertainty-aware Auto Voice Setup analysis and native-chain validation.

Recommendations use energy-VAD-masked speech, BS.1770 K-weighted momentary and
three-second short-term loudness, active loudness spread, and robust band-energy
summaries. Candidate settings are checked through the offline DSP simulator
before the UI offers to apply them.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
from scipy.signal import lfilter, resample_poly

from .auto_eq import analyze_auto_eq, simulate_candidate_chain
from .deesser_fusion import (
    CLIP_FEATURE_NAMES,
    ENABLE_PROBABILITY_THRESHOLD,
    MODEL_VERSION as DEESSER_MODEL_VERSION,
    predict_clip_probability,
    predict_frame_probabilities,
)
from .noise_reference import (
    MIN_NOISE_DURATION_S,
    CaptureMetadata,
    analyze_noise_reference,
)
from .spectrum import (
    _interpolate_vad_probabilities,
    analyze_voice_spectrum,
    smooth_spectrum_perceptual,
)
from .vad import (
    VAD_SPEECH_EVIDENCE_THRESHOLD,
    VAD_STRONG_SPEECH_THRESHOLD,
    analyze_offline_vad,
)
from ..config import EQ_FREQUENCIES

NOISE_MIN_DURATION_S = MIN_NOISE_DURATION_S
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

DYNAMICS_PROFILES: dict[str, dict[str, float]] = {
    "gentle": {
        "target_p95_db": 2.0,
        "target_median_db": 0.7,
        "peak_cap_db": 6.0,
        "ratio_scale": 0.82,
    },
    "balanced": {
        "target_p95_db": 3.5,
        "target_median_db": 1.4,
        "peak_cap_db": 8.0,
        "ratio_scale": 1.0,
    },
    "dense": {
        "target_p95_db": 5.5,
        "target_median_db": 2.5,
        "peak_cap_db": 10.0,
        "ratio_scale": 1.22,
    },
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
            (
                (frame_vad_probabilities >= VAD_SPEECH_EVIDENCE_THRESHOLD)
                & supported_energy
            )
            | (frame_vad_probabilities >= VAD_STRONG_SPEECH_THRESHOLD)
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
    momentary_lufs = float(np.median(momentary_loudness))
    short_term_window_count = int(short_term_loudness.size)
    short_term_lufs = (
        float(np.median(short_term_loudness))
        if short_term_window_count
        else momentary_lufs
    )
    short_term_lufs_source = (
        "short_term_3s" if short_term_window_count else "momentary_400ms_fallback"
    )
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

    deesser_evidence: dict[str, Any] = {
        "available": False,
        "confidence": 0.0,
        "frame_probabilities": np.empty(0, dtype=float),
        "frame_feature_rows": np.empty((0, 6), dtype=float),
        "frame_indices": np.empty(0, dtype=int),
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
            local_sibilance_power = active_power[:, sibilance_mask]
            local_sibilance_db = 10.0 * np.log10(
                np.maximum(local_sibilance_power, 1e-18)
            )
            local_freqs = frequencies[sibilance_mask]
            peak_indices = np.argmax(local_sibilance_power, axis=1)
            peak_freqs = local_freqs[peak_indices]
            peak_prominence_db = np.max(local_sibilance_db, axis=1) - np.median(
                local_sibilance_db,
                axis=1,
            )
            if frame_vad_probabilities is not None:
                active_vad = np.clip(
                    frame_vad_probabilities[active_indices],
                    0.0,
                    1.0,
                )
                unvoiced_evidence = 1.0 - active_vad
            else:
                unvoiced_evidence = np.full(excess_rows.shape, 0.5, dtype=float)
            peak_location_score = np.exp(
                -0.5
                * np.square(
                    np.log2(np.maximum(peak_freqs, 1.0) / 6500.0) / 0.70
                )
            )
            frame_feature_rows = np.column_stack(
                [
                    np.clip((excess_rows - 0.50) / 5.0, 0.0, 1.0),
                    np.clip(
                        (excess_rows - excess_median - 0.20) / 3.0,
                        0.0,
                        1.0,
                    ),
                    np.clip((sibilance_snr_rows - 3.0) / 15.0, 0.0, 1.0),
                    unvoiced_evidence,
                    np.clip((peak_prominence_db - 1.0) / 8.0, 0.0, 1.0),
                    np.clip(peak_location_score, 0.0, 1.0),
                ]
            )
            frame_probabilities = predict_frame_probabilities(frame_feature_rows)
            candidate_ratio = float(np.mean(frame_probabilities))
            probability_sum = max(float(np.sum(frame_probabilities)), 1e-9)
            candidate_snr = float(
                np.dot(frame_probabilities, sibilance_snr_rows) / probability_sum
            )
            candidate_spectrum = np.average(
                active_power,
                axis=0,
                weights=np.maximum(frame_probabilities, 1e-6),
            )
            peak_hz = float(
                local_freqs[
                    int(np.argmax(candidate_spectrum[sibilance_mask]))
                ]
            )

            temporal_score = _clamp((temporal_contrast - 0.50) / 2.5, 0.0, 1.0)
            frame_p90 = float(np.percentile(frame_probabilities, 90.0))
            top_count = max(1, int(np.ceil(frame_probabilities.size * 0.10)))
            frame_top_mean = float(
                np.mean(np.partition(frame_probabilities, -top_count)[-top_count:])
            )
            deesser_evidence = {
                "available": True,
                "confidence": frame_p90,
                "frame_probabilities": frame_probabilities,
                "frame_feature_rows": frame_feature_rows,
                "frame_indices": active_indices,
                "frame_probability_p90": frame_p90,
                "frame_probability_top_mean": frame_top_mean,
                "temporal_score": temporal_score,
                "absolute_hf_strength_p90": float(
                    np.percentile(frame_feature_rows[:, 0], 90.0)
                ),
                "noise_reliability_p90": float(
                    np.percentile(frame_feature_rows[:, 2], 90.0)
                ),
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
            float(
                np.mean(
                    frame_vad_probabilities >= VAD_SPEECH_EVIDENCE_THRESHOLD
                )
            )
            if frame_vad_probabilities is not None
            else 0.0
        ),
        "short_term_lufs": short_term_lufs,
        "short_term_window_count": short_term_window_count,
        "short_term_lufs_source": short_term_lufs_source,
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
    # Corrected and calibrated Silero v6.2.1 is stable near 0.45-0.48 across
    # 0-20 dB mixtures. Keep only a small SNR adjustment instead of the old
    # wide range tuned against the shipped stateless/truncated VAD adapter.
    vad_threshold = _clamp(0.46 - (speech_snr_db - 10.0) / 800.0, 0.42, 0.50)
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
    noise_reference_quality: float = 1.0,
    noise_reference_status: str = "usable",
    robust_sibilance_excess_db: float | None = None,
    frame_evidence: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    evidence_confidence = 0.0
    sibilance_excess_db = (
        float(frame_data.get("excess_p90_db", aggregate_excess_db))
        if frame_available
        else aggregate_excess_db
    )
    if frame_available:
        peak_hz = float(frame_data.get("peak_hz", peak_hz))
    clip_feature_values = np.asarray(
        [
            float(frame_data.get("frame_probability_p90", 0.0)),
            float(frame_data.get("frame_probability_top_mean", 0.0)),
            float(frame_data.get("candidate_frame_ratio", 0.0)),
            float(frame_data.get("temporal_score", 0.0)),
            float(frame_data.get("absolute_hf_strength_p90", 0.0)),
            float(frame_data.get("noise_reliability_p90", 0.0)),
        ],
        dtype=float,
    )
    detection_probability = 0.0
    if frame_available:
        detection_probability = predict_clip_probability(clip_feature_values)
        evidence_confidence = _bounded_quality_score(
            [
                (detection_probability, 0.70),
                (noise_reference_quality, 0.20),
                (capture_confidence, 0.10),
            ]
        )
    invalid_evidence = bool(
        not frame_available
        or str(noise_reference_status).strip().lower() == "invalid"
        or not np.isfinite(clip_feature_values).all()
    )
    enabled = bool(
        not invalid_evidence
        and detection_probability >= ENABLE_PROBABILITY_THRESHOLD
    )
    auto_amount = _clamp(
        0.18
        + 0.55 * detection_probability
        + 0.12 * _clamp(sibilance_excess_db / 6.0, 0.0, 1.0),
        0.20,
        0.85,
    )
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
        "detection_probability": detection_probability,
        "enable_probability_threshold": ENABLE_PROBABILITY_THRESHOLD,
        "model_version": DEESSER_MODEL_VERSION,
        "clip_features": {
            name: float(value)
            for name, value in zip(
                CLIP_FEATURE_NAMES,
                clip_feature_values,
                strict=True,
            )
        },
        "invalid_evidence": invalid_evidence,
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
    dynamics_intensity: str,
    custom_target_p95_db: float,
    custom_peak_cap_db: float,
) -> tuple[dict[str, Any], dict[str, float | bool]]:
    profile_name = dynamics_intensity.lower()
    if profile_name == "custom":
        bounded_target_p95 = _clamp(custom_target_p95_db, 1.0, 8.0)
        profile = {
            "target_p95_db": bounded_target_p95,
            "target_median_db": _clamp(bounded_target_p95 * 0.42, 0.3, 4.0),
            "peak_cap_db": _clamp(
                custom_peak_cap_db,
                bounded_target_p95 + 0.5,
                12.0,
            ),
            "ratio_scale": _clamp(0.72 + bounded_target_p95 / 12.5, 0.8, 1.35),
        }
    else:
        profile_name = profile_name if profile_name in DYNAMICS_PROFILES else "balanced"
        profile = DYNAMICS_PROFILES[profile_name]
    target_lufs = TARGET_LUFS_BY_CURVE.get(target_preset, -18.0)
    threshold_db = _clamp(speech_body_db - 5.5, -48.0, -14.0)
    ratio = _clamp(
        (2.2 + loudness_range_db / 5.0) * profile["ratio_scale"],
        1.8,
        5.5,
    )
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
        "dynamics_intensity": profile_name,
        "target_p95_reduction_db": profile["target_p95_db"],
        "peak_reduction_cap_db": profile["peak_cap_db"],
    }
    diagnostics = {
        "auto_makeup_enabled": auto_makeup_enabled,
        "target_lufs": target_lufs,
        "dynamics_intensity": profile_name,
        "target_p95_reduction_db": profile["target_p95_db"],
        "target_median_reduction_db": profile["target_median_db"],
        "peak_reduction_cap_db": profile["peak_cap_db"],
    }
    return settings, diagnostics


_COMPRESSOR_SEARCH_BUDGET = 68
_COMPRESSOR_SEARCH_BOUNDS = {
    "threshold_db": (-55.0, -6.0),
    "ratio": (1.5, 6.0),
    "attack_ms": (3.0, 25.0),
    "release_ms": (60.0, 320.0),
}
_COMPRESSOR_OBJECTIVE_NORMALIZERS = {
    "loudness_error_db": 2.0,
    "median_gr_error_db": 1.0,
    "p95_gr_error_db": 1.0,
    "headroom_shortfall_db": 1.0,
    "pumping_score_db": 1.0,
    "silence_gain_excess_db": 1.0,
    "activity_ratio_deficit": 0.20,
}
_COMPRESSOR_OBJECTIVE_WEIGHTS = {
    "loudness": 1.00,
    "median_gr": 0.35,
    "p95_gr": 0.90,
    "headroom": 0.45,
    "pumping": 0.30,
    "silence_gain": 1.50,
    "activity": 0.25,
    "prior": 0.08,
}


def _huber(value: float) -> float:
    magnitude = abs(float(value))
    return 0.5 * magnitude * magnitude if magnitude <= 1.0 else magnitude - 0.5


def _halton(index: int, base: int) -> float:
    result = 0.0
    scale = 1.0
    while index > 0:
        scale /= base
        result += scale * (index % base)
        index //= base
    return result


def _calibrate_compressor_threshold(
    *,
    speech_audio: np.ndarray,
    sample_rate: int,
    eq_settings: dict[str, Any],
    deesser_settings: dict[str, Any],
    compressor_settings: dict[str, Any],
    target_p95_db: float,
    target_median_db: float,
    peak_cap_db: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit four compressor controls with a bounded deterministic native search."""
    calibrated = dict(compressor_settings)
    diagnostics: dict[str, Any] = {
        "backend": "unavailable",
        "objective": "bounded_multi_objective_compressor_search_v1",
        "target_p95_gain_reduction_db": target_p95_db,
        "target_median_gain_reduction_db": target_median_db,
        "peak_gain_reduction_cap_db": peak_cap_db,
        "measured_p95_gain_reduction_db": 0.0,
        "measured_median_gain_reduction_db": 0.0,
        "measured_peak_gain_reduction_db": 0.0,
        "iterations": 0,
        "candidate_budget": _COMPRESSOR_SEARCH_BUDGET,
        "objective_normalizers": dict(_COMPRESSOR_OBJECTIVE_NORMALIZERS),
        "objective_weights": dict(_COMPRESSOR_OBJECTIVE_WEIGHTS),
    }
    started = time.perf_counter()
    incumbent = {
        key: _clamp(
            float(calibrated[key]),
            *_COMPRESSOR_SEARCH_BOUNDS[key],
        )
        for key in _COMPRESSOR_SEARCH_BOUNDS
    }
    evaluated: dict[tuple[float, ...], tuple[float, dict[str, Any], dict[str, float]]] = {}

    def key_for(candidate: Mapping[str, float]) -> tuple[float, ...]:
        return tuple(round(float(candidate[key]), 6) for key in _COMPRESSOR_SEARCH_BOUNDS)

    def evaluate(candidate_values: Mapping[str, float]) -> None:
        if len(evaluated) >= _COMPRESSOR_SEARCH_BUDGET - 1:
            return
        candidate_key = key_for(candidate_values)
        if candidate_key in evaluated:
            return
        candidate = dict(calibrated)
        candidate.update(
            {
                key: _clamp(
                    float(candidate_values[key]),
                    *_COMPRESSOR_SEARCH_BOUNDS[key],
                )
                for key in _COMPRESSOR_SEARCH_BOUNDS
            }
        )
        simulation_compressor = dict(candidate)
        if simulation_compressor.get("auto_makeup_enabled", False):
            simulation_compressor["auto_makeup_enabled"] = False
            simulation_compressor["makeup_gain_db"] = 0.0
        simulation = simulate_candidate_chain(
            speech_audio.astype(np.float32, copy=False),
            sample_rate,
            eq_settings,
            {
                "deesser": deesser_settings,
                "compressor": simulation_compressor,
                "limiter": {
                    "enabled": True,
                    "ceiling_db": -1.5,
                    "release_ms": 80.0,
                    "careful_output_enabled": True,
                },
            },
        )
        if simulation.get("simulation_backend") != "rust":
            evaluated[candidate_key] = (
                float("inf"),
                simulation,
                dict(candidate_values),
            )
            return
        peak = float(simulation.get("compressor_gain_reduction_db", 0.0))
        median = float(
            simulation.get("compressor_gain_reduction_median_db", peak)
        )
        p95 = float(simulation.get("compressor_gain_reduction_p95_db", peak))
        active_ratio = float(
            simulation.get("compressor_gain_reduction_active_ratio", 0.0)
        )
        active_gain = float(simulation.get("active_output_gain_db", 0.0))
        target_lufs = float(calibrated.get("target_lufs", -18.0))
        output_lufs = (
            target_lufs
            if calibrated.get("auto_makeup_enabled", False)
            else float(calibrated.get("measured_short_term_lufs", -18.0))
            + active_gain
        )
        output_true_peak = float(simulation.get("output_true_peak_db", 120.0))
        ceiling = float(simulation.get("limiter_effective_ceiling_db", -1.5))
        pre_limiter_headroom = float(
            simulation.get("pre_limiter_true_peak_headroom_db", -120.0)
        )
        pumping = float(simulation.get("compressor_pumping_score_db", 120.0))
        silence_gain = float(simulation.get("silence_output_gain_db", 120.0))
        non_finite = bool(simulation.get("non_finite_output", True))
        finite_values = np.asarray(
            [
                peak,
                median,
                p95,
                active_ratio,
                output_lufs,
                output_true_peak,
                pre_limiter_headroom,
                pumping,
                silence_gain,
            ],
            dtype=float,
        )
        hard_rejected = bool(
            non_finite
            or not np.isfinite(finite_values).all()
            or output_true_peak > ceiling + 0.10
            or peak > peak_cap_db + 1.0e-6
        )
        prior_terms = []
        for key, (lower, upper) in _COMPRESSOR_SEARCH_BOUNDS.items():
            span = upper - lower
            prior_terms.append(
                ((float(candidate[key]) - incumbent[key]) / span) ** 2
            )
        terms = {
            "loudness": _huber(
                (output_lufs - target_lufs)
                / _COMPRESSOR_OBJECTIVE_NORMALIZERS["loudness_error_db"]
            ),
            "median_gr": _huber(
                (median - target_median_db)
                / _COMPRESSOR_OBJECTIVE_NORMALIZERS["median_gr_error_db"]
            ),
            "p95_gr": _huber(
                (p95 - target_p95_db)
                / _COMPRESSOR_OBJECTIVE_NORMALIZERS["p95_gr_error_db"]
            ),
            "headroom": _huber(
                max(0.0, 1.0 - pre_limiter_headroom)
                / _COMPRESSOR_OBJECTIVE_NORMALIZERS["headroom_shortfall_db"]
            ),
            "pumping": _huber(
                pumping / _COMPRESSOR_OBJECTIVE_NORMALIZERS["pumping_score_db"]
            ),
            "silence_gain": _huber(
                max(0.0, silence_gain - 0.25)
                / _COMPRESSOR_OBJECTIVE_NORMALIZERS["silence_gain_excess_db"]
            ),
            "activity": _huber(
                max(0.0, 0.20 - active_ratio)
                / _COMPRESSOR_OBJECTIVE_NORMALIZERS["activity_ratio_deficit"]
            ),
            "prior": float(np.mean(prior_terms)),
        }
        score = sum(
            _COMPRESSOR_OBJECTIVE_WEIGHTS[name] * value
            for name, value in terms.items()
        )
        if hard_rejected:
            score = float("inf")
        evaluated[candidate_key] = (
            float(score),
            simulation,
            {key: float(candidate[key]) for key in _COMPRESSOR_SEARCH_BOUNDS},
        )

    evaluate(incumbent)
    for threshold in np.linspace(-55.0, -6.0, 33):
        threshold_candidate = dict(incumbent)
        threshold_candidate["threshold_db"] = float(threshold)
        evaluate(threshold_candidate)
    for index in range(1, 17):
        candidate = {}
        for key, base in zip(_COMPRESSOR_SEARCH_BOUNDS, (2, 3, 5, 7)):
            lower, upper = _COMPRESSOR_SEARCH_BOUNDS[key]
            candidate[key] = lower + _halton(index, base) * (upper - lower)
        evaluate(candidate)

    feasible = sorted(
        (item for item in evaluated.values() if np.isfinite(item[0])),
        key=lambda item: (item[0], key_for(item[2])),
    )
    if not feasible:
        diagnostics["iterations"] = len(evaluated)
        diagnostics["search_runtime_ms"] = (time.perf_counter() - started) * 1000.0
        return calibrated, diagnostics

    local_steps = {
        "threshold_db": 3.0,
        "ratio": 0.5,
        "attack_ms": 3.0,
        "release_ms": 25.0,
    }
    refinement_seeds = [feasible[0]]
    multivariable_seed = next(
        (
            item
            for item in feasible
            if any(
                abs(item[2][key] - incumbent[key]) > 1.0e-6
                for key in ("ratio", "attack_ms", "release_ms")
            )
        ),
        None,
    )
    if multivariable_seed is not None and key_for(multivariable_seed[2]) != key_for(
        refinement_seeds[0][2]
    ):
        refinement_seeds.append(multivariable_seed)
    else:
        refinement_seeds.extend(feasible[1:2])
    for _, _, seed in refinement_seeds:
        for key, step in local_steps.items():
            for direction in (-1.0, 1.0):
                candidate = dict(seed)
                candidate[key] += direction * step
                evaluate(candidate)

    feasible = sorted(
        (item for item in evaluated.values() if np.isfinite(item[0])),
        key=lambda item: (item[0], key_for(item[2])),
    )
    threshold_only = min(
        (
            item
            for item in feasible
            if all(
                abs(item[2][key] - incumbent[key]) <= 1.0e-6
                for key in ("ratio", "attack_ms", "release_ms")
            )
        ),
        key=lambda item: (item[0], key_for(item[2])),
        default=None,
    )
    expanded = feasible[0]
    if threshold_only is None:
        expanded_selected = True
        best_score, best_simulation, best_values = expanded
    else:
        required_tie_break_improvement = max(0.001, 0.01 * threshold_only[0])
        expanded_selected = bool(
            threshold_only[0] - expanded[0] > required_tie_break_improvement
        )
        best_score, best_simulation, best_values = (
            expanded if expanded_selected else threshold_only
        )
    calibrated.update(best_values)
    winner_verification = simulate_candidate_chain(
        speech_audio.astype(np.float32, copy=False),
        sample_rate,
        eq_settings,
        {
            "deesser": deesser_settings,
            "compressor": {
                **calibrated,
                **(
                    {
                        "auto_makeup_enabled": False,
                        "makeup_gain_db": 0.0,
                    }
                    if calibrated.get("auto_makeup_enabled", False)
                    else {}
                ),
            },
            "limiter": {
                "enabled": True,
                "ceiling_db": -1.5,
                "release_ms": 80.0,
                "careful_output_enabled": True,
            },
        },
    )
    if winner_verification.get("simulation_backend") == "rust":
        best_simulation = winner_verification
    median = float(best_simulation["compressor_gain_reduction_median_db"])
    p95 = float(best_simulation["compressor_gain_reduction_p95_db"])
    peak = float(best_simulation["compressor_gain_reduction_db"])
    active_ratio = float(best_simulation["compressor_gain_reduction_active_ratio"])
    threshold_only_scores = [
        score
        for score, _, values in evaluated.values()
        if all(
            abs(values[key] - incumbent[key]) <= 1.0e-6
            for key in ("ratio", "attack_ms", "release_ms")
        )
    ]
    incumbent_entry = evaluated.get(key_for(incumbent))
    diagnostics.update(
        {
            "backend": "rust",
            "measured_median_gain_reduction_db": median,
            "measured_p95_gain_reduction_db": p95,
            "measured_peak_gain_reduction_db": peak,
            "active_reduction_ratio": active_ratio,
            "peak_cap_passed": peak <= peak_cap_db + 1.0e-6,
            "total_objective": best_score,
            "incumbent_objective": (
                incumbent_entry[0] if incumbent_entry is not None else float("inf")
            ),
            "threshold_only_objective": min(threshold_only_scores, default=float("inf")),
            "expanded_candidate_objective": expanded[0],
            "expanded_search_selected": expanded_selected,
            "active_output_gain_db": float(
                best_simulation.get("active_output_gain_db", 0.0)
            ),
            "silence_output_gain_db": float(
                best_simulation.get("silence_output_gain_db", 0.0)
            ),
            "compressor_pumping_score_db": float(
                best_simulation.get("compressor_pumping_score_db", 0.0)
            ),
            "output_true_peak_db": float(
                best_simulation.get("output_true_peak_db", -120.0)
            ),
            "pre_limiter_true_peak_headroom_db": float(
                best_simulation.get("pre_limiter_true_peak_headroom_db", 0.0)
            ),
            "search_runtime_ms": (time.perf_counter() - started) * 1000.0,
            "candidate_count": len(evaluated) + 1,
            "iterations": len(evaluated) + 1,
            # Compatibility aliases for older diagnostic consumers.
            "target_gain_reduction_db": target_p95_db,
            "measured_gain_reduction_db": p95,
            "threshold_db": calibrated["threshold_db"],
            "ratio": calibrated["ratio"],
            "attack_ms": calibrated["attack_ms"],
            "release_ms": calibrated["release_ms"],
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
    dynamics_intensity: str = "balanced",
    custom_target_p95_db: float = 3.5,
    custom_peak_cap_db: float = 8.0,
    noise_metadata: CaptureMetadata | Mapping[str, Any] | None = None,
    speech_metadata: CaptureMetadata | Mapping[str, Any] | None = None,
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
    noise_vad_probabilities = None
    noise_vad_backend = "energy_fallback"
    if vad_available:
        noise_vad_probabilities, noise_vad_backend = analyze_offline_vad(
            noise_arr,
            sample_rate,
        )
    noise_reference = analyze_noise_reference(
        noise_arr,
        speech_arr,
        sample_rate,
        noise_metadata=noise_metadata,
        speech_metadata=speech_metadata,
        noise_vad_probabilities=noise_vad_probabilities,
        speech_vad_probabilities=vad_probabilities,
    )
    conservative_noise_spectrum = (
        noise_reference.frequencies,
        noise_reference.conservative_spectrum_db,
    )
    conservative_noise_rms_db = noise_reference.conservative_noise_rms_db
    features = _vad_masked_speech_features(
        speech_arr,
        sample_rate,
        conservative_noise_rms_db,
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
    speech_snr_db = speech_body_db - conservative_noise_rms_db

    spectrum_result = analyze_voice_spectrum(
        speech_arr,
        sample_rate,
        vad_probabilities=vad_probabilities,
        noise_audio=noise_arr,
        noise_spectrum_override=conservative_noise_spectrum,
        noise_reference_source_override="validated_conservative",
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
    if int(features["short_term_window_count"]) == 0:
        loudness_confidence *= 0.6
    capture_confidence = _bounded_quality_score(
        [
            (spectral_confidence, 0.30),
            (snr_confidence, 0.22),
            (noise_reference.quality_score, 0.23),
            (active_duration_confidence, 0.17),
            (loudness_confidence, 0.08),
        ]
    )
    if noise_referenced_snr_db < 6.0:
        capture_confidence = min(capture_confidence, 0.40)
    if float(features["active_duration_s"]) < 2.0:
        capture_confidence = min(capture_confidence, 0.45)
    if spectrum_result.used_single_spectrum_fallback:
        capture_confidence = min(capture_confidence, 0.40)
    if noise_reference.status == "questionable":
        capture_confidence = min(capture_confidence, 0.49)
    elif noise_reference.status == "invalid":
        capture_confidence = min(capture_confidence, 0.20)

    gate_settings = _recommend_gate_settings(
        vad_available=vad_available,
        noise_rms_db=conservative_noise_rms_db,
        speech_floor_db=speech_floor_db,
        speech_body_db=speech_body_db,
        speech_snr_db=speech_snr_db,
        speech_dynamic_range_db=speech_dynamic_range_db,
    )
    deesser_settings, deesser_diag = _recommend_deesser_settings(
        freqs=spectrum_result.freqs,
        spectrum_db=smoothed_spectrum,
        capture_confidence=capture_confidence,
        noise_reference_quality=noise_reference.quality_score,
        noise_reference_status=noise_reference.status,
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
        dynamics_intensity=dynamics_intensity,
        custom_target_p95_db=custom_target_p95_db,
        custom_peak_cap_db=custom_peak_cap_db,
    )
    compressor_settings["noise_reference_reliability"] = float(
        np.clip(noise_reference.quality_score, 0.0, 1.0)
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
            noise_spectrum_override=conservative_noise_spectrum,
            noise_reference_quality=noise_reference.quality_score,
            noise_reference_status=noise_reference.status,
            noise_reference_reasons=noise_reference.reasons,
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
                target_p95_db=float(
                    compressor_diag["target_p95_reduction_db"]
                ),
                target_median_db=float(
                    compressor_diag["target_median_reduction_db"]
                ),
                peak_cap_db=float(compressor_diag["peak_reduction_cap_db"]),
            )
        )

    dynamics_confidence = _clamp(speech_dynamic_range_db / 8.0, 0.0, 1.0)
    quiet_room_confidence = _clamp(
        (-32.0 - conservative_noise_rms_db) / 18.0,
        0.0,
        1.0,
    )
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
        compressor_p95 = float(
            offline_validation.get("compressor_gain_reduction_p95_db", compressor_gr)
        )
        deesser_gr = float(offline_validation.get("deesser_gain_reduction_db", 120.0))
        offline_validation_passed = bool(
            np.isfinite([output_true_peak, compressor_gr, deesser_gr]).all()
            and output_true_peak <= ceiling + 0.15
            and compressor_gr
            <= float(compressor_diag["peak_reduction_cap_db"]) + 0.25
            and compressor_p95
            <= float(compressor_diag["target_p95_reduction_db"]) + 1.25
            and deesser_gr <= 10.0
        )
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        offline_validation = {"error": str(exc), "simulation_backend": "unavailable"}

    uncertainty_reasons: list[str] = []
    uncertainty_reasons.extend(noise_reference.reasons)
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
        or noise_reference.status != "usable"
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
            "conservative_noise_rms_db": conservative_noise_rms_db,
            "noise_reference_quality": noise_reference.diagnostics(),
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
            "short_term_lufs_source": features["short_term_lufs_source"],
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
            "deesser_detection_probability": deesser_diag[
                "detection_probability"
            ],
            "deesser_enable_probability_threshold": deesser_diag[
                "enable_probability_threshold"
            ],
            "deesser_model_version": deesser_diag["model_version"],
            "deesser_clip_features": deesser_diag["clip_features"],
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
            "dynamics_intensity": compressor_diag["dynamics_intensity"],
            "compressor_calibration": compressor_calibration,
            "vad_available": bool(vad_available),
            "vad_analysis_backend": vad_analysis_backend,
            "noise_vad_analysis_backend": noise_vad_backend,
            "vad_probability_used": bool(features["vad_probability_used"]),
            "vad_active_frame_ratio": float(features["vad_active_frame_ratio"]),
            "offline_validation_passed": offline_validation_passed,
            "offline_validation": offline_validation,
        },
    }


def _shape_error_db(
    measured_freqs: np.ndarray,
    measured_db: np.ndarray,
    target_preset: str,
) -> float:
    """Return level-invariant voice-band error against the selected house curve."""
    from .auto_eq_parts.target import get_target_curve

    mask = (measured_freqs >= 80.0) & (measured_freqs <= 12_000.0)
    if np.count_nonzero(mask) < 8:
        return float("inf")
    measured = np.asarray(measured_db[mask], dtype=float)
    freqs = np.asarray(measured_freqs[mask], dtype=float)
    target = np.asarray(
        get_target_curve(freqs, target_preset, measured, target_mode="adaptive"),
        dtype=float,
    )
    measured -= float(np.median(measured))
    target -= float(np.median(target))
    return float(np.sqrt(np.mean(np.square(measured - target))))


def validate_voice_setup_verification(
    noise_audio: np.ndarray,
    original_speech_audio: np.ndarray,
    verification_speech_audio: np.ndarray,
    sample_rate: int,
    setup_result: Mapping[str, Any],
    target_preset: str,
) -> dict[str, Any]:
    """Validate a second passage through the exact native candidate chain.

    This is an engineering validation of repeatability and DSP constraints. It
    deliberately does not claim that a listener will prefer the result.
    """
    noise = np.asarray(noise_audio, dtype=np.float32)
    original = np.asarray(original_speech_audio, dtype=np.float32)
    verification = np.asarray(verification_speech_audio, dtype=np.float32)
    if verification.size < int(sample_rate * SPEECH_MIN_DURATION_S):
        return {
            "decision": "retry",
            "reasons": ["verification passage was too short"],
            "perceptual_validation": False,
        }
    if not np.isfinite(verification).all() or float(np.max(np.abs(verification))) >= 0.999:
        return {
            "decision": "retry",
            "reasons": ["verification passage was non-finite or clipped"],
            "perceptual_validation": False,
        }

    eq_settings = dict(setup_result.get("eq_settings") or {})
    if not eq_settings:
        eq_settings = {
            "band_freqs": list(EQ_FREQUENCIES),
            "band_gains": [0.0] * len(EQ_FREQUENCIES),
            "band_qs": [1.41] * len(EQ_FREQUENCIES),
        }
    chain = {
        "deesser": dict(setup_result.get("deesser_settings") or {}),
        "compressor": dict(setup_result.get("compressor_settings") or {}),
        "limiter": {
            "enabled": True,
            "ceiling_db": -1.5,
            "release_ms": 80.0,
            "careful_output_enabled": True,
        },
        "return_output_audio": True,
    }
    processed = simulate_candidate_chain(
        verification,
        sample_rate,
        eq_settings,
        chain,
    )
    processed_noise = simulate_candidate_chain(
        noise,
        sample_rate,
        eq_settings,
        chain,
    )
    if (
        processed.get("simulation_backend") != "rust"
        or "output_audio" not in processed
        or "output_audio" not in processed_noise
    ):
        return {
            "decision": "retry",
            "reasons": ["native verification renderer is unavailable"],
            "simulation_backend": processed.get("simulation_backend", "unavailable"),
            "perceptual_validation": False,
        }

    rendered = np.asarray(processed.pop("output_audio"), dtype=np.float32)
    rendered_noise = np.asarray(processed_noise.pop("output_audio"), dtype=np.float32)
    original_spectrum = analyze_voice_spectrum(original, sample_rate)
    before_spectrum = analyze_voice_spectrum(
        verification,
        sample_rate,
        noise_audio=noise,
    )
    after_spectrum = analyze_voice_spectrum(
        rendered,
        sample_rate,
        noise_audio=rendered_noise,
    )
    before_error = _shape_error_db(
        before_spectrum.freqs,
        before_spectrum.median_spectrum_db,
        target_preset,
    )
    after_error = _shape_error_db(
        after_spectrum.freqs,
        after_spectrum.median_spectrum_db,
        target_preset,
    )
    original_shape = np.interp(
        before_spectrum.freqs,
        original_spectrum.freqs,
        original_spectrum.median_spectrum_db,
    )
    repeatability_mask = (
        (before_spectrum.freqs >= 80.0) & (before_spectrum.freqs <= 12_000.0)
    )
    repeatability_delta = (
        before_spectrum.median_spectrum_db[repeatability_mask]
        - original_shape[repeatability_mask]
    )
    repeatability_delta -= float(np.median(repeatability_delta))
    shape_delta = float(np.sqrt(np.mean(np.square(repeatability_delta))))
    original_level = _rms_db(original)
    verification_level = _rms_db(verification)
    before_features = _vad_masked_speech_features(
        verification,
        sample_rate,
        _rms_db(noise),
        noise_audio=noise,
    )
    after_features = _vad_masked_speech_features(
        rendered,
        sample_rate,
        _rms_db(rendered_noise),
        noise_audio=rendered_noise,
    )
    compressor = setup_result.get("compressor_settings") or {}
    target_p95 = float(compressor.get("target_p95_reduction_db", 3.5))
    peak_cap = float(compressor.get("peak_reduction_cap_db", 8.0))
    measured_p95 = float(
        processed.get("compressor_gain_reduction_p95_db", 120.0)
    )
    measured_peak = float(processed.get("compressor_gain_reduction_db", 120.0))
    output_true_peak = float(processed.get("output_true_peak_db", 120.0))
    ceiling = float(processed.get("limiter_effective_ceiling_db", -1.5))
    limiter_events = int(processed.get("true_peak_limited_events", 0))
    noise_change_db = _rms_db(rendered_noise) - _rms_db(noise)
    speech_gain_db = float(
        processed.get("output_rms_db", _rms_db(rendered))
    ) - float(processed.get("input_rms_db", _rms_db(verification)))
    relative_noise_change_db = noise_change_db - speech_gain_db
    snr_change_db = float(after_spectrum.snr_db - before_spectrum.snr_db)
    reasons: list[str] = []

    if abs(verification_level - original_level) > 8.0 or shape_delta > 5.0:
        decision = "retry"
        reasons.append("verification delivery differs too much from the setup passage")
    elif (
        after_error > before_error + 1.0
        or relative_noise_change_db > 4.0
        or snr_change_db < -4.0
        or measured_peak > peak_cap + 0.25
        or output_true_peak > ceiling + 0.15
    ):
        decision = "rollback"
        reasons.append("candidate chain worsened the target or exceeded a safety limit")
    elif (
        measured_p95 > target_p95 + 0.75
        or limiter_events > 0
        or relative_noise_change_db > 3.0
        or float(processed.get("deesser_gain_reduction_p95_db", 0.0))
        > float((setup_result.get("deesser_settings") or {}).get("max_reduction_db", 6.0))
        * 0.9
    ):
        decision = "reduce"
        reasons.append("processing is safe but stronger than the selected intensity")
    else:
        decision = "accept"
        reasons.append("repeatability and native-chain constraints passed")

    spectral_snr = after_spectrum.spectral_snr_db
    snr_bands: dict[str, float] = {}
    if spectral_snr is not None:
        for name, low, high in (
            ("low", 80.0, 250.0),
            ("body", 250.0, 1_000.0),
            ("presence", 1_000.0, 4_500.0),
            ("sibilance", 4_500.0, 10_000.0),
        ):
            mask = (after_spectrum.freqs >= low) & (after_spectrum.freqs < high)
            if np.any(mask):
                snr_bands[name] = float(np.median(spectral_snr[mask]))

    return {
        "decision": decision,
        "reasons": reasons,
        "perceptual_validation": False,
        "evidence_scope": "repeatability_and_exact_native_chain_constraints",
        "spectral_target_error_before_db": before_error,
        "spectral_target_error_after_db": after_error,
        "frequency_dependent_snr_db": snr_bands,
        "loudness_variation_before_db": float(
            before_features["active_loudness_spread_db"]
        ),
        "loudness_variation_after_db": float(
            after_features["active_loudness_spread_db"]
        ),
        "noise_floor_change_db": noise_change_db,
        "relative_noise_floor_change_db": relative_noise_change_db,
        "snr_change_db": snr_change_db,
        "compressor_gain_reduction_median_db": float(
            processed.get("compressor_gain_reduction_median_db", 0.0)
        ),
        "compressor_gain_reduction_p95_db": measured_p95,
        "compressor_gain_reduction_peak_db": measured_peak,
        "deesser_gain_reduction_median_db": float(
            processed.get("deesser_gain_reduction_median_db", 0.0)
        ),
        "deesser_gain_reduction_p95_db": float(
            processed.get("deesser_gain_reduction_p95_db", 0.0)
        ),
        "output_true_peak_db": output_true_peak,
        "limiter_activity_events": limiter_events,
        "clipped": bool(np.max(np.abs(rendered)) >= 1.0),
        "simulation_backend": processed.get("simulation_backend"),
    }


__all__ = [
    "DYNAMICS_PROFILES",
    "GATE_MODE_LABELS",
    "analyze_voice_setup",
    "validate_voice_setup_verification",
]
