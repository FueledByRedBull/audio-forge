"""Focused checks for Voice Setup's combined-chain verification."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from mic_eq.analysis import voice_setup
from mic_eq.config import EQ_FREQUENCIES, EQSettings, Preset, build_eq_candidate_settings


def _setup_result() -> dict:
    preset = Preset()
    return {
        "eq_settings": preset.eq.to_dict(),
        "gate_settings": asdict(preset.gate),
        "suppressor_settings": asdict(preset.rnnoise),
        "deesser_settings": asdict(preset.deesser),
        "compressor_settings": {
            **asdict(preset.compressor),
            "target_p95_reduction_db": 3.5,
            "peak_reduction_cap_db": 8.0,
        },
        "limiter_settings": asdict(preset.limiter),
        "verification_chain_settings": {
            "full_chain": True,
            "input_pre_filtered": False,
            "input_cleanup_mode": "gentle",
            "processing_mode": "normal",
            "gate": asdict(preset.gate),
            "suppressor": asdict(preset.rnnoise),
        },
    }


def _patch_measurements(monkeypatch, *, drop_voice: bool) -> list[dict]:
    calls: list[dict] = []

    class Spectrum:
        freqs = np.geomspace(80.0, 12_000.0, 16)
        median_spectrum_db = np.zeros(16)
        spectral_snr_db = np.zeros(16)
        snr_db = 0.0

    def simulate(audio, _sample_rate, eq_settings, chain):
        calls.append({"eq": eq_settings, "chain": dict(chain)})
        output = np.zeros_like(audio) if drop_voice and len(calls) == 1 else audio
        return {
            "simulation_backend": "rust",
            "output_audio": output.tolist(),
            "compressor_gain_reduction_p95_db": 0.0,
            "compressor_gain_reduction_db": 0.0,
            "deesser_gain_reduction_p95_db": 0.0,
            "output_true_peak_db": -3.0,
            "limiter_effective_ceiling_db": -1.5,
            "true_peak_limited_events": 0,
            "limiter_gain_reduction_db": 0.0,
            "true_peak_limiter_gain_reduction_db": 0.0,
            "output_rms_db": -26.0,
            "input_rms_db": -26.0,
        }

    monkeypatch.setattr(voice_setup, "simulate_candidate_chain", simulate)
    monkeypatch.setattr(voice_setup, "analyze_voice_spectrum", lambda *_args, **_kwargs: Spectrum())
    monkeypatch.setattr(voice_setup, "_shape_error_db", lambda *_args, **_kwargs: 0.0)
    def features(audio, *_args, **_kwargs):
        frame_db = -120.0 if not np.any(audio) else -26.0
        count = max(1, len(audio) // 960)
        return {
            "active_loudness_spread_db": 0.0,
            "frame_db": np.full(count, frame_db),
            "active_frame_mask": np.ones(count, dtype=bool),
        }

    monkeypatch.setattr(voice_setup, "_vad_masked_speech_features", features)
    return calls


def test_verification_renders_exact_full_chain_and_typed_eq(monkeypatch) -> None:
    calls = _patch_measurements(monkeypatch, drop_voice=False)
    sample_rate = 48_000
    noise = np.full(sample_rate * 2, 0.001, dtype=np.float32)
    original = np.full(sample_rate * 3, 0.04, dtype=np.float32)
    verification = np.full(sample_rate * 3, 0.04, dtype=np.float32)

    result = voice_setup.validate_voice_setup_verification(
        noise,
        original,
        verification,
        sample_rate,
        _setup_result(),
        "flat",
        raw_noise_audio=noise,
        raw_verification_speech_audio=verification,
    )

    assert result["decision"] == "accept"
    assert result["full_chain"] is True
    assert len(calls) == 2
    assert all(call["chain"]["full_chain"] is True for call in calls)
    assert all(call["chain"]["input_pre_filtered"] is False for call in calls)
    assert all(call["chain"]["input_cleanup_mode"] == "gentle" for call in calls)
    assert all(call["chain"]["processing_mode"] == "normal" for call in calls)
    assert all("bands" in call["eq"] for call in calls)


def test_verification_rejects_full_chain_voice_dropout(monkeypatch) -> None:
    _patch_measurements(monkeypatch, drop_voice=True)
    sample_rate = 48_000
    noise = np.full(sample_rate * 2, 0.001, dtype=np.float32)
    speech = np.full(sample_rate * 3, 0.04, dtype=np.float32)

    result = voice_setup.validate_voice_setup_verification(
        noise,
        speech,
        speech,
        sample_rate,
        _setup_result(),
        "flat",
        raw_noise_audio=noise,
        raw_verification_speech_audio=speech,
    )

    assert result["decision"] == "rollback"
    assert result["reason_codes"] == ["voice_gate_silencing"]
    assert result["full_chain_voice_dropout"] is True


def test_initial_analysis_renders_raw_full_chain_with_the_applied_eq_layers(monkeypatch):
    sample_rate = 48_000
    noise = np.random.default_rng(42).normal(0, 0.001, sample_rate * 2).astype(np.float32)
    t = np.arange(sample_rate * 4) / sample_rate
    speech = ((0.08 + 0.06 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    preset = Preset()
    incumbent_eq = preset.eq.to_dict()
    incumbent_eq["bands"][0]["gain_db"] = 2.5
    candidate_eq = {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [1.0] * len(EQ_FREQUENCIES),
        "band_qs": [1.41] * len(EQ_FREQUENCIES),
        "analysis_confidence": 0.9,
        "apply_recommended": True,
    }
    proposed_gate = {**asdict(preset.gate), "threshold_db": -36.0}
    proposed_suppressor = {"enabled": True, "strength": 0.65, "model": "deepfilter"}
    calls = []

    def simulate(audio, _sample_rate, eq_settings, chain):
        calls.append((audio, eq_settings, chain))
        return {
            "simulation_backend": "rust", "output_audio": audio,
            "output_true_peak_db": -3.0, "limiter_effective_ceiling_db": -1.5,
            "compressor_gain_reduction_db": 0.0, "compressor_gain_reduction_p95_db": 0.0,
            "deesser_gain_reduction_db": 0.0, "pre_limiter_true_peak_headroom_db": 2.0,
            "limiter_gain_reduction_db": 0.0, "true_peak_limiter_gain_reduction_db": 0.0,
        }

    # Leave capture measurement real; stub only the searches and final renderer.
    monkeypatch.setattr(voice_setup, "analyze_auto_eq", lambda *_a, **_k: (candidate_eq, {}))
    monkeypatch.setattr(voice_setup, "_calibrate_compressor_threshold",
                        lambda **kw: (kw["compressor_settings"], {"backend": "rust"}))
    monkeypatch.setattr(voice_setup, "tune_gate_suppression_dynamics", lambda *_a, **_k: {
        "apply_recommended": True, "gate_settings": proposed_gate,
        "suppressor_settings": proposed_suppressor,
    })
    monkeypatch.setattr(voice_setup, "simulate_candidate_chain", simulate)
    result = voice_setup.analyze_voice_setup(
        noise * 0.9, speech * 0.9, sample_rate, vad_available=False,
        incumbent_settings={"eq": incumbent_eq, "gate": asdict(preset.gate),
                            "rnnoise": asdict(preset.rnnoise)},
        raw_noise_audio=noise, raw_speech_audio=speech,
        input_cleanup_mode="gentle", processing_mode="bypass",
    )
    assert len(calls) == 1
    audio, eq, chain = calls[0]
    np.testing.assert_array_equal(audio, speech)
    assert chain["full_chain"] is True
    assert chain["input_pre_filtered"] is False
    assert chain["input_cleanup_mode"] == "gentle"
    assert chain["processing_mode"] == "bypass"
    assert chain["gate"] == proposed_gate
    assert chain["rnnoise"] == proposed_suppressor
    expected_eq = build_eq_candidate_settings(
        EQSettings.from_dict(incumbent_eq), candidate_eq["band_freqs"],
        candidate_eq["band_gains"], candidate_eq["band_qs"], layer="correction",
    ).to_dict()
    assert eq == expected_eq
    assert eq["layers"]["tone"][0]["gain_db"] == 0.0
    assert result["diagnostics"]["offline_validation_passed"] is True
    assert result["diagnostics"]["offline_validation_full_chain"] is True
    assert result["diagnostics"]["full_chain_raw_capture_used"] is True
